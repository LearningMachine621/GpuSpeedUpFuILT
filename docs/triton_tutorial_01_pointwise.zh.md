# Triton 教程 01 — Pointwise Kernel

**目标：** 把 `fused_pointwise_grad_loss_kernel`（CUDA）移植到 Triton
**目标文件：** `src/fuilt/ilt/func/triton_fused_pointwise.py`
**你来写：** 大约 5-6 行 math（TODO 块里）
**验证：** `python -m fuilt.ilt.func.triton_fused_pointwise`

这是一个 *pointwise* kernel：每个输出元素只依赖对应位置的输入元素。没有
reduction、没有 shared memory、没有 warp 级合作。这是最简单的一类 GPU
kernel —— 适合在打 bigbuf 的 scatter/gather 之前先建立 Triton 肌肉记忆。

---

## 第 0 部分 — 你要计算的 math

对每个元素 `i`：

```
z_i       = print_steepness * (aerial_i - target_density)
printed_i = sigmoid(z_i)
diff_i    = printed_i - target_i
grad_i    = 2 * diff_i * print_steepness * printed_i * (1 - printed_i)
diff_sq_i = diff_i ** 2
```

`grad_i` 是 L2 风格 loss `L = sum((printed - target)^2)` 对 `aerial_i` 的
梯度。链式法则给出 `2 * diff * print_steepness * printed * (1 - printed)`
的形式 —— 因为 `sigmoid'(z) = printed * (1 - printed)`。

就这些。五行。

---

## 第 1 部分 — Triton 的心智模型

Triton kernel 跑在一个 **grid of programs** 上。每个 program 通过
`tl.program_id(axis)` 拿到自己的 id，然后处理一块数据（tile）。

```
 ┌──────────────────────────────────────────────────────────┐
 │  Grid (1D, num_programs = cdiv(N, BLOCK_SIZE))            │
 │                                                            │
 │  program 0     program 1     program 2    ...  program P-1│
 │  [0..B)       [B..2B)       [2B..3B)            [P*B..N) │
 │                                                            │
 │  每个 program：                                             │
 │    1. 算自己负责哪些元素（offsets）                          │
 │    2. load 自己那块输入                                     │
 │    3. 做 math                                              │
 │    4. store 自己那块输出                                    │
 └──────────────────────────────────────────────────────────┘
```

**关键差别**：在一个 program 内部，**threads 由 Triton 编译器自动管理**
——不需要 `threadIdx.x`、不需要 warp shuffle、不需要手动向量化。你写的是
**block 级**代码：load 一个向量、对向量做 math、store 一个向量。

对比 CUDA：你要写 `idx = blockIdx.x * blockDim.x + threadIdx.x`、手动
处理向量化 load、手动 mask 尾部 thread、手动 launch 配置。Triton 用
`tl.load` / `tl.store` + `mask=` 把这些全藏起来了。

---

## 第 2 部分 — 五步走

### Step 1 — 找到自己的 program id

```python
pid = tl.program_id(0)   # 当前 program 在 axis 0 上的 id
```

`0` 表示第一个（也是我们这里唯一的）grid 轴。如果是 2D grid，还要调
`tl.program_id(1)` 拿第二个轴的 id。

### Step 2 — 算这个 program 负责哪些元素

```python
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask    = offsets < n_elements
```

`tl.arange(0, BLOCK_SIZE)` 相当于整个 block 的 `threadIdx.x` —— 一个向量
`[0, 1, 2, ..., BLOCK_SIZE-1]`。

`mask` 标识哪些 offset 是真实存在的。**最后一个 program** 可能负责超出
数组长度的位置，那些 offset 越界，每次 load 和 store 都必须跳过。

### Step 3 — Load 自己那一块输入

```python
a = tl.load(aerial_ptr + offsets, mask=mask, other=0.0)
t = tl.load(target_ptr + offsets, mask=mask, other=0.0)
```

`aerial_ptr + offsets` 是**指针 + 向量**的算术 —— Triton 理解这个语义，
会帮你生成向量化 load。结果是一个长度 `BLOCK_SIZE` 的 fp32 向量，存在
寄存器里（Triton 在寄存器不够时会自动 spill 到 shared memory）。

`other=0.0` 是被 mask 掉的位置填什么值。反正你不会写到那些位置，填什么
都行，`0.0` 是约定俗成。

到这里 `a` 和 `t` 是两个**长度 `BLOCK_SIZE` 的向量**，已经在寄存器里了。

### Step 4 — math （← **你来写**）

#### 这里要做什么

从 `a` 和 `t` 计算两个向量：

```
grad     = ...   # BLOCK_SIZE 长的 fp32 向量
diff_sq  = ...   # BLOCK_SIZE 长的 fp32 向量
```

#### math 再说一遍

```
z       = print_steepness * (a - target_density)
printed = sigmoid(z)
diff    = printed - t
grad    = 2 * diff * print_steepness * printed * (1 - printed)
diff_sq = diff * diff
```

#### 提示（按顺序读；不需要更多就停下）

**提示 1 — Triton op 是向量化的。**
`a - target_density` 能跑是因为 Triton 会把 Python 端标量 `target_density`
（这里是 `tl.constexpr`）自动广播到整个向量。`* print_steepness`、
`* 2.0` 等同理。

**提示 2 — Sigmoid。**
`tl.sigmoid(x)` 在 Ada/Hopper 上是单条 PTX 指令。直接用。
（手写 `1.0 / (1.0 + tl.exp(-x))` 也行，但是两条 op，fp32 下没好处。）

**提示 3 — 公共子表达式。**
`printed * (1 - printed)` 在 `grad` 里出现一次。单独存一个临时变量：
```python
sigmoid_deriv = printed * (1.0 - printed)
```
然后 `grad = 2.0 * diff * print_steepness * sigmoid_deriv`。
Triton 编译器 CSE 不错但不是魔法，**显式写更稳**。

**提示 4 — 最终形态。**
4-6 行。无循环、无 `tl.where`、无条件分支。纯算术。

<details>
<summary><b>提示 5 — 一个可行答案（卡住了再点开）</b></summary>

```python
z              = print_steepness * (a - target_density)
printed        = tl.sigmoid(z)
diff           = printed - t
sigmoid_deriv  = printed * (1.0 - printed)
grad           = 2.0 * diff * print_steepness * sigmoid_deriv
diff_sq        = diff * diff
```

</details>

### Step 5 — Store 自己那一块输出

把 TODO 下面两行的注释取消掉：

```python
tl.store(grad_ptr + offsets, grad, mask=mask)
tl.store(diff_sq_ptr + offsets, diff_sq, mask=mask)
```

`mask=mask` **必须有** —— 没有的话最后一个 program 会写出 output buffer
的末尾，破坏 GPU 内存里紧邻的其他数据。

---

## 第 3 部分 — 验证

填完 Step 4 + 取消两个 `tl.store` 的注释后：

```bash
python -m fuilt.ilt.func.triton_fused_pointwise
```

这会跑 `verify_vs_cuda()`，它会：

1. 编译原始 CUDA kernel 作为 oracle
2. 两个 kernel 在同一份 1M 元素的随机输入上跑
3. 逐元素对比输出

期望输出：

```
============================================================
Triton port vs CUDA reference (numerical equivalence)
============================================================
  n_elements            : 1048576
  block_size            : 1024
  grad_max_abs_diff     : 1.2e-06      ← 期望 < 1e-5
  diff_sq_max_abs_diff  : 0.8e-06      ← 期望 < 1e-5
  grad_pass             : True
  diff_sq_pass          : True
  overall_pass          : True
============================================================
PASS
```

如果 `FAIL`：检查两件事 —— (a) 两行 `tl.store` 都取消注释了；(b) `grad`
和 `diff_sq` 是你**最后算的两个**变量名（store 语句引用的变量名必须匹配）。

---

## 第 4 部分 — 为什么 Triton 版本"更好"

虽然两个 kernel 算的东西一模一样：

| 维度 | CUDA 原版 | Triton port |
|---|---|---|
| 代码行数 | 91 | ~20（kernel 本体） |
| 手写 `__device__` template 处理 | 有 | 无 |
| 手写 `<<<grid, block>>>` launch 计算 | 有 | 被 `[grid]` 语法隐藏 |
| 向量化 load/store | 手写 | 自动（`tl.load`） |
| 尾部 mask | 手写（`if (idx >= N) return;`） | 自动（`mask=`） |
| `target_density` / `print_steepness` 特化 | 否（runtime 参数） | 是（`tl.constexpr` → 每个值组合生成专门 kernel） |
| 可调 `BLOCK_SIZE` + autotune | 需要手写 switch | `@triton.autotune` 装饰器（这里没用，但加起来很容易） |
| 维护者上手成本 | 高（CUDA + ATen + pybind11） | 低（纯 Python） |

这正是 vLLM、SGLang 等基于 Triton 的 serving 栈把 hot kernel 从 raw CUDA
迁到 Triton 的同样理由。**你不损失性能** —— Triton 的 PTX 后端在 elementwise
和标准 reduction 上和手写 CUDA 持平 —— 但你拿到可移植性、可读性、以及
大得多的贡献者池。

---

## 第 5 部分 — 接下来

这个 kernel 落地之后：

1. **在 v7 的 CUDA Graph capture 内运行它。** Triton kernel 第一次调用
   时 JIT 编译（warmup 阶段就编完了），后续在 graph 内的调用只是 replay
   编译好的 binary。**不需要为 CUDA Graph 做任何特殊改动。**
2. **把它接进 v7。** 把 `baseline/test_baseline_v7_fuselevelset.py:179-180`
   里的 `ext.fused_pointwise_forward(...)` 替换成
   `fused_pointwise_forward_triton(...)`。跑一次 benchmark，和 CUDA 版对比。
3. **bigbuf 移植（教程 02）。** 同样的骨架，但 math 涉及 2D scatter/gather
   —— `tl.load`/`tl.store` 用计算出来的 offset，而不是简单的 `+ offsets`。
