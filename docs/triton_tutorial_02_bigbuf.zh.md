# Triton 教程 02 — bigbuf fuse4 / split4

**目标：** 把 `fuse4_from_bigbuf_kernel` 和 `split4_to_bigbuf_kernel`（CUDA）移植到 Triton
**目标文件：** `src/fuilt/fusion_split/triton_bigbuf_ext.py`
**你来写：** ~6 行（fuse4）+ ~12 行（split4），在两个独立的 TODO 块里
**验证：** `python -m fuilt.fusion_split.triton_bigbuf_ext`

本教程建立在[教程 01](./triton_tutorial_01_pointwise.zh.md) 之上。如果你还没做过教程 01，回去先做 —— 教程 02 假设你理解了 `tl.program_id`、`tl.arange`、`tl.load(mask=)`、`tl.store(mask=)`。

教程 02 的新复杂度：
- **2D 坐标**（我们用 1D 索引再分解成 `(x, y)`）
- **4 个输入/输出区域**（A、B、C、D 子块）per kernel
- **算术 mask**（`(x < S) & (y >= P)` 等）—— 每个 output pixel 看到的子块集合不同
- **Scatter pattern**（split4 是 1 读 → 4 写）

---

## 第 0 部分 — 几何模型

这是最需要内化的东西。**请读两遍。**

我们有一个 big buffer（`in_big`，shape `[H, W]`）。里面有 4 个 sub-block A、B、C、D 散布在不同位置，加上一个输出区域 O。每个区域用左上角坐标标识：

```
   in_big[H, W]:
   ┌─────────────────────────────────────────────┐
   │                                              │
   │     ┌─────────┐         ┌─────────┐          │
   │     │   A     │         │   B     │          │
   │  (ax0,ay0)    │    (bx0,by0)     │          │
   │     └─────────┘         └─────────┘          │
   │                                              │
   │     ┌─────────┐         ┌─────────┐          │
   │     │   C     │         │   D     │          │
   │  (cx0,cy0)    │    (dx0,dy0)     │          │
   │     └─────────┘         └─────────┘          │
   │                                              │
   │                       ┌────────────────┐     │
   │                       │      O (out)   │     │
   │                       │  (ox0, oy0)    │     │
   │                       │  S_out × S_out │     │
   │                       └────────────────┘     │
   │                                              │
   └─────────────────────────────────────────────┘
```

每个 sub-block A/B/C/D 边长 `S`。它们以 overlap `O` 排列，相邻 sub-block 之间 stride 是 `P = S - O`。输出区域大小 `S_out × S_out`，其中 `S_out = S + P`。

### fuse4 覆盖图（output 本地坐标）

输出区域逻辑上分成 4 个象限。每个 output pixel `(x, y)`（output 本地坐标 `[0, S_out) × [0, S_out)`）被以下子块覆盖：

```
   输出区域（S_out × S_out）：
   ┌──────────────┬──────────────┐
   │              │              │       x ∈ [0, S)
   │  仅 A        │  A + B       │   且  x ∈ [P, P+S)
   │              │              │       都在 y ∈ [0, S) 内
   ├──────────────┼──────────────┤
   │              │              │
   │  A + C       │  A+B+C+D     │       y ∈ [P, P+S)
   │              │              │
   └──────────────┴──────────────┘
        x ∈ [0,S)      x ∈ [P, P+S)
```

- 左上象限：只有 A 覆盖
- 右上：A + B（sub-block B 相对 A 右移了 P）
- 左下：A + C（sub-block C 相对 A 下移了 P）
- 右下：A + B + C + D

这就是为什么 `mask_A` / `mask_B` / `mask_C` / `mask_D` 是独立计算的 —— 每个 output pixel 看到的子块集合不同。

### split4 scatter 图

split4 是 fuse4 的逆运算：读**一个** `S_out × S_out` 的 parent block，把它的 4 个象限 scatter 回 4 个 child sub-block。

对每个 child-local `(x, y) ∈ [0, S) × [0, S)`：

```
   child A[x, y] = parent[ix0 + x,     iy0 + y]       # 左上象限
   child B[x, y] = parent[ix0 + P + x, iy0 + y]       # 右上象限
   child C[x, y] = parent[ix0 + x,     iy0 + P + y]   # 左下
   child D[x, y] = parent[ix0 + P + x, iy0 + P + y]   # 右下
```

每个 thread 写 4 个 output pixel。

---

## 第 1 部分 — fuse4 kernel 步骤分解

kernel 概念上分 6 步。Step 1–5 **已经在框架里写好**。Step 6 是你的 TODO。

### Step 1 — 1D grid 分解

```python
pid = tl.program_id(0)
idx = pid * BLOCK + tl.arange(0, BLOCK)   # 1D 索引，覆盖 S_OUT*S_OUT output
```

我们用 1D grid，因为输出区域在 `(y * S_OUT + x)` 顺序下是连续的。每个 program 处理 `BLOCK` 个 output pixel。

### Step 2 — 1D → 2D 分解

```python
x = idx % S_OUT
y = idx // S_OUT
```

`x` 是列（水平），`y` 是行（竖直）。两者都是 BLOCK 长的 int32 向量。

### Step 3 — 全局 output 坐标

```python
gx = ox0 + x       # 全局列，在 [0, W) 内
gy = oy0 + y       # 全局行，在 [0, H) 内
```

### Step 4 — Output mask（边界处理）

```python
out_mask = (idx < n_out) & (gx < W) & (gy < H)
```

True 表示这个 output pixel 真实存在。注意 `idx < n_out` 处理 grid 尾部（最后一个 program 可能延伸到 `S_OUT * S_OUT` 之外）。

### Step 5 — Coverage mask + load

```python
mask_A = (x < S) & (y < S)
mask_B = (x >= P) & (x < P + S) & (y < S)
mask_C = (x < S) & (y >= P) & (y < P + S)
mask_D = (x >= P) & (x < P + S) & (y >= P) & (y < P + S)

a = tl.load(in_ptr + (ay0 + y) * W + (ax0 + x),         mask=mask_A, other=0.0)
b = tl.load(in_ptr + (by0 + y) * W + (bx0 + x - P),     mask=mask_B, other=0.0)
c = tl.load(in_ptr + (cy0 + y - P) * W + (cx0 + x),     mask=mask_C, other=0.0)
d = tl.load(in_ptr + (dy0 + y - P) * W + (dx0 + x - P), mask=mask_D, other=0.0)
```

到这里，`a / b / c / d` 都是 BLOCK 长的 fp32 向量。对应 mask 是 False 的位置，load 返回 `0.0`（即 `other=` 的值）。

**注意每个 sub-block 的地址算术**：
- A 在 `(ay0 + y, ax0 + x)` —— 直读
- B 在 `(by0 + y, bx0 + x - P)` —— B 相对 A 在 output 中右移了 `P`，所以当 output `x ∈ [P, P+S)` 时，B 的本地坐标是 `x - P`
- C 和 D 同理，只是 y 方向用 `y - P`

### Step 6 — Mean + store  （← **你来写**）

#### 这里要做什么

计算覆盖 sub-block 的平均值，store 时 cast 到 fp16。

#### 提示（按顺序读）

**提示 1 — 统计覆盖的 sub-block 数量。**
你有 4 个 bool mask。把每个转成 int32（0/1）再加起来：
```python
count = mask_A.to(tl.int32) + mask_B.to(tl.int32) + mask_C.to(tl.int32) + mask_D.to(tl.int32)
```

**提示 2 — 求和。**
mask 是 False 的位置 load 时返回了 `0.0`，所以被 mask 掉的 sub-block 对 sum 贡献为 0：
```python
total = a + b + c + d
```

**提示 3 — Mean + 除零保护。**
输出区域的角落只被 A 覆盖（count == 1），所以除零保护不是严格必须 —— 但右下角 count == 4，且边界 pixel 可能 count == 0（因为全局边界）。安全起见：
```python
mean = tl.where(count > 0, total / count, 0.0)
```

**提示 4 — Store 时 cast 到 fp16。**
匹配 CUDA kernel 硬编码的 fp16 输出：
```python
tl.store(out_ptr + gy * W + gx, mean.to(tl.float16), mask=out_mask)
```

<details>
<summary><b>提示 5 — fuse4 完整答案（卡住了再点开）</b></summary>

```python
count = mask_A.to(tl.int32) + mask_B.to(tl.int32) + mask_C.to(tl.int32) + mask_D.to(tl.int32)
total = a + b + c + d
mean  = tl.where(count > 0, total / count, 0.0)
tl.store(out_ptr + gy * W + gx, mean.to(tl.float16), mask=out_mask)
```

</details>

---

## 第 2 部分 — split4 kernel 步骤分解

split4 在结构上比 fuse4 简单（没有求平均，就是 4 次 copy），但每个 thread 的任务是反过来的（1 读 → 4 写）。

### Step 1–5（框架已写）

Step 1–4 和 fuse4 一样。Step 5 读 4 个 parent 值：

```python
pa_x = ix0 + x;     pa_y = iy0 + y
pb_x = ix0 + P + x; pb_y = iy0 + y
pc_x = ix0 + x;     pc_y = iy0 + P + y
pd_x = ix0 + P + x; pd_y = iy0 + P + y

pa_mask = in_child & (pa_x < W) & (pa_y < H)
pb_mask = in_child & (pb_x < W) & (pb_y < H)
pc_mask = in_child & (pc_x < W) & (pc_y < H)
pd_mask = in_child & (pd_x < W) & (pd_y < H)

va = tl.load(in_ptr + pa_y * W + pa_x, mask=pa_mask, other=0.0)
vb = tl.load(in_ptr + pb_y * W + pb_x, mask=pb_mask, other=0.0)
vc = tl.load(in_ptr + pc_y * W + pc_x, mask=pc_mask, other=0.0)
vd = tl.load(in_ptr + pd_y * W + pd_x, mask=pd_mask, other=0.0)
```

到这里你有 4 个 BLOCK 长的 fp32 向量，等待 scatter。

### Step 6 — Scatter 到 4 个 child 位置 （← **你来写**）

#### 提示

对每个 sub-block（A、B、C、D），做三件事：

**模板（4 个都用这个）：**
```python
oa_x = ax0 + x;     oa_y = ay0 + y
oa_addr = oa_y * W + oa_x
oa_mask = in_child & (oa_x < W) & (oa_y < H)
tl.store(out_ptr + oa_addr, va.to(tl.float16), mask=oa_mask)
```

对 B（bx0, by0, vb）、C（cx0, cy0, vc）、D（dx0, dy0, vd）重复。

<details>
<summary><b>split4 完整答案（卡住了再点开）</b></summary>

```python
# A
oa_x = ax0 + x; oa_y = ay0 + y
tl.store(out_ptr + oa_y * W + oa_x, va.to(tl.float16),
         mask=in_child & (oa_x < W) & (oa_y < H))
# B
ob_x = bx0 + x; ob_y = by0 + y
tl.store(out_ptr + ob_y * W + ob_x, vb.to(tl.float16),
         mask=in_child & (ob_x < W) & (ob_y < H))
# C
oc_x = cx0 + x; oc_y = cy0 + y
tl.store(out_ptr + oc_y * W + oc_x, vc.to(tl.float16),
         mask=in_child & (oc_x < W) & (oc_y < H))
# D
od_x = dx0 + x; od_y = dy0 + y
tl.store(out_ptr + od_y * W + od_x, vd.to(tl.float16),
         mask=in_child & (od_x < W) & (od_y < H))
```

</details>

---

## 第 3 部分 — 验证

两个 TODO 块都填完后：

```bash
python -m fuilt.fusion_split.triton_bigbuf_ext
```

期望输出：

```
============================================================
bigbuf Triton port vs CUDA reference (numerical equivalence)
============================================================
  kernel            : fuse4_from_bigbuf
  region            : 1984x1984
  max_abs_diff      : 0.0005         ← 期望 < 1e-3（fp16 cast 容差）
  mean_abs_diff     : 0.0001
  pass              : True
------------------------------------------------------------
  kernel            : split4_to_bigbuf
  max_abs_diff      : 0.0            ← 完全相等（纯 copy）
  mean_abs_diff     : 0.0
  pass              : True
------------------------------------------------------------
```

`atol=1e-3` 因为我们在比较时做了 fp32 → fp16 → fp32 的 cast。如果你看到更大 diff，检查 sub-block B/C/D 的地址算术（`- P` 和 `+ P` 偏移）。

---

## 第 4 部分 — 为什么这比教程 01 难

| 概念 | 教程 01（pointwise） | 教程 02（bigbuf） |
|---|---|---|
| Grid 维度 | 1D over flat array | 1D，但通过 `% S_OUT` 分解为 2D |
| 每 thread 输入 | 1 load | 4 个条件 load（不同地址） |
| 每 thread 输出 | 1 store | 1（fuse4）或 4（split4）store |
| Mask | 1 个边界 mask | 4 个覆盖 mask + 边界 mask |
| Math | 纯算术 | 算术 + count + 除零保护 |
| 访存模式 | 完美合并（contiguous offset） | gather/scatter（offset 依赖 x, y） |

好消息：教程做完后，你见过了**所有** production serving kernel 涉及的 Triton pattern（attention、paged KV cache、RoPE 等都是某种 scatter/gather 的变体）。教程 03（如果做的话）会讲 reduction —— `tl.reduce`、`tl.sum` over 某个轴 —— 这是 Triton 心智模型的最后一块拼图。

---

## 第 5 部分 — 接下来

两个 kernel 都落地、verify 通过之后：

1. **Benchmark Triton vs CUDA。** `verify_*_vs_cuda` 里的 micro-bench 是个起点；要做 pipeline 级对比，需要把 `triton_bigbuf` 路径接进 v8 host（教程 04 / 任务 D）。
2. **试不同 `BLOCK_SIZE`。** 默认 1024 处理 32×32 output tile / program。更大的 block（4096 = 64×64）减少 launch 数但增加寄存器压力。`@triton.autotune` 可以自动 sweep —— 见 `docs/triton_tutorial_01_pointwise.zh.md` 第 4 部分关于 autotune decorator 的介绍。
3. **反思 trade-off。** bigbuf 比 pointwise 复杂但仍完全在 Triton 能力范围内。CUDA 原版每个约 80 行；Triton port 每个约 30 行。和教程 01 是同样的教训：你用低层控制权换可移植性 + 可读性。
