# Triton 教程 03 — v8 Backend Ablation

**目标：** 从零重新实现三个 `BigbufBackend` 子类作为学习练习
**参考实现：** `src/fuilt/fusion_split/backends/`
**Ablation harness：** `baseline/test_baseline_v8_bigbuf_ablation.py`
**你来写：** 4 个文件约 150 行（`base.py`、`pytorch_inplace.py`、
`cuda_bigbuf.py`、`triton_bigbuf.py`）

本教程建立在[教程 01](./triton_tutorial_01_pointwise.zh.md) 和
[教程 02](./triton_tutorial_02_bigbuf.zh.md) 之上。它锻炼的是另一种技能：
**API 设计 + adapter 模式**。每个 backend 把已有实现包成统一接口，让它们
能公平 benchmark 对比。

---

## 第 0 部分 — 为什么有这个教程

serving 工程师的典型工作流是：

1. Profile 一个 pipeline，找到 hot kernel
2. 试 PyTorch eager → 太慢就试 Triton → Triton 还不够就写 CUDA
3. 三个 path 都 benchmark，附 bit-exact 等价检查
4. 选赢家，用数字 justify

教程 01 + 02 教了 step 2（怎么写 kernel）。教程 03 教 step 1 + 3 + 4：
搭一个能公平对比的 ablation harness。这是把"我写了个快 kernel"变成
"我有**证据**我的 kernel 是对的选择"的关键技能。

---

## 第 1 部分 — Backend 契约

三个 backend 都实现 `BigbufBackend`，定义在 `base.py`：

```python
class BigbufBackend(ABC):
    name: str = "base"

    @abstractmethod
    def fuse4(self, in_big, out_big, offsets, S, O) -> torch.Tensor: ...

    @abstractmethod
    def split4(self, in_big, out_big, offsets, S, O) -> torch.Tensor: ...
```

签名和教程 02 的 `fuse4_from_bigbuf` / `split4_to_bigbuf` 完全一致。
`offsets` 是 `[5, 2]` int32 CPU tensor，编码 A/B/C/D 四个子块和 output/input
区域的 (x, y) 左上角坐标。

### 为什么用抽象基类而不是 duck typing？

三个理由：
1. **编辑器自动补全** —— Pyright/Pylance 理解 `BigbufBackend.fuse4`，
   但不理解"任何带 `fuse4` 方法的对象"
2. **错误信息清晰** —— 忘记实现 `split4` 会得到干净的
   `TypeError: Can't instantiate abstract class`，而不是运行时迷惑的
   `AttributeError`
3. **文档** —— ABC *就是* 契约。未来的贡献者读一个文件就能理解 API

### 为什么 `name: str` 作为类属性？

这样可以 `print(backend)` 看到 "pytorch_inplace" 而不是
"PytorchInplaceBackend at 0x7f..."。对 ablation harness 输出 CSV 也有用。

---

## 第 2 部分 — 实现 PytorchInplaceBackend（最难的一个）

这是唯一一个有非平凡逻辑的 backend。另外两个都是 1 行 wrapper。

### 用纯 PyTorch 实现 fuse4

CUDA/Triton kernel 对 `S_out × S_out` 输出区域的每个 pixel `(x, y)` 计算：

```
acc[x, y]   = 覆盖 (x, y) 的 sub-block 值之和
count[x, y] = 覆盖 (x, y) 的 sub-block 数量（1、2 或 4）
output[x, y] = acc / count
```

#### Step 1 — 分配 fp32 accumulator 和 count buffer

```python
acc = torch.zeros((S_out, S_out), dtype=torch.float32, device=in_big.device)
cnt = torch.zeros((S_out, S_out), dtype=torch.float32, device=in_big.device)
```

为什么 fp32 不是 fp16？因为 4 个 fp16 值求平均会丢精度。CUDA/Triton
kernel 内部也是用 fp32 累加。

#### Step 2 — 循环 4 个 sub-block 累加

4 个 sub-block 在输出区域里有以下象限位置：
- A: `[0:S, 0:S]`
- B: `[0:S, P:P+S]`
- C: `[P:P+S, 0:S]`
- D: `[P:P+S, P:P+S]`

其中 `P = S - O`。

对每个 sub-block，slice 出 `in_big` 里的源区域，加到 `acc` 对应 slice。
`cnt` 同理（加 1.0）：

```python
P = S - O
for (sx, sy), (x0, y0) in [
    ((0, 0), (ax0, ay0)),
    ((P, 0), (bx0, by0)),
    ((0, P), (cx0, cy0)),
    ((P, P), (dx0, dy0)),
]:
    acc[sy:sy + S, sx:sx + S].add_(in_big[y0:y0 + S, x0:x0 + S].float())
    cnt[sy:sy + S, sx:sx + S] += 1.0
```

#### Step 3 — 相除（带除零保护）+ 存为 fp16

```python
cnt[cnt == 0] = 1.0
acc.div_(cnt)
out_big[oy0:oy0 + S_out, ox0:ox0 + S_out] = acc.to(torch.float16)
```

### 用纯 PyTorch 实现 split4

split4 更简单 —— 4 次 slice copy + fp16 cast：

```python
P = S - O
out_big[ay0:ay0 + S, ax0:ax0 + S] = in_big[iy0:iy0 + S,         ix0:ix0 + S].to(torch.float16)
out_big[by0:by0 + S, bx0:bx0 + S] = in_big[iy0:iy0 + S,         ix0 + P:ix0 + P + S].to(torch.float16)
out_big[cy0:cy0 + S, cx0:cx0 + S] = in_big[iy0 + P:iy0 + P + S, ix0:ix0 + S].to(torch.float16)
out_big[dy0:dy0 + S, dx0:dx0 + S] = in_big[iy0 + P:iy0 + P + S, ix0 + P:ix0 + P + S].to(torch.float16)
```

注意 `in_big` 的**四个不同读偏移**：A 读 parent 的左上象限，B 读右上，
C 读左下，D 读右下。每个 child 写到 `out_big` 里自己的位置。

---

## 第 3 部分 — 实现 CudaBigbufBackend（trivial wrapper）

```python
from ..fuse4_bigbuf_ext import fuse4_from_bigbuf as _fuse4_cuda
from ..split4_bigbuf_ext import split4_to_bigbuf as _split4_cuda

class CudaBigbufBackend(BigbufBackend):
    name = "cuda_bigbuf"

    def fuse4(self, in_big, out_big, offsets, S, O):
        assert out_big.dtype == torch.float16, "cuda_bigbuf requires fp16 output"
        return _fuse4_cuda(in_big, out_big, offsets, S, O)

    def split4(self, in_big, out_big, offsets, S, O):
        return _split4_cuda(in_big, out_big, offsets, S, O)
```

整个类约 10 行。`assert` 是唯一的安全网 —— CUDA kernel 在你传 fp32 output
时会静默崩溃（它硬编码了 `at::Half`）。

---

## 第 4 部分 — 实现 TritonBigbufBackend（也是 trivial）

形状和 CUDA 一样，只是从 `triton_bigbuf_ext`（你的教程 02 产物）导入：

```python
from ..triton_bigbuf_ext import (
    fuse4_from_bigbuf_triton as _fuse4_triton,
    split4_to_bigbuf_triton as _split4_triton,
)

class TritonBigbufBackend(BigbufBackend):
    name = "triton_bigbuf"

    def fuse4(self, in_big, out_big, offsets, S, O):
        assert out_big.dtype == torch.float16
        return _fuse4_triton(in_big, out_big, offsets, S, O)

    def split4(self, in_big, out_big, offsets, S, O):
        return _split4_triton(in_big, out_big, offsets, S, O)
```

---

## 第 5 部分 — Factory 函数

`__init__.py` 暴露一个 factory：

```python
def get_backend(name: str) -> BigbufBackend:
    mapping = {
        "pytorch_inplace": PytorchInplaceBackend,
        "cuda_bigbuf":     CudaBigbufBackend,
        "triton_bigbuf":   TritonBigbufBackend,
    }
    if name not in mapping:
        raise ValueError(f"unknown backend: {name!r}")
    return mapping[name]()
```

为什么用 factory 而不是让用户直接 `PytorchInplaceBackend()`？

1. **基于字符串选择** —— ablation harness 从 CLI 参数拿 backend 名字，
   不是类。Factory 集中 name→class 映射
2. **延迟导入** —— 不跑 cuda_bigbuf 就不付它的导入成本（会触发 .so 编译）
3. **未来扩展性** —— 加新 backend 只改 `mapping` 一行，其他不变

---

## 第 6 部分 — Ablation harness

`baseline/test_baseline_v8_bigbuf_ablation.py` 约 200 行。关键部分：

### `_make_layout(S, O, H, W)`

构造 `(fuse_offsets, split_offsets)` tensor。layout 必须在 bigbuf 范围内 ——
见 `assert ix0 + S_out <= W` 检查。

### `_benchmark_backend(name, ...)`

对一个 backend 做：

1. **Warmup**（5 次）—— 触发 Triton JIT 编译，预热 cache
2. **Timed fuse4** N 次，前后 `torch.cuda.synchronize()` 拿 wall-clock 时间
3. **`torch.cuda.reset_peak_memory_stats()` + max_memory_allocated()** ——
   捕获调用期间的峰值 VRAM
4. **数值等价检查** —— 对相同输入跑 `cuda_bigbuf` 和待测 backend，计算
   `max(abs(diff))`

### 重要细节

- **总是先 benchmark `cuda_bigbuf`** —— 它的 .so 必须先编译，不然其他
  backend 的数值等价检查会触发延迟编译
- **`torch.cuda.synchronize()` 在 warmup 和 timed reps 之间** —— 不加这个，
  异步 CUDA stream 会让计时无意义
- **`torch.cuda.reset_peak_memory_stats()` 在每个 kernel 类型之前** ——
  不然 fuse4 的 VRAM 会被算到 split4 上

---

## 第 7 部分 — 解读结果

最近一次 RTX 4090 跑（S=1024, O=64, reps=30）：

| Backend | fuse4 µs | split4 µs |
|---|---:|---:|
| cuda_bigbuf | 103.8 | 99.9 |
| pytorch_inplace | 294.8（**慢 2.83×**） | 121.1（慢 1.21×） |
| triton_bigbuf | **97.2**（**快 6%**） | 99.0（持平） |

### 自己跑的结果该看什么

1. **`triton_bigbuf` 是否在 `cuda_bigbuf` 的 10% 以内？** 是的话，Triton
   port 是可行的 production 替代
2. **`pytorch_inplace` 是否在两个 kernel 上都在 cuda 的 2× 以内？** 是的
   话，可能根本不需要 custom kernel
3. **diff_vs_cuda 值是否都接近 0？** 任何 > 1e-3 的值意味着 backend 之间
   数值不一致 —— 通常是 offset 或 dtype bug

### 这个 ablation 的反直觉发现

- **Triton 在 fuse4 上比 CUDA 快 6%。** 这不常见 —— Triton 的 vectorizer
  在 scatter + average pattern 上恰好找到了比手写 kernel 更优的指令序列
- **PyTorch split4 只比 CUDA 慢 1.21×。** Slicing + copy 在 PyTorch 里
  已经接近最优，custom kernel 帮助不大
- **PyTorch fuse4 慢 2.83×。** accumulator + count buffer + 4 次 add_
  这种 pattern 正是 eager PyTorch 没 fuse 的

这三个发现合起来支撑了项目的主叙事：**正确的 backend 取决于 kernel
shape，必须测了才知道。**

---

## 第 8 部分 — 接下来

重新实现完所有 4 个文件，且 ablation 数字匹配参考（±20% 噪声范围内）后，
下一步可以试：

1. **加第 4 个 backend：`triton_autotuned`。** 给同样的 Triton kernel
   加 `@triton.autotune`，sweep `BLOCK_SIZE ∈ {256, 512, 1024, 2048, 4096}`，
   看 autotune 能否找到比默认 1024 更好的配置
2. **试不同 shape。** S=2048、O=128 会怎样？S=256、O=32 呢？相对排名可能
   翻转 —— 那是个有趣的故事
3. **加 NVTX 标注。** 在每个 backend 调用前后包
   `torch.cuda.nvtx.range_push/pop`，能在 nsys-ui 里看到 per-backend
   timeline。让 ablation 可视化
4. **接入 v7 主线。** 用 `get_backend("triton_bigbuf").fuse4(...)` 替换
   v7 的 `StaticFusionSplitManager` 调用。这是 integration test ——
   Triton 路径能否产出和 v7 默认相同的最终 loss 曲线？
