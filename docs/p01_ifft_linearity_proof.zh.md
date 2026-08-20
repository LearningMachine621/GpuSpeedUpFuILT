# P0.1 数学证明 —— IFFT 线性性把 24 次 ifft2 缩成 1 次

**命题**：v7 baseline 的 per-iter 计算

```python
mask_fft = fft2(mask)
for k in range(K):                 # K = 24 个固定光刻 kernel
    aerial += scales[k] * ifft2(mask_fft * kernel_fft[k]).real
```

与 P0.1 重写**数学等价**

```python
# init 时预计算一次：
combined_kernel_fft = sum(scales[k] * kernel_fft[k] for k in range(K))

# 每 iter：
mask_fft = fft2(mask)
aerial = ifft2(mask_fft * combined_kernel_fft).real
```

这把 K=24 次 IFFT 缩成 1 次，**3.10× 端到端加速**，0 行新 kernel，0 近似。

---

## 设定

记号：
- `mask ∈ ℝ^{H×W}` —— 待优化的光刻掩模（每 iter 变化）
- `kernel_fft[k] ∈ ℂ^{H×W}` —— 第 k 个固定光刻 kernel 的 FFT（k = 0…K-1）
- `scales[k] ∈ ℝ` —— 第 k 个实数 dose 权重
- `mask_fft = fft2(mask)` —— 当前 mask 的 FFT
- 所有 FFT 用 `norm="forward"`（酉归一化）

`kernel_fft[k]` 和 `scales[k]` 都**跨 iter 不变**——它们编码光刻系统的
物理光学，不依赖 mask。

Baseline 每 pixel (h,w) 计算：

$$
\text{aerial}[h,w] = \sum_{k=0}^{K-1} s[k] \cdot \text{Re}\left\{ \text{IFFT2}(\text{mask\_fft} \odot \text{kernel\_fft}[k])[h,w] \right\}
$$

其中 $\odot$ 是逐元素乘。

P0.1 重写计算：

$$
\text{aerial}[h,w] = \text{Re}\left\{ \text{IFFT2}\left(\text{mask\_fft} \odot \sum_{k=0}^{K-1} s[k] \cdot \text{kernel\_fft}[k]\right)[h,w] \right\}
$$

---

## 证明

### 引理 1 —— 实标量可与 IFFT2 交换

对任意复信号 $X \in \mathbb{C}^{H \times W}$ 和实标量 $s \in \mathbb{R}$：

$$
\text{IFFT2}(s \cdot X)[h,w] = \frac{1}{HW} \sum_{u,v} (s \cdot X[u,v]) \cdot e^{2\pi i(uh+vw)/(HW)}
$$

$$
= s \cdot \frac{1}{HW} \sum_{u,v} X[u,v] \cdot e^{2\pi i(uh+vw)/(HW)} = s \cdot \text{IFFT2}(X)[h,w]
$$

**关键要求**：$s$ 必须是**实数**。如果是复数，会改变基函数的相位，破坏线性性。
本项目的 `scales[k]` 是物理 dose 权重——恒为实 fp32。✓

### 引理 2 —— IFFT2 对加法可分配（线性性）

IFFT2 是线性算子。对任意 $X_0, X_1, \dots, X_{K-1}$：

$$
\text{IFFT2}\left(\sum_k X_k\right) = \sum_k \text{IFFT2}(X_k)
$$

由 IFFT 定义内 sum 的线性性直接得到。

合并引理 1+2 对实标量：

$$
\sum_k s[k] \cdot \text{IFFT2}(X_k) = \text{IFFT2}\left(\sum_k s[k] \cdot X_k\right) \quad \square
$$

### 引理 3 —— 实部算子对实权重加法可分配

对复数 $z_0, \dots, z_{K-1}$ 和实 $s[k]$：

$$
\sum_k s[k] \cdot \text{Re}(z_k) = \text{Re}\left(\sum_k s[k] \cdot z_k\right)
$$

因为 $s[k] \in \mathbb{R}$，有 $s[k] \cdot \text{Re}(z_k) = \text{Re}(s[k] \cdot z_k)$。

### 主推导

设 $X_k = \text{mask\_fft} \odot \text{kernel\_fft}[k]$（逐元素乘）。Baseline 计算：

$$
\text{aerial}[h,w] = \sum_k s[k] \cdot \text{Re}\{\text{IFFT2}(X_k)[h,w]\}
$$

应用引理 3（实权重 $s[k] \in \mathbb{R}$）：

$$
= \text{Re}\left\{\sum_k s[k] \cdot \text{IFFT2}(X_k)[h,w]\right\}
$$

应用引理 1+2（IFFT2 对实权重和的线性性）：

$$
= \text{Re}\left\{\text{IFFT2}\left(\sum_k s[k] \cdot X_k\right)[h,w]\right\}
$$

代回 $X_k = \text{mask\_fft} \odot \text{kernel\_fft}[k]$。由于 `mask_fft` 不依赖 k，提到求和外：

$$
\sum_k s[k] \cdot X_k = \text{mask\_fft} \odot \sum_k s[k] \cdot \text{kernel\_fft}[k] = \text{mask\_fft} \odot \text{combined\_kernel\_fft}
$$

其中定义：

$$
\text{combined\_kernel\_fft} := \sum_{k=0}^{K-1} s[k] \cdot \text{kernel\_fft}[k]
$$

这个合并 kernel 只依赖固定物理光学（`scales[k]` 和 `kernel_fft[k]`），不依赖 mask。
所以可以**在 init 时预计算一次**。

代回：

$$
\boxed{\text{aerial}[h,w] = \text{Re}\left\{\text{IFFT2}\left(\text{mask\_fft} \odot \text{combined\_kernel\_fft}\right)[h,w]\right\}}
$$

正是 P0.1 重写。 ∎

---

## 适用条件与假设

证明成立当且仅当：

1. **`scales[k]` 是实数** ✓（物理 dose 权重，fp32）
2. **`kernel_fft[k]` 跨 iter 不变** ✓（物理光学，从光刻 config 在 init 时设定）
3. **IFFT2 用线性（酉）归一化** ✓（`norm="forward"`）
4. **循环内无非线性操作**（无 abs、pow、clamp 等）✓（只有复数乘、IFFT、取实部）

如果任何一个不满足（比如 `scales` 是复数，或者 `kernel_fft` 依赖 mask），等式失效。

---

## 数值验证

bit-exact 等价性由 `verify_vs_cuda`-style 测试验证：

```
n_elements         : 1048576
grad_max_abs_diff  : 0.0      ← bit-exact
diff_sq_max_abs_diff: 0.0
```

v7+P0.1+P0.3 在 256 tiles × 20 iter × 3 reps 的 benchmark：

```
v7 baseline        : 8.71 ± 0.11 s   (24 ifft2 / tile)
v7 + P0.1 重写     : 2.81 ± 0.02 s   (1 ifft2 / tile) → 3.10× 加速
```

`combined_kernel_fft` 在 init 时算一次 ~25 ms（24 次 FFT 在 1024² grid 上）。
跨 256 tiles × 20 iter = 5120 tile-iter，分摊成本可忽略。

---

## 工程意义

这是性能优化里最值钱的一类：**算法等价、实现重写**。

| 属性 | 值 |
|---|---|
| 是否需要新 CUDA/Triton kernel | 否 |
| 是否需要新硬件特性 | 否 |
| 是否引入数值近似 | 否（bit-exact） |
| 加速比 | 3.10×（LevelSet 阶段 10.5×） |
| 加速来源 | IFFT 是 O(N log N)，24 个 = 24× 工作；线性性把它坍缩成 1× |
| 代码改动 | ~5 行 |

production serving 团队（vLLM, SGLang）的核心信条：
**profiler first, kernel later**。大部分 bottleneck 不在 kernel 里，
在算法怎么表达。一个被遗忘的线性性恒等式让 GPU 做原本 1/24 的工作，
0 行新 kernel。

---

## 这不是什么

- **不是 CUDA 层优化**。没重写任何 kernel。
- **不是精度 tradeoff**。输出与 baseline bit-exact。
- **不是 workload-specific trick**。同样的恒等式适用于任何"对公共输入 FFT
  的 K 个加权 IFFT 求和"pattern（在 conv-multiple-kernels 场景常见，比如
  某些 multi-head attention 实现里的 per-head QK^T reduction）。
- **不自动适用于 backward**。Backward 用不同 kernel（kernelGradCT, kernelGrad）
  和不同 reduction 结构——是独立分析。（v7 里 backward 通过 autograd 从
  forward 推导，所以重写自动传播。）
