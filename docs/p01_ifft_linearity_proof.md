# P0.1 Mathematical Proof — IFFT Linearity Reduces 24 ifft2 to 1

**Claim**: The v7 baseline's per-iter compute

```python
mask_fft = fft2(mask)
for k in range(K):                 # K = 24 fixed lithography kernels
    aerial += scales[k] * ifft2(mask_fft * kernel_fft[k]).real
```

is mathematically equivalent to the P0.1 rewrite

```python
# Pre-computed once at init:
combined_kernel_fft = sum(scales[k] * kernel_fft[k] for k in range(K))

# Per iter:
mask_fft = fft2(mask)
aerial = ifft2(mask_fft * combined_kernel_fft).real
```

This collapses K=24 IFFTs into 1, yielding **3.10× end-to-end speedup** with
zero new kernels and zero approximation (2026-08-04 rerun: 8.96 → 2.80 s = 3.20×,
`headline_e2e.json`; the numbers inside this doc are the original measurement).

---

## Setup

Notation:
- `mask ∈ ℝ^{H×W}` — the photo mask being optimized (changes per iter)
- `kernel_fft[k] ∈ ℂ^{H×W}` — the FFT of the k-th fixed lithography kernel (k = 0…K-1)
- `scales[k] ∈ ℝ` — the k-th real-valued dose weight
- `mask_fft = fft2(mask)` — the FFT of the current mask
- All FFTs use `norm="forward"` (unitary normalization)

Both `kernel_fft[k]` and `scales[k]` are **invariant across iterations** —
they encode the physical optics of the lithography system, not the mask.

The baseline computes, per pixel (h,w):

$$
\text{aerial}[h,w] = \sum_{k=0}^{K-1} s[k] \cdot \text{Re}\left\{ \text{IFFT2}(\text{mask\_fft} \odot \text{kernel\_fft}[k])[h,w] \right\}
$$

where $\odot$ is element-wise multiplication.

The P0.1 rewrite computes:

$$
\text{aerial}[h,w] = \text{Re}\left\{ \text{IFFT2}\left(\text{mask\_fft} \odot \sum_{k=0}^{K-1} s[k] \cdot \text{kernel\_fft}[k]\right)[h,w] \right\}
$$

---

## Proof

### Lemma 1 — Real scalars commute with IFFT2

For any complex signal $X \in \mathbb{C}^{H \times W}$ and real scalar $s \in \mathbb{R}$:

$$
\text{IFFT2}(s \cdot X)[h,w] = \frac{1}{HW} \sum_{u,v} (s \cdot X[u,v]) \cdot e^{2\pi i(uh+vw)/(HW)}
$$

$$
= s \cdot \frac{1}{HW} \sum_{u,v} X[u,v] \cdot e^{2\pi i(uh+vw)/(HW)} = s \cdot \text{IFFT2}(X)[h,w]
$$

**Key requirement**: $s$ must be **real**. A complex scalar would shift the
phase of each basis function and break linearity. In this project,
`scales[k]` are physical dose weights — always real fp32. ✓

### Lemma 2 — IFFT2 distributes over addition (linearity)

IFFT2 is a linear operator. For any $X_0, X_1, \dots, X_{K-1}$:

$$
\text{IFFT2}\left(\sum_k X_k\right) = \sum_k \text{IFFT2}(X_k)
$$

This follows directly from the linearity of the sum inside the IFFT definition.

Combining Lemmas 1+2 for real scalars:

$$
\sum_k s[k] \cdot \text{IFFT2}(X_k) = \text{IFFT2}\left(\sum_k s[k] \cdot X_k\right) \quad \square
$$

### Lemma 3 — Real-part operator distributes over real-weighted sums

For complex $z_0, \dots, z_{K-1}$ and real $s[k]$:

$$
\sum_k s[k] \cdot \text{Re}(z_k) = \text{Re}\left(\sum_k s[k] \cdot z_k\right)
$$

Because $s[k] \in \mathbb{R}$, we have $s[k] \cdot \text{Re}(z_k) = \text{Re}(s[k] \cdot z_k)$.

### Main derivation

Let $X_k = \text{mask\_fft} \odot \text{kernel\_fft}[k]$. The baseline computes:

$$
\text{aerial}[h,w] = \sum_k s[k] \cdot \text{Re}\{\text{IFFT2}(X_k)[h,w]\}
$$

Apply Lemma 3 (real weights $s[k] \in \mathbb{R}$):

$$
= \text{Re}\left\{\sum_k s[k] \cdot \text{IFFT2}(X_k)[h,w]\right\}
$$

Apply Lemma 1+2 (linearity of IFFT2 over real-weighted sums):

$$
= \text{Re}\left\{\text{IFFT2}\left(\sum_k s[k] \cdot X_k\right)[h,w]\right\}
$$

Substitute $X_k = \text{mask\_fft} \odot \text{kernel\_fft}[k]$. Since
`mask_fft` does not depend on k, factor it out:

$$
\sum_k s[k] \cdot X_k = \text{mask\_fft} \odot \sum_k s[k] \cdot \text{kernel\_fft}[k] = \text{mask\_fft} \odot \text{combined\_kernel\_fft}
$$

where we define:

$$
\text{combined\_kernel\_fft} := \sum_{k=0}^{K-1} s[k] \cdot \text{kernel\_fft}[k]
$$

This combined kernel depends only on the fixed physical optics
(`scales[k]` and `kernel_fft[k]`), not on the mask. So it can be
**precomputed once at init time**.

Substituting back:

$$
\boxed{\text{aerial}[h,w] = \text{Re}\left\{\text{IFFT2}\left(\text{mask\_fft} \odot \text{combined\_kernel\_fft}\right)[h,w]\right\}}
$$

which is exactly the P0.1 rewrite. ∎

---

## Conditions and assumptions

The proof holds iff:

1. **`scales[k]` are real-valued.** ✓ (physical dose weights, fp32)
2. **`kernel_fft[k]` is invariant across iterations.** ✓ (physical optics,
   set at init from lithography config)
3. **IFFT2 uses linear (unitary) normalization.** ✓ (`norm="forward"`)
4. **No non-linear operations inside the loop** (e.g. no `abs`, `pow`,
   `clamp`). ✓ (the only ops are complex multiply, IFFT, real-part)

If any of these broke (e.g. if `scales` were complex, or if `kernel_fft`
depended on the mask), the identity would fail.

---

## Numerical verification

> **2026-08-20 retraction + re-verification.** An earlier revision of this
> section cited `grad_max_abs_diff = 0.0` as proof of bit-exactness. That
> number came from the **Triton-vs-CUDA pointwise kernel** check
> (`triton_fused_pointwise.py::verify_vs_cuda`) — a different comparison that
> has nothing to do with the 24-loop-vs-combined rewrite. P0.1 itself had no
> dedicated numerical check; the "bit-exact" claim was a mis-attribution.

The dedicated check now exists:
[`baseline/verify_p01_equivalence.py`](../baseline/verify_p01_equivalence.py) —
independent of the pipeline, both branches actually executed, with a 1-ulp
perturbation sanity arm proving the comparator detects nonzero. Data mimics
the pipeline structure (K=24 hermitian kernel FFTs from real non-negative
PSFs, real dose scales, `norm="forward"`), 5 seeds, RTX 4090 / torch 2.11:

| dtype | max_abs_diff (5 seeds) | nonzero elements |
|---|---|---|
| complex64 (fp32) | 5.8e-10 … 3.7e-9 | ~80% of 1024² |
| complex128 (fp64) | 1.1e-18 … 6.9e-18 | ~80% |

Reading: **strictly equal in exact arithmetic** — the fp64 arm confirms the
residual is pure floating-point rounding. **Not bit-equal in fp32**: the two
paths sum in different domains (image vs frequency) with different rounding
orders and different FFT counts, so bit-equality is a per-data measurement at
best, never a theorem. Raw: `benchmarks/results/p01_equivalence.json`.

The v7+P0.1+P0.3 benchmark on 256 tiles × 20 iter × 3 reps gives:

```
v7 baseline        : 8.71 ± 0.11 s   (24 ifft2 per tile)
v7 + P0.1 rewrite  : 2.81 ± 0.02 s   (1 ifft2 per tile) → 3.10× speedup
```

The `combined_kernel_fft` is computed once at init in ~25 ms (24 FFTs on
1024² grids). Across 256 tiles × 20 iters = 5120 tile-iters, the amortized
cost is negligible.

---

## Engineering significance

This is the most valuable class of optimization: **algorithmic equivalence,
implementation rewrite**.

| Property | Value |
|---|---|
| New CUDA/Triton kernel required | No |
| New hardware feature required | No |
| Numerical approximation introduced | No — strictly equivalent; measured fp32 max_abs_diff ≤ 3.7e-9 (pure rounding) |
| Speedup | 3.10× (LevelSet stage 10.5×) |
| Source of speedup | IFFT is O(N log N); 24 of them is 24× the work, linearity collapses to 1× |
| Code change | ~5 lines |

The takeaway that production serving teams (vLLM, SGLang) live by:
**profiler first, kernel later**. Most bottlenecks aren't in the kernel,
they're in how the algorithm is expressed. A forgotten linearity identity
let the GPU do 1/24 of the work, with zero new kernels.

---

## What this is NOT

- **Not a CUDA-level optimization.** No kernel was rewritten.
- **Not a precision tradeoff.** It is a strict algebraic identity; on this hardware / PyTorch FFT the measured fp32 output differs from the 24-loop baseline by ≤ 3.7e-9 (rounding only, ~80% of elements).
- **Not a workload-specific trick.** The same identity applies to any
  "weighted sum of K IFFTs against a common input FFT" pattern (common
  in conv-multiple-kernels scenarios, e.g. multi-head attention's
  per-head QK^T reduction in some implementations).
- **Not automatically applicable to backward.** The backward pass uses
  different kernels (kernelGradCT, kernelGrad) and a different reduction
  structure — it's a separate analysis. (In v7 the backward is computed
  via autograd from the forward, so the rewrite propagates automatically.)
