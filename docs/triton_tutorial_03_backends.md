# Triton Tutorial 03 — v8 Backend Ablation

**Goal:** Re-implement the three `BigbufBackend` subclasses from scratch as a
learning exercise.
**Reference solutions:** `src/fuilt/fusion_split/backends/`
**Ablation harness:** `baseline/test_baseline_v8_bigbuf_ablation.py`
**You write:** ~150 lines across 4 files (`base.py`, `pytorch_inplace.py`,
`cuda_bigbuf.py`, `triton_bigbuf.py`).

This tutorial builds on [Tutorial 01](./triton_tutorial_01_pointwise.md) and
[Tutorial 02](./triton_tutorial_02_bigbuf.md). It exercises a different skill:
**API design + adapter pattern**. Each backend wraps an existing implementation
into a common interface so they can be benchmarked head-to-head.

---

## Part 0 — Why this tutorial exists

The typical serving engineer workflow is:

1. Profile a pipeline, find a hot kernel
2. Try PyTorch eager → if too slow, try Triton → if Triton not enough, write CUDA
3. Benchmark all three with bit-exact equivalence checks
4. Pick the winner, justify with numbers

Tutorials 01 + 02 taught step 2 (writing the kernel). Tutorial 03 teaches
steps 1 + 3 + 4: building the ablation harness that lets you compare paths
fairly. This is the skill that turns "I wrote a fast kernel" into "I have
**evidence** my kernel is the right choice."

---

## Part 1 — The backend contract

All three backends implement `BigbufBackend`, defined in `base.py`:

```python
class BigbufBackend(ABC):
    name: str = "base"

    @abstractmethod
    def fuse4(self, in_big, out_big, offsets, S, O) -> torch.Tensor: ...

    @abstractmethod
    def split4(self, in_big, out_big, offsets, S, O) -> torch.Tensor: ...
```

The signatures match `fuse4_from_bigbuf` / `split4_to_bigbuf` from Tutorial 02
exactly. `offsets` is a `[5, 2]` int32 CPU tensor encoding the (x, y)
top-left corners of A, B, C, D, and the output/input region.

### Why abstract base class instead of duck typing?

Three reasons:
1. **Editor autocomplete** — Pyright/Pylance understand `BigbufBackend.fuse4`
   but not "any object with a `fuse4` method".
2. **Error messages** — if you forget to implement `split4`, you get a clean
   `TypeError: Can't instantiate abstract class` instead of a confusing
   `AttributeError` at runtime.
3. **Documentation** — the ABC *is* the contract. Future contributors read
   one file to understand the API.

### Why `name: str` as a class attribute?

So we can `print(backend)` and see "pytorch_inplace" instead of
"PytorchInplaceBackend at 0x7f...". Also useful for CSV output of the
ablation harness.

---

## Part 2 — Implementing PytorchInplaceBackend (the hard one)

This is the only backend with non-trivial logic. The other two are 1-line
wrappers.

### fuse4 in pure PyTorch

The CUDA/Triton kernel computes, for each output pixel `(x, y)` in the
`S_out × S_out` region:

```
acc[x, y]   = sum of sub-block values covering (x, y)
count[x, y] = number of sub-blocks covering (x, y)  (1, 2, or 4)
output[x, y] = acc / count
```

#### Step 1 — Allocate fp32 accumulator and count buffers

```python
acc = torch.zeros((S_out, S_out), dtype=torch.float32, device=in_big.device)
cnt = torch.zeros((S_out, S_out), dtype=torch.float32, device=in_big.device)
```

Why fp32 not fp16? Because averaging 4 fp16 values can lose precision. The
CUDA/Triton kernels also do this accumulation in fp32 internally.

#### Step 2 — Loop over the 4 sub-blocks, accumulating

The 4 sub-blocks have these quadrant positions in the output region:
- A: `[0:S, 0:S]`
- B: `[0:S, P:P+S]`
- C: `[P:P+S, 0:S]`
- D: `[P:P+S, P:P+S]`

where `P = S - O`.

For each sub-block, slice its source region in `in_big` and add to the
corresponding slice of `acc`. Same for `cnt` (just add 1.0):

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

#### Step 3 — Divide (with divide-by-zero guard) and store as fp16

```python
cnt[cnt == 0] = 1.0
acc.div_(cnt)
out_big[oy0:oy0 + S_out, ox0:ox0 + S_out] = acc.to(torch.float16)
```

### split4 in pure PyTorch

split4 is simpler — just 4 slice copies with fp16 cast:

```python
P = S - O
out_big[ay0:ay0 + S, ax0:ax0 + S] = in_big[iy0:iy0 + S,         ix0:ix0 + S].to(torch.float16)
out_big[by0:by0 + S, bx0:bx0 + S] = in_big[iy0:iy0 + S,         ix0 + P:ix0 + P + S].to(torch.float16)
out_big[cy0:cy0 + S, cx0:cx0 + S] = in_big[iy0 + P:iy0 + P + S, ix0:ix0 + S].to(torch.float16)
out_big[dy0:dy0 + S, dx0:dx0 + S] = in_big[iy0 + P:iy0 + P + S, ix0 + P:ix0 + P + S].to(torch.float16)
```

Note the **four different read offsets** in `in_big`: A reads the top-left
quadrant of the parent, B reads the top-right, C reads the bottom-left, D
reads the bottom-right. Each child writes to its own position in `out_big`.

---

## Part 3 — Implementing CudaBigbufBackend (trivial wrapper)

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

The whole class is ~10 lines. The `assert` is the only safety net — the
CUDA kernel would crash silently if you passed fp32 output (it has
`at::Half` hardcoded).

---

## Part 4 — Implementing TritonBigbufBackend (also trivial)

Same shape as CUDA, just imports from `triton_bigbuf_ext` (your Tutorial 02
output):

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

## Part 5 — The factory function

`__init__.py` exposes a single factory:

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

Why a factory instead of letting users call `PytorchInplaceBackend()` directly?

1. **String-based selection** — the ablation harness takes backend names from
   CLI args, not classes. Factory centralizes the name→class mapping.
2. **Lazy import** — if you don't run cuda_bigbuf, you don't pay its import
   cost (which triggers .so compilation).
3. **Future-proofing** — adding a new backend is one line in `mapping`,
   nothing else changes.

---

## Part 6 — The ablation harness

`baseline/test_baseline_v8_bigbuf_ablation.py` is ~200 lines. The interesting
parts:

### `_make_layout(S, O, H, W)`

Builds the `(fuse_offsets, split_offsets)` tensors. The layout must be
in-bounds of the bigbuf — see the `assert ix0 + S_out <= W` check.

### `_benchmark_backend(name, ...)`

For one backend, does:

1. **Warmup** (5 calls) — triggers Triton JIT compilation, primes caches.
2. **Time fuse4** for N reps, with `torch.cuda.synchronize()` before and
   after to get wall-clock time.
3. **`torch.cuda.reset_peak_memory_stats()` + max_memory_allocated()** —
   captures peak VRAM during the call.
4. **Numerical equivalence check** — runs `cuda_bigbuf` and the test
   backend on identical input, computes `max(abs(diff))`.

### Important details

- **Always benchmark `cuda_bigbuf` first** so its `.so` is compiled before
  any non-cuda backend's numerical-equivalence check needs it.
- **`torch.cuda.synchronize()` between warmup and timed reps** — without
  this, async CUDA streams make timings meaningless.
- **`torch.cuda.reset_peak_memory_stats()` before each kernel type** —
  otherwise fuse4's VRAM gets attributed to split4 too.

---

## Part 7 — Reading the results

From a re-run on RTX 4090 (GPU 7, isolated, 2026-08-04; **S=1024, O=64,
canvas H=W=4096, warmup=5, reps=50** — replaces an earlier 2026-08-03 run
whose CUDA number was anomalously slow and produced a bogus "Triton 22%
faster" conclusion):

| Backend | fuse4 µs | split4 µs | total µs |
|---|---:|---:|---:|
| cuda_bigbuf | 29.2 | 26.5 | 55.7 |
| pytorch_inplace | 179.8 (**6.2×** slower) | 101.8 (**3.8×** slower) | 281.6 |
| triton_bigbuf | 46.0 (**1.57×** slower) | 44.4 (**1.68×** slower) | 90.4 |

> **Correction note (2026-08-04, mechanism verified):** the previous table in
> this file reported `cuda_bigbuf` fuse4 = 60.1 µs and "Triton 22% faster".
> Re-measuring at the **same declared config** (S=1024, O=64, canvas H=W=4096)
> on GPU 7 isolated gives CUDA fuse4 = **29.0–29.2 µs** — reproduced at both
> `reps=50` (`benchmarks/results/v8_ablation.json`) and `reps=100` (the
> tutorial's original count; 29.0 / 29.1 µs across two runs), so the 60.1 µs
> was **not** a reps/canvas/config difference. It was a one-off anomalous
> measurement (the host was running the loaded 7/8-GPU 2026-08-03 E2E
> re-verification; a transient clock/contention state inflated only that one
> kernel's timed window). Triton fuse4 (46.9 → 46.0 µs) and split4 (25.4 →
> 26.5 µs) were stable across both sessions, so only the single CUDA-fuse4
> sample was off. The direction flips: **CUDA wins fuse4 ~1.57×** at this
> config. The takeaway is unchanged: the right backend is config-sensitive and
> must be measured, and for serving the graph-safety axis matters more than
> eager µs.

### What to look for in your own results

1. **Is `triton_bigbuf` within 10% of `cuda_bigbuf`?** If yes, the Triton
   port is a viable production replacement. (Here it's ~1.6× slower on
   eager fuse4 — shape-dependent.)
2. **Is `pytorch_inplace` within 2× of cuda on both kernels?** If yes, you
   may not need a custom kernel at all.
3. **Are diff_vs_cuda values all near 0?** Any non-zero value > 1e-3 means
   the backends disagree numerically — usually a bug in offsets or dtypes.

### Surprising findings from this specific ablation

- **CUDA wins `fuse4` ~1.57×** (29.2 vs 46.0 µs) at this canvas+S shape.
  Triton's vectorizer is not consistently better for the scatter + average
  pattern — the winner is kernel-shape-dependent.
- **Triton is ~1.68× slower than CUDA on `split4`** (44.4 vs 26.5 µs).
  The 4-slice copy favors the hand-written kernel.
- **Net eager: Triton is ~1.6× slower than CUDA end-to-end** (90.4 vs 55.7 µs
  total). This is *why* the README's v8/P1 story is **not** "Triton is faster
  in eager" — it's **graph-safety**: Triton enters CUDA Graphs by default,
  while the hand-written CUDA kernel needed the capture-stream fix (P2).
- **PyTorch fuse4 is ~6.2× slower, split4 ~3.8× slower than CUDA.** The
  accumulator + count-buffer + 4 `add_` pattern is exactly what eager PyTorch
  can't fuse.

These findings together justify the project's main narrative: **the right
backend depends on the kernel shape, and you have to measure to know** — and
for a *serving* deployment, the graph-compatibility axis matters more than
eager µs.

---

## Part 8 — What's next

Once you've re-implemented all four files and your ablation matches the
reference numbers (within ±20% noise), the next things to try:

1. **Add a 4th backend: `triton_autotuned`.** Wrap the same Triton kernel
   with `@triton.autotune` over `BLOCK_SIZE ∈ {256, 512, 1024, 2048, 4096}`
   and see if autotune finds a better config than the default 1024.
2. **Try a different shape.** What happens at S=2048, O=128? At S=256, O=32?
   The relative ranking may flip — that's an interesting story.
3. **Add NVTX markers.** Wrap each backend's call with
   `torch.cuda.nvtx.range_push/pop` so you can see the per-backend timeline
   in nsys-ui. This makes the ablation visualizable.
4. **Wire it into the v7 pipeline.** Replace v7's `StaticFusionSplitManager`
   calls with `get_backend("triton_bigbuf").fuse4(...)`. This is the
   integration test — does the Triton path produce the same final loss
   curve as v7's default?
