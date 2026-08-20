# Nsight Systems Trace Guide

How to generate, open, and analyze the project's `v7_p013_nvtx.nsys-rep`
trace file. Covers three access modes (GUI, command-line stats, headless
download) and explains the NVTX stage annotations.

---

## Why this trace exists

Every performance claim in this project's README is backed by either:

1. **v7's self-reported `time.time()` table** — fast, but the application
   measures itself, no third-party validation.
2. **Nsight Systems trace** — NVIDIA's official profiler; captures every
   CUDA kernel launch, NVTX range, and CUDA runtime API call. This is the
   production-grade proof.

The trace file (`outputs/nsys_traces/v7_p013_nvtx.nsys-rep`, ~11 MB) is the
"show your work" artifact. Anyone can reproduce it or download it to verify
the numbers themselves.

---

## 1. Generate the trace

### Quick repro

```bash
CUDA_VISIBLE_DEVICES=<isolated GPU> \
FUILT_TILES_DIR=/data/lyj/FuILT/tiles_from_patch \
FUILT_ITERS=5 FUILT_BENCH_WARMUP=2 FUILT_TILE_MULTIPLIER=16 \
FUILT_USE_COMBINED_KERNEL=1 FUILT_USE_ASYNC_D2H=1 FUILT_USE_NVTX=1 \
nsys profile -t cuda,nvtx,osrt --stats=false \
  -o outputs/nsys_traces/v7_p013_nvtx \
  python -m baseline.test_baseline_v7_fuselevelset --mode pytorch_fft
```

### Key env vars

| Var | Effect |
|---|---|
| `FUILT_USE_NVTX=1` | Wrap each pipeline stage in `torch.cuda.nvtx.range_push/pop` so the timeline shows colored bands. |
| `FUILT_USE_COMBINED_KERNEL=1` | Enable P0.1 (IFFT linearity rewrite, 3.10× speedup). |
| `FUILT_USE_ASYNC_D2H=1` | Enable P0.3 (pinned + non_blocking + overlap, 1.20× speedup). |
| `FUILT_ITERS=5` | Use 5 iters for trace (vs 20 for benchmark) to keep file size ~10 MB. |
| `FUILT_TILE_MULTIPLIER=16` | 256 tiles total (16 real × 16 cloned). Use 64 for 1024-tile batch scaling test. |

### Why `--stats=false`?

`nsys profile` can auto-run reports after capture. We disable it because:
- The output is verbose and goes to stdout, not a file.
- We want to choose reports later with `nsys stats --report <name>`.

---

## 2. Open the trace

### Option A: GUI via X11 forwarding (works, but slow)

If your SSH client supports X11 forwarding:

```bash
# On your local machine:
ssh -X lyj@<server>  # or -Y for trusted forwarding
sudo apt install libxcb-cursor0  # if you get the xcb plugin error
nsys-ui outputs/nsys_traces/v7_p013_nvtx.nsys-rep
```

**Caveat**: nsys-ui is an OpenGL app. Over X11 forwarding, every frame's
pixels travel over the network — expect single-digit FPS. Usable for
inspection, painful for navigation.

### Option B: Download to local machine (best experience)

```bash
# On your local machine:
scp lyj@<server>:/home/lyj/code/GpuSpeedUpFuILT/outputs/nsys_traces/v7_p013_nvtx.nsys-rep ./

# Install Nsight Systems locally (free):
#   macOS:   brew install --cask cuda
#   Windows: https://developer.nvidia.com/nsight-systems
#   Linux:   same package as server

nsys-ui v7_p013_nvtx.nsys-rep
```

11 MB transfers in seconds; the local GUI is fully responsive.

### Option C: Command-line stats only (no GUI needed)

This is what we use for the README numbers. Three useful reports:

```bash
# Top GPU kernels by total time
nsys stats --report cuda_gpu_kern_sum outputs/nsys_traces/v7_p013_nvtx.nsys-rep

# NVTX range tree (per-stage timing, what we annotated)
nsys stats --report nvtx_pushpop_trace outputs/nsys_traces/v7_p013_nvtx.nsys-rep

# CUDA runtime API summary (cudaMemcpy, cudaLaunchKernel, etc.)
nsys stats --report cuda_api_sum outputs/nsys_traces/v7_p013_nvtx.nsys-rep
```

---

## 3. What's in the trace

### NVTX stage annotations

The trace is annotated with 7 stage ranges (visible as colored bands in
nsys-ui):

| Stage | Color band | What it covers |
|---|---|---|
| `IO_Raster` | (outer) | OAS file → numpy mask read |
| `H2D_Init` | (outer) | CPU SDF compute + 256× H2D + kernel_fft build + JIT compile |
| `iter_NNN` | (per-iter outer) | One optimization iteration |
| ↳ `LevelSet` | (per-iter inner) | FFT forward + pointwise grad + loss |
| ↳ `Fuse` | (per-iter inner) | Per-tile grad → canvas accumulate |
| ↳ `Split` | (per-iter inner) | Canvas → per-tile next-iter init |
| ↳ `Apply` | (per-iter inner) | `params -= lr * grad` |
| `D2H_Save` | (outer) | Final result → CPU + ThreadPool write |

### Stage timing (5-iter trace)

| Stage | Mean (ms) | Std (ms) | Notes |
|---|---:|---:|---|
| IO_Raster | 177.34 | — | one-shot |
| H2D_Init | 6129.76 | — | one-shot, dominated by first-call JIT compile of `fused_pointwise_ext` |
| LevelSet | 52.98 | 13.32 | per-iter; std inflated by iter_000 cold start |
| Fuse | 3.10 | 0.06 | per-iter; extremely stable |
| Split | 3.60 | 0.02 | per-iter; extremely stable |
| Apply | 4.96 | 0.10 | per-iter; extremely stable |
| D2H_Save | 741.17 | — | one-shot |

> **Methodological note**: trace-time numbers differ from the headline
> benchmark numbers (v7+P0.1+P0.3 = 2.34s) because:
> 1. Trace uses 5 iters, not 20
> 2. First-call JIT compile is charged to H2D_Init in trace mode
> 3. NVTX recording has small per-event overhead
>
> Use the trace for **stage proportion analysis**, use the isolated 20-iter
> benchmark for **absolute latency claims**.

### Top GPU kernels (from `cuda_gpu_kern_sum`)

For the full kernel breakdown, see the "GPU kernel breakdown" section of
the main README. Summary: FFT chain dominates at ~50%, our optimized
`fused_pointwise_grad_loss` kernel is 0.3%.

---

## 4. Common pitfalls

### "qt.qpa.xcb: could not connect to display"

You're on a headless SSH session. Use one of:
- `ssh -X` (X11 forwarding)
- Download trace to local machine
- Use `nsys stats` command-line (Option C above)

### "libxcb-cursor0 needed"

```bash
sudo apt install libxcb-cursor0
```

Required by Qt 6.5+ (nsys-ui's GUI framework).

### "OpenGL version is too low"

Server has no GPU-accelerated OpenGL. Mesa software fallback will work but
be slow. For real OpenGL, use a local machine with discrete GPU.

### Trace file too large

- 5-iter trace: ~11 MB
- 20-iter trace: ~40 MB
- 1024-tile × 20-iter: ~200 MB

If file size matters (e.g., CI), reduce `FUILT_ITERS` or use `--cuda-um-profiling off`.

### NVTX ranges missing

Make sure:
- `FUILT_USE_NVTX=1` is set
- `nsys profile -t cuda,nvtx,osrt` includes `nvtx` in the `-t` list
- The Python process actually exercised the annotated code path (run
  `--mode pytorch_fft`, not `cuda_op` which uses a different path)

---

## 5. Reproducibility checklist

To verify this trace from scratch:

```bash
# 1. Confirm env
nvidia-smi  # GPU available
python -c "import torch; print(torch.__version__)"  # 2.11+
python -c "import triton; print(triton.__version__)"  # 3.6+

# 2. Generate trace (commands above)
# 3. Verify NVTX captured
nsys stats --report nvtx_pushpop_trace outputs/nsys_traces/v7_p013_nvtx.nsys-rep \
  | grep -c ":LevelSet"   # should be > 0
# 4. Verify kernel breakdown
nsys stats --report cuda_gpu_kern_sum outputs/nsys_traces/v7_p013_nvtx.nsys-rep \
  | head -20
```

If step 3 returns 0, NVTX env var wasn't set or the `-t nvtx` flag was missing.
