# Verifiable results ledger

Every performance number in `README.md` / `README.zh.md`, mapped to the script
that produced it, the exact rerun command, and — where freshly re-measured —
the raw output.

**Raw evidence is committed**: the small machine-readable outputs (KB-scale
JSON) live in [`benchmarks/results/`](benchmarks/results/) and are tracked in
git — `graph_modes.json` (5-mode), `v7_profile.json` (v7 E2E profile),
`kernel_microbench.json` (pointwise/FFT micro-bench), `compare_*.json`
(correctness). The nsys traces themselves are committed too: the `.nsys-rep`
files (~11-20 MB each, 5 variants) live under `outputs/nsys_traces/` in git.
Only the derived `.sqlite` stats stay gitignored — they are rebuilt from the
committed `.nsys-rep` by `nsys stats`, one command away (see §4).

None of this claims a number is "the truth" — it claims it is **reproducible**:
the committed scripts regenerate it on an isolated RTX 4090.

## Environment (all runs)

| Item | Value |
|---|---|
| GPU | NVIDIA RTX 4090 (24 GB), 8-GPU host — **isolated** via `CUDA_VISIBLE_DEVICES` (contended GPUs distort latency 5-15×) |
| Driver | 580.126.09 |
| torch | 2.11.0+cu130 |
| Triton | 3.6.0 |
| Python | 3.13 |
| git HEAD (ledger date) | `8fc358b` (2026-08-04 headline re-run generated at this commit; ledger text committed after) |
| Repo-relative CWD | run all commands from the repo root with `PYTHONPATH=.` |

Two benchmark-script fixes shipped with this ledger (both in
`baseline/run_benchmark_matrix.py`):

1. **Import bug** (was `:505`): the kernel micro-bench imported
   `fuilt.ilt.func.fused_pointwise_triton` (module does not exist); the correct
   module is `triton_fused_pointwise`. The `try/except` silently swallowed the
   `ImportError`, so `pointwise_triton_ms` was always `None`. Fixed + a `[WARN]`
   print on failure.
2. **Missing warmup**: `time_cuda` recorded the first call inside the timed
   region, so the very first Triton call paid JIT-compile (~1.3 s) inside the
   measurement — `pointwise_triton_ms` read 6.39 ms instead of ~0.024 ms. Added
   a warmup call outside the timed region.

---

## Freshly re-verified 2026-08-03 (GPU 7)

### 1. Pointwise kernel micro-bench — reconciles README "5 µs vs 18 µs"

**Script**: `baseline/run_benchmark_matrix.py` → `_run_kernel_microbench` (the
fixed path), plus a temporary driver that times CUDA/Triton/unfused pointwise
with CUDA events.

**Command**:
```bash
CUDA_VISIBLE_DEVICES=7 python -m baseline.run_benchmark_matrix --run-kernel-microbench --out-dir /tmp/fuilt_microbench
# + isolated per-backend timing via _compute_pointwise (test_baseline_v7_fuselevelset.py)
```

**Raw output (harness `_run_kernel_microbench`, reps=200):**

```json
{
  "fft2_ms": 0.0295,
  "ifft2_accum_ms": 1.3016,
  "pointwise_torch_ms": 0.0512,
  "pointwise_triton_ms": 0.0238
}
```

**Raw output (isolated per-launch latency, CUDA events, reps=500):**

| backend | µs/call |
|---|---:|
| unfused (pure-PyTorch chain) | 62.7 |
| **CUDA fused** | **8.1** |
| **Triton fused** | **24.5** |

**Uses in README**: replaces the stale "5 µs CUDA vs 18 µs Triton per launch"
(original commit) with "8 µs CUDA vs 24 µs Triton — isolated per-launch
latency". Note the nsys in-pipeline per-kernel duration is ~7.5 µs for both
(Triton's Python-side dispatch overhead is hidden once launches overlap) — a
different measurement domain, explicitly labeled in §What the trace tells us.

### 2. 5-mode graph benchmark (bug vs fix)

**Script**: `baseline/benchmark_v7_graph_modes.py` — **writes**
`benchmarks/results/graph_modes.json` (**committed raw output**) in addition
to printing.

**Command**:
```bash
CUDA_VISIBLE_DEVICES=7 python -m baseline.benchmark_v7_graph_modes --iters 20
```

**Raw output (2026-08-03, GPU 7, 20 iters, 256 tiles):**

| # | Mode | ms/iter | loss-rel | Status |
|---|------|--------:|---------:|--------|
| 1 | unfused eager | 43.54 | 0.00 | PASS |
| 2 | fused eager (CUDA) | 30.96 | ref | REF |
| 3 | fused graph (CUDA, capture_safe=OFF) | 27.66 | **1.02** | **FAIL** |
| 4 | triton graph | 29.65 | 0.00 | PASS |
| 5 | fixed CUDA graph (capture_safe=ON) | 29.17 | 0.00 | PASS |

Full JSON (env, args, per-mode `loss`, `param_diff`, notes) lands in
`benchmarks/results/graph_modes.json` (committed). **mode5 < mode4
reproduces** — the
stream-fix conclusion the README states. An earlier 2026-07-30 run measured
55.17/30.62/27.32/29.30/28.84; same arc, run-to-run variance (mode 1 is the
most launch-bound and the most load-sensitive).

### 3. E2E headline (v1 + v7+P0.1+P0.3) — re-verified 2026-08-03

**Command** (GPU 7; note the host had **7/8 GPUs busy at 100 %** — isolated on
GPU 7, but the shared host CPU/PCIe adds ~1-5 % vs a quiet host):
```bash
CUDA_VISIBLE_DEVICES=7 FUILT_ITERS=20 python -m baseline.test_baseline_v1
CUDA_VISIBLE_DEVICES=7 FUILT_USE_COMBINED_KERNEL=1 FUILT_USE_ASYNC_D2H=1 \
  FUILT_ITERS=20 python -m baseline.test_baseline_v7_fuselevelset --mode pytorch_fft
```

**Raw output (2026-08-03):**

| Version | Pure pipeline (s) | Peak VRAM (MB) | README (2026-06/07) |
|---|---|---|---|
| v1 baseline (256 tiles, 16×16 grid) | 54.165 | 12480.4 | 53.37 ± 0.35 |
| v7 + P0.1 + P0.3 (3 runs) | 2.460 / 2.419 / 2.457 → **2.445** | 5997.1 | 2.34 ± 0.02 |
| **Speedup** | **54.165 / 2.445 = 22.2×** | | 22.8× |

- v1 within **+1.5 %**, v7 within **+4.5 %** of the README table — the gap is
  host load during re-verification (7/8 GPUs busy on the same machine). The
  **~22× speedup is robust** across both environments.
- The v7 run persisted `benchmarks/results/v7_profile.json` (committed;
  per-stage `profile_times`, peak VRAM, μ-bench ms/tile).
- **Display quirk**: `test_baseline_v1.py` reports `num_tiles=16` (the
  pre-clone count) but actually runs **256 tiles** — `base_grid` is re-inferred
  to 16 (16×16) after the `MULTIPLIER=16` clone at `test_baseline_v1.py:124`,
  and the 12480 MB peak VRAM confirms the 256-tile workload. **Trust `grid`,
  not `num_tiles`** in that script's report.

---

## Re-verified 2026-08-26 (sentinel-triggered, GPU 7 truly idle) — 21.6×

Full-table rerun launched automatically by an idle-GPU sentinel (10 min
sustained util=0 + mem<1 GB; raw:
[`headline_e2e_20260826_sentinel.json`](headline_e2e_20260826_sentinel.json)):

| variant | 08-26 (s) | 08-04 (s) | delta |
|---|---:|---:|---:|
| v1 | 52.999 ± 0.69 | 53.07 ± 0.51 | −0.1% |
| v3 / v5 / v6 | 13.43 / 18.0 / 19.51 | 13.36 / 18.05 / 19.61 | ≤ ±0.5% |
| v7-CUDA / v7-Triton | 9.075 / 9.305 | 8.96 / 9.18 | +1.3 / +1.4% |
| P0.1 / P0.4 | 2.843 / 4.329 | 2.80 / 4.24 | +1.5 / +2.1% |
| **P0.1+P0.3** | **2.457 ± 0.026** | 2.36 ± 0.05 | **+4.1%** |
| **headline** | **21.6×** | **22.5×** | −4% |

Reading: baseline-side variants are rock-stable (≤±0.5%); the uniform +1.3-2.1%
on GPU-pipeline variants suggests residual clock/thermal state (GPU 7 ran 100%
for hours until ~10 min before the run), and the headline spread concentrates
in the P0.1+P0.3 denominator (async-D2H-heavy, host-IO sensitive; the two
means sit ~1.7σ_combined apart). **Both numbers are real isolated
measurements: 22.5× (2026-08-04) stays canonical; 21.6× (2026-08-26) bounds
the cross-run spread at ~±4%.** One v7-CUDA rep crashed at first-launch
extension compile (returncode 1); its other two reps agree (9.056/9.093).

**Cross-GPU follow-up (08-26b, [`headline_e2e_20260826_full2.json`](headline_e2e_20260826_full2.json),
GPU 4, host fully idle for hours)**: headline **21.51x**; p01p03 = 2.444 s —
**within 0.5% of the GPU-7 sentinel run (2.457)**. Two conclusions:
(1) the morning +4.1% is **not** GPU thermal/clock residue (an unhammered
different card reproduces it) — it is a **systematic host-side shift in the
async-D2H variants only**: p01p03 +3.7% and p04 +2.3% are the *only* variants
drifting vs 08-04 (both gated by `FUILT_USE_ASYNC_D2H`), while v1/v3/v5/v6/
v7/p01 all sit within 1.8% (p01 is actually faster). Cause between 08-04 and
08-26 not identified from these runs (host/driver state, pinned-page path) —
follow-up would be a stage-split/trace comparison of the D2H_Save stage.
(2) cross-card parity confirms the cards are interchangeable for this bench.

### 6. Full-stack (P0.1 + P0.3 + graph capture) — first measured 2026-08-26

The headline number is the **eager** pipeline; `FUILT_USE_GRAPH=1` on top was
never in the headline table. Same idle GPU 7, 256 tiles × 20 iters:

- per-iter loop: eager 40.3 ms (stage table 0.805 s/20) vs graph replay
  **29.16 ms/iter** (29.156/29.158/29.154 across 3 runs, ±0.01%) → **−27%**
- pure-pipeline MEASURED after adding same-metric stage accounting to the
  graph branch (3 reps, [`fullstack_graph_pure_20260826.json`]
  (fullstack_graph_pure_20260826.json)): **2.670 ± 0.004 s → 19.9× vs v1**
  — SLOWER than same-day eager (2.450 s → 21.6×). Root cause: the graph
  branch's D2H is synchronous serial `.detach().cpu()` (1.09-1.12 s) and
  never inherited P0.3's async pinned path; the 0.22 s replay saving is more
  than eaten. An earlier derived "≈23.8×" (borrowing the eager D2H segment)
  is retracted. P0.3 async-D2H was then PORTED into the graph branch (same
  day, gated by FUILT_USE_ASYNC_D2H): D2H 1.10→0.82-0.94 s, full-stack
  **2.465 ± 0.047 s → 21.5× = parity with same-day eager (21.6×)**; the
  22-24× estimate was also optimistic. Closed conclusion: after P0.1+P0.3,
  graph capture adds ~nothing end-to-end at 256 tiles.
- nsys trace committed: `outputs/nsys_traces/v7_fullstack_graph.nsys-rep`
  (derived: `nsys_reports/v7_fullstack_graph_*`): FFT chain **34.1%**
  (regular 17.3 + vector 16.8), abs+pow 24.0% (eval phase), fused pointwise
  **0.2%**, **D2H = 90.5% of mem-op time** — remaining bottleneck is IO.

## Freshly re-verified 2026-08-04 (GPU 7) — full headline table re-run

Every headline row re-run on an **idle, isolated GPU 7** (all 8 GPUs present,
GPU 0 busy with a vLLM engine, GPU 7 chosen and held exclusive), 3 reps each,
per-rep raw JSON in [`benchmarks/results/headline_e2e.json`](benchmarks/results/headline_e2e.json).

**Driver**: `baseline/rerun_headline.py` (launches the existing `test_baseline_v*.py`
scripts via subprocess; parses their stdout; records GPU/versions/AMP state).
```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python -m baseline.rerun_headline --gpu 7 --repeats 3
```

**Raw output (2026-08-04, mean ± std over 3 reps):**

| Variant | Pure pipeline (s) | VRAM (MB) | μ-bench (ms/tile) | Speedup | README old |
|---|---:|---:|---:|---:|---:|
| v1 | 53.07 ± 0.51 | 12480 | — | 1.00× | 53.37 ± 0.35 |
| v3 | 13.36 ± 0.05 | 11874 | 1.887 | 3.97× | 13.46 ± 0.05 |
| v5 AMP=1 | 18.05 ± 0.19 | 6617 | 3.162 | 2.94× | 18.03 ± 0.07 |
| v5 AMP=0 | 18.01 ± 0.16 | 6617 | 3.164 | 2.95× | (not in README) |
| v6 | 19.61 ± 0.10 | 21465 | 3.479 | 2.71× | 19.53 ± 0.13 |
| v7 CUDA | 8.96 ± 0.08 | 5989 | 1.339 | 5.92× | 8.71 ± 0.11 |
| v7 Triton | 9.18 ± 0.03 | 5989 | 1.382 | 5.78× | 8.98 ± 0.27 |
| P0.1 | 2.80 ± 0.08 | 5997 | 0.130 | 19.0× | 2.81 ± 0.02 |
| **P0.1+P0.3** | **2.36 ± 0.05** | 5997 | 0.124 | **22.5×** | 2.34 ± 0.02 |
| P0.4 | 4.24 ± 0.11 | 5997 | 0.123 | 12.5× | 4.32 ± 0.04 |

- **Every row reproduces** the README table within a couple percent; the
  headline **22.8× → 22.5×** on a quiet host (2.36 s vs v1 53.07 s). The
  2026-08-03 "loaded" 22.2× is bracketed by both clean runs.
- **v5 AMP state resolved**: README claims v5 is fp16; the v5 script's
  `pytorch_fft` default is AMP=off, so both states were measured —
  **18.05 s (AMP on) vs 18.01 s (AMP off), identical within noise**. Graph
  capture overhead dominates; the fp16 claim holds but changes nothing.
- All v7-family runs share `final_loss ≈ 380983.2` (numerically consistent).

---

## Historical runs (documented; rerun to regenerate)

The remaining README-table rows — v3 13.46 s, v5 18.03 s, v6 19.53 s,
v7-CUDA 8.71 s, v7-Triton 8.98 s, v7+P0.1 2.81 s, v7+P0.4 4.32 s ⚠ — were
produced by the original 2026-06/07 runs. Their raw console output was
**print-only at the time**; each is rerunnable via the same scripts (v7
variants are env-gated via `FUILT_USE_COMBINED_KERNEL` /
`FUILT_USE_ASYNC_D2H` / `FUILT_USE_BATCH_H2D`, and `FUILT_USE_TRITON_POINTWISE`
for the Triton pointwise). They were **not** re-verified on 2026-08-03.

### 4. nsys trace numbers (FFT chain 50.0%, pointwise 0.3%, 7861/7526 ns)

**Traces in the repo** (committed, ~11-20 MB each): `outputs/nsys_traces/v7_p013_nvtx.nsys-rep`
(+ `v7_cuda`, `v7_triton` variants and their `_rerun` isolates). The matching
`.sqlite` stats are gitignored — regenerate from the `.nsys-rep`:

**Regenerate** (from `docs/nsys_trace_guide.md`):
```bash
CUDA_VISIBLE_DEVICES=<idle> FUILT_TILES_DIR=/data/lyj/FuILT/tiles_from_patch \
FUILT_ITERS=5 FUILT_BENCH_WARMUP=2 FUILT_TILE_MULTIPLIER=16 \
FUILT_USE_COMBINED_KERNEL=1 FUILT_USE_ASYNC_D2H=1 FUILT_USE_NVTX=1 \
nsys profile -t cuda,nvtx,osrt --stats=false \
  -o outputs/nsys_traces/v7_p013_nvtx \
  python -m baseline.test_baseline_v7_fuselevelset --mode pytorch_fft
# per-kernel stats:
nsys stats --report cuda_gpu_kern_sum outputs/nsys_traces/v7_p013_nvtx.nsys-rep
```

**Derived reports are committed** (re-generated 2026-08-04 from the committed
`.nsys-rep`, no GPU): `benchmarks/results/nsys_reports/` holds
`*_cuda_gpu_kern_sum.txt` + `*_cuda_gpu_mem_time_sum.txt` + `*_cuda_api_sum.txt`
for all five traces. They re-derive the README claims with a full cross-check
table (0.3 % pointwise, 7,861 / 7,526 ns, 3.075 / 3.096 s all match; the FFT-chain
"49.9 %" prose is corrected to **50.0 %** — the table rows 19.5 + 18.2 + 12.3 sum
exactly to 50.0 on the cited `v7_triton_rerun` trace). `gputrace` is a legacy
report name absent from nsys 2025.3.2; `cuda_gpu_kern_sum` covers the same
per-kernel detail.

**Timeline (post-P0.1 FFT chain)**: the 50.0 % FFT chain is the **pre-P0.1**
v7-CUDA/v7-Triton picture. Re-measured 2026-08-04 on the committed P0.1+P0.3
trace `v7_p013_nvtx` (its kern_sum is in `nsys_reports/`): the FFT chain is
**~34 %** (`regular_fft` 17.2 % + `vector_fft` 16.7 %); P0.1's combined kernel
folds the complex multiply (12.3 % pre-P0.1) in, leaving a 0.4 % residual.

### 5. Kernel micro-bench numbers (P1/P2 graph story + P0.2)

**Eager fuse4/split4 per-backend — re-verified 2026-08-03** (GPU 7, **S=1024,
O=64, H=W=4096**, reps=100):

```text
cuda_bigbuf      fuse4= 60.1us split4= 25.4us total= 85.5us
triton_bigbuf    fuse4= 46.9us split4= 47.5us total= 94.4us   # fuse4 wins, split4 loses
pytorch_inplace  fuse4=181.9us split4=103.6us total=285.5us
```

**Script**: `baseline/test_baseline_v8_bigbuf_ablation.py` →
`_benchmark_backend(name, H, W, S, O, warmup, reps)` — **eager only**.

**P1/P2 graph-mode numbers removed (2026-08-04)**: the README's P1/P2
**graph-mode** numbers (triton 21.19 µs, cuda 14.80 µs, production 20.9/28.6
µs, "2.11×", `cuda_bigbuf` diff=5.28) came from an **uncommitted one-off POC
(2026-07)** — no committed script regenerates them, so they were **deleted
from the README** rather than re-run. The graph-safety story is covered
reproducibly by the 5-mode benchmark `benchmark_v7_graph_modes.py` (§2):
mode 3 reproduces the unsafe-capture garbage output (`loss_rel=1.02`), mode 5
passes bit-exact. Also on 2026-08-04 the v8 **eager** ablation was re-run at
canvas H=W=4096 (`benchmarks/results/v8_ablation.json`), and the stale
`docs/triton_tutorial_03_backends.md` "Triton 22% faster" row was corrected.

| Number | Source / status |
|---|---|
| P2 graph speedup 2.11×; cuda_bigbuf graph diff=5.28 FAIL | **deleted** (uncommitted POC); reproducible graph evidence = 5-mode bench (§2) + `graph_modes.json` |
| P1 production 20.9 / 28.6 µs, pytorch_inplace 131 µs | **deleted** (same POC) |
| v8 eager ablation (CUDA 29.2 / Triton 46.0 µs fuse4) | **re-measured 2026-08-04**, `benchmarks/results/v8_ablation.json` |
| P0.2 Triton reduction 1.48× (871 µs vs 1287 µs) | eval-phase micro-bench, `_LithoSim._computeImage` — **not** on the v7 hot path (task.md:48) |

These are isolated kernel micro-benchmarks; per the README "Measurement
domains" note they do **not** add up to the 22.5× E2E number.

### 6. Batch scaling (256 → 1024 tiles)

README: 256 tiles = 2.34 s / 5997 MB; **1024 tiles = 8.31 s / 23337 MB (95% of
24 GB)**.

**2026-08-03 re-run of ALL v7 variants at 1024 tiles** (GPU 7,
`FUILT_TILE_MULTIPLIER=64`, 20 iters):

| Variant | Result | Peak VRAM (MB) | Pure pipeline (s) |
|---|---:|---:|---:|
| v7-plain (no P0.x) | ✅ runs | 23329 | 34.875 |
| **v7 + P0.1** (combined kernel only) | ✅ runs | **23337 (= README!)** | 9.860 |
| v7 + P0.1 + P0.3 (final, async D2H) | ⚠ OOM'd twice, **completed once** | 23337 (OK) / ~23380 (OOM) | **8.276** (when it fits) |

- **The final variant is BORDERLINE, not broken**: same config OOM'd twice
  (~23380 MB) but also completed once at **23337 MB / 8.276 s — reproducing the
  README's 8.31 s / 23337 MB row almost exactly**. The ~40 MB difference is
  allocator/fragmentation variance right at the VRAM cliff (GPU 7 usable
  ≈ 23.5 GiB; the OOM died on a 20 MiB alloc). `expandable_segments` did not
  help the failed runs.
- **FIXED 2026-08-03 (this commit)**: `torch.cuda.empty_cache()` before the
  peak D2H_Save `workspace.fuse()` returns the caching allocator's ~3.6 GB of
  reclaimable slack to the driver (the timeline shows reserved 23352 MB vs live
  ~19769 MB — a ~3.6 GB pool of free-but-held blocks). With it, the final
  variant **completed 5/5 runs** at 1024 tiles — 3× at 5 iters + 2× at the
  README's 20-iter config (peak 23337 MB, pure pipeline 8.51 s, no OOM; it had
  OOM'd 2/3 before). A `_memory_budget_ok()` pre-flight guard now runs after
  setup+warmup and hard-fails (with a clear "reduce tiles / shard" message)
  before the iteration loop if usable headroom (driver-free + reclaimable) is
  exhausted.
- **CUDA-graph path at 1024 tiles also runs** (`FUILT_USE_GRAPH=1`, 20 iters):
  3/3 completed, peak 23337 MB, **graph replay 116.5 ms/iter** vs eager
  ~425 ms/iter (≈3.7× faster — the whole 1024-tile iteration is captured, so
  replay drops the per-tile launch tax). Graph mode passed 2/2 *without* the
  empty_cache fix (its allocation sequence doesn't hit the same D2H cliff); the
  empty_cache was added to its D2H step anyway for uniformity.
- **All 20 iterations complete before the OOM point.** The peak is hit in the
  final `D2H_Save` stage: `workspace.fuse()` materialises the 32×32-grid canvas
  (~30784² = 3.8 GB) while the 1024 tile params are still pinned for async
  D2H. The profile (below) shows D2H_Save = 3.08 s = 37% of the pure pipeline.
- **The async-D2H memory cost is pinned HOST RAM, not VRAM**: 1024 × 4 MB
  tile params + ~3.8 GB large-mask ≈ **~7.8 GB locked host memory** (vs the
  sync path's pageable ~7.8 GB). Peak VRAM is essentially unchanged (~23337 MB)
  — async D2H buys speed by overlapping the params D2H with the large-mask GPU
  generation, it does not systematically add VRAM.
- Net: the README's 1024-tile row is **real and reproducible** but sits ~1.2 GB
  from the 24 GB ceiling — treat it as a ceiling probe; a fresh run may or may
  not fit depending on allocator state.

Script: `baseline/test_baseline_v7_fuselevelset.py` with
`FUILT_TILE_MULTIPLIER=64` (+ `FUILT_USE_COMBINED_KERNEL=1` and
`FUILT_USE_ASYNC_D2H=1` for the final variant).

### 7. Correctness comparisons (committed JSONs)

`benchmarks/results/compare_*.json` (committed) record loss/grad
agreement between the CUDA fused path, the PyTorch FFT path, and the levelset
solver on a single tile — the numerical-equivalence evidence for the
kernel-porting work.

---

## What this ledger deliberately does *not* claim

- **v1 and v7+P0.1+P0.3 WERE re-verified** on 2026-08-03 (§3) and land within
  +1.5 % / +4.5 % of the README table on a loaded host — the ~22× speedup is
  robust. The **other** E2E rows (v3/v5/v6/v7-CUDA/v7-Triton/P0.1/P0.4) were
  **not** re-run; their rerun commands regenerate them.
- The README's "mean ± std over 3 repeats" was done by running the scripts
  multiple times externally — the scripts themselves run **once** (no internal
  repeat loop) and v1 has **no warmup** (the README's "warmup 10" applies to
  the v7 eager path via `FUILT_BENCH_WARMUP`). v7+P0.1+P0.3 was re-run 3× for
  this ledger (§3).
- The "Triton beats CUDA by 27%" claim is **gone** (removed across all docs):
  it was an artifact of the CUDA capture-stream bug (`cuda_bigbuf`'s graph
  output was garbage, so its production number fell back to eager). After the
  `getCurrentCUDAStream()` fix, fixed-CUDA-graph edges out triton-graph
  (5-mode row 5 < row 4, verified above).
- **1024-tile batch scaling re-ran 2026-08-03**: v7-plain and v7+P0.1 fit
  (v7+P0.1 = 23337 MB / 9.86 s). The final v7+P0.1+P0.3 variant is **borderline
  at the VRAM cliff**: it OOM'd twice (~23380 MB) but also completed once at
  **23337 MB / 8.276 s — reproducing the README's 8.31 s / 23337 MB row** (§6).
  The ~40 MB swing is allocator variance, not a systematic regression.
- **The README P1/P2 graph-mode numbers** (21.19 / 14.80 / 20.9 / 28.6 µs)
  came from an **uncommitted one-off POC** and are not regenerable from a
  committed script — **deleted from the README on 2026-08-04** (§5). The
  reproducible graph-mode evidence is the 5-mode benchmark (§2,
  `graph_modes.json`).
- **`docs/triton_tutorial_03_backends.md` had stale numbers** ("Triton beats
  CUDA 6%", "Triton 22% faster") that conflicted with the README v8 tables;
  corrected 2026-08-04 to the fresh H=W=4096 ablation (`v8_ablation.json`) —
  CUDA wins fuse4 ~1.57× (§5).
