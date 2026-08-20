# Committed benchmark results (raw evidence)

The raw machine-readable outputs behind the performance numbers in
`README.md`. These are **tracked in git** so a reviewer can inspect the actual
measurements without re-running. Each file lists which script produced it and
how to regenerate it (see [`docs/verifiable_results.md`](../docs/verifiable_results.md)
for the full provenance + environment).

| File | What it records | Produced by | Regenerate |
|---|---|---|---|
| `graph_modes.json` | 5-mode CUDA-graph bug-vs-fix benchmark (per-mode ms/iter, loss, status) | `baseline/benchmark_v7_graph_modes.py` | `CUDA_VISIBLE_DEVICES=<idle> python -m baseline.benchmark_v7_graph_modes --iters 20` |
| `v7_profile.json` | v7+P0.1+P0.3 E2E profile (per-stage seconds, peak VRAM, μ-bench ms/tile) | `baseline/test_baseline_v7_fuselevelset.py` | `CUDA_VISIBLE_DEVICES=<idle> FUILT_USE_COMBINED_KERNEL=1 FUILT_USE_ASYNC_D2H=1 FUILT_ITERS=20 python -m baseline.test_baseline_v7_fuselevelset --mode pytorch_fft` |
| `kernel_microbench.json` | pointwise / FFT micro-bench (ms per call, CUDA events) | `baseline/run_benchmark_matrix.py` `_run_kernel_microbench` | `CUDA_VISIBLE_DEVICES=<idle> python -m baseline.run_benchmark_matrix --run-kernel-microbench --out-dir benchmarks/results` |
| `compare_*.json` | single-tile loss/grad agreement between CUDA-fused / PyTorch-FFT / levelset paths | `outputs/compare_single_tile/` originals (copy preserved here) | see `baseline/` compare scripts |
| `headline_e2e.json` | README headline E2E table re-run (per-rep + mean/std, GPU/versions/AMP metadata) | `baseline/rerun_headline.py` | `CUDA_VISIBLE_DEVICES=<idle> PYTHONPATH=. python -m baseline.rerun_headline --repeats 3` |
| `v8_ablation.json` | fuse4/split4 backend ablation at canvas H=W=4096 (CUDA/Triton/PyTorch eager µs) | `baseline/test_baseline_v8_bigbuf_ablation.py` | `python -m baseline.test_baseline_v8_bigbuf_ablation --out-json benchmarks/results/v8_ablation.json` |
| `hierarchical.json` | flat vs hierarchical fuse/split at N∈{4,16,64,256} (ms) | `baseline/test_baseline_v8_hierarchical_pipeline.py` | `python -m baseline.test_baseline_v8_hierarchical_pipeline --out-json benchmarks/results/hierarchical.json` |
| `p01_equivalence.json` | P0.1 24-loop vs combined-kernel A/B: fp32 max_abs_diff ≤ 3.7e-9, fp64 ~1e-18, ulp-perturbation sanity | `baseline/verify_p01_equivalence.py` | `CUDA_VISIBLE_DEVICES=<any> python -m baseline.verify_p01_equivalence --out-json benchmarks/results/p01_equivalence.json` |
| `nsys_reports/` | derived `nsys stats` reports (kern_sum / mem_time_sum / api_sum per trace) | `nsys stats` on committed `outputs/nsys_traces/*.nsys-rep` | see `nsys_reports/README.md` |

The nsys traces themselves (`outputs/nsys_traces/*.nsys-rep`, ~11-20 MB each)
**are committed**; only the derived `.sqlite` stats are gitignored (rebuilt from
the `.nsys-rep` by `nsys stats`). Regeneration commands in
`docs/verifiable_results.md` §4.

Canonical run: 2026-08-03, GPU 7 (isolated RTX 4090), torch 2.11.0+cu130,
triton 3.6.0, driver 580.126.09.
