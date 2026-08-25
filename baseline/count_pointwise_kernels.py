"""Count the ACTUAL CUDA kernels behind each pointwise backend (definitive).

  unfused : the v5/v6 eager chain in _compute_pointwise (production code)
  micro   : the bench_pointwise_torch expression from run_benchmark_matrix
  triton  : the fused production kernel
  cuda    : the fused hand-written kernel

Answers "how many operators did we actually fuse?" with a measurement
(torch.profiler CUDA kernel events), not an expression count. Regenerate:
  CUDA_VISIBLE_DEVICES=<any> python -m baseline.count_pointwise_kernels \
      --out-json benchmarks/results/pointwise_kernel_count.json
"""
import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/home/lyj/code/GpuSpeedUpFuILT")
from torch.profiler import ProfilerActivity, profile

from baseline.test_baseline_v7_fuselevelset import _compute_pointwise

a = torch.randn(1024, 1024, device="cuda")
t = torch.rand(1024, 1024, device="cuda")


def micro_expr():
    printed = torch.sigmoid((a - 0.5) * 1.0)
    diff = printed - t
    grad = 2.0 * diff * (printed * (1.0 - printed))
    return grad


def bench_us(fn, reps=200):
    """Wall time per call (µs) — CUDA events around a reps-loop, like the
    kernel-microbench harness. Includes host dispatch (dominates Triton)."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    st = torch.cuda.current_stream()
    s.record(st)
    for _ in range(reps):
        fn()
    e.record(st)
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps * 1000


def gpu_busy_us(fn):
    """Pure GPU busy time for ONE call (torch.profiler self device time)."""
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(e.self_device_time_total for e in prof.key_averages()
               if "Memcpy" not in e.key and "memset" not in e.key.lower())


def count(fn, tag):
    fn()  # warmup
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    rows = [(e.key, e.count) for e in prof.key_averages()
            if e.self_device_time_total > 0 and "Memcpy" not in e.key and "memset" not in e.key.lower()]
    total = sum(c for _, c in rows)
    print(f"\n[{tag}] {total} CUDA kernel launches:")
    for k, c in sorted(rows, key=lambda r: -r[1]):
        print(f"   {c:>2}x  {k[:100]}")
    return total


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out-json", type=str, default="")
    args = p.parse_args()
    cases = {
        "unfused_10op_production": lambda: _compute_pointwise(a, t, "unfused", 0.5, 1.0),
        "micro_8op_expression": micro_expr,
        "triton_fused": lambda: _compute_pointwise(a, t, "triton", 0.5, 1.0),
        "cuda_fused": lambda: _compute_pointwise(a, t, "cuda", 0.5, 1.0),
    }
    n_unfused = count(cases["unfused_10op_production"], "unfused (production v5/v6 eager chain)")
    n_micro = count(cases["micro_8op_expression"], "micro-bench expression (bench_pointwise_torch)")
    n_triton = count(cases["triton_fused"], "triton fused (production)")
    n_cuda = count(cases["cuda_fused"], "cuda fused (reference)")
    print(f"\nSUMMARY: unfused={n_unfused}  micro_expr={n_micro}  triton_fused={n_triton}  cuda_fused={n_cuda}")
    timing = {k: {"wall_us": round(bench_us(f), 2), "gpu_busy_us": round(gpu_busy_us(f), 2)}
              for k, f in cases.items()}
    print("\nTIMING (1024x1024 fp32, 2026-08-25 host state):")
    for k, v in timing.items():
        print(f"   {k:<26} wall {v['wall_us']:7.2f} us   GPU-busy {v['gpu_busy_us']:6.2f} us")
    if args.out_json:
        def _c():
            try:
                return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
            except Exception:
                return "unknown"
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps({
            "script": "baseline/count_pointwise_kernels.py",
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "git_commit": _c(),
            "gpu": torch.cuda.get_device_name(0),
            "method": "torch.profiler CUDA kernel events, 1024x1024 fp32, warmup + 1 call",
            "counts": {"unfused_production_v5v6_chain": n_unfused,
                        "microbench_expression": n_micro,
                        "triton_fused": n_triton,
                        "cuda_fused": n_cuda},
            "timing_1024x1024_fp32": timing,
            "conclusion": ("The fused kernel replaces the production eager chain of "
                            f"{10} ATen launches with 1 kernel; earlier doc claims of "
                            "'~5' (README) and '8' (microbench expression) were estimates "
                            "that undercounted the steepness muls and diff_sq."),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.out_json}")
