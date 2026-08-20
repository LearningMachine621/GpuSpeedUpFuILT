"""v8 ablation: compare fuse4/split4 across three backends.

Three implementations of the same operation, measured under identical
conditions:

  pytorch_inplace  — pure PyTorch slice + add_/mul_/copy_
  cuda_bigbuf      — hand-written CUDA kernel (perf ceiling reference)
  triton_bigbuf    — Triton port (recommended production path)

For each backend, we measure:
  - fuse4 throughput (calls/s)
  - split4 throughput
  - peak VRAM during the call
  - numerical diff vs cuda_bigbuf

Run:
  python -m baseline.test_baseline_v8_bigbuf_ablation
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import torch

from fuilt.fusion_split.backends import get_backend


@dataclass
class BackendResult:
    name: str
    fuse_us_per_call: float
    split_us_per_call: float
    fuse_vram_mb: float
    split_vram_mb: float
    fuse_max_diff_vs_cuda: float
    split_max_diff_vs_cuda: float
    fuse_pass: bool
    split_pass: bool


def _make_layout(H: int, W: int, S: int, O: int):
    """Build a valid in-bounds layout for fuse4 and split4 tests.

    Returns (fuse_offsets, split_offsets, parent_pos).
    """
    P = S - O
    S_out = S + P

    # Sub-blocks A, B, C, D in a 2x2 grid in the top-left of in_big.
    ax0, ay0 = 0, 0
    bx0, by0 = S, 0
    cx0, cy0 = 0, S
    dx0, dy0 = S, S

    # Parent block (output of fuse / input of split) goes in the top-right.
    ix0, iy0 = 2 * S, 0

    assert ix0 + S_out <= W, f"bigbuf too small: need W >= {ix0 + S_out}, got {W}"
    assert dy0 + S <= H,    f"bigbuf too small: need H >= {dy0 + S}, got {H}"

    fuse_offsets = torch.tensor(
        [[ax0, ay0], [bx0, by0], [cx0, cy0], [dx0, dy0], [ix0, iy0]],
        dtype=torch.int32,
    )
    split_offsets = torch.tensor(
        [[ax0, ay0], [bx0, by0], [cx0, cy0], [dx0, dy0], [ix0, iy0]],
        dtype=torch.int32,
    )
    return fuse_offsets, split_offsets


def _benchmark_backend(
    backend_name: str,
    H: int = 4096,
    W: int = 4096,
    S: int = 1024,
    O: int = 64,
    warmup: int = 5,
    reps: int = 50,
    atol: float = 1e-3,
) -> BackendResult:
    """Benchmark one backend: throughput + numerical equivalence vs cuda_bigbuf."""
    device = torch.device("cuda")
    backend = get_backend(backend_name)
    P = S - O
    S_out = S + P
    fuse_offsets, split_offsets = _make_layout(H, W, S, O)

    # Same input for all backends — deterministic seed for reproducibility
    torch.manual_seed(0)
    in_big = torch.randn(H, W, device=device, dtype=torch.float32)

    # ============================================================
    # fuse4 benchmark
    # ============================================================
    out_big = torch.zeros(H, W, device=device, dtype=torch.float16)

    # Warmup (also triggers Triton JIT on first call)
    for _ in range(warmup):
        out_big.zero_()
        backend.fuse4(in_big, out_big, fuse_offsets, S, O)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(reps):
        out_big.zero_()
        backend.fuse4(in_big, out_big, fuse_offsets, S, O)
    torch.cuda.synchronize()
    fuse_us = (time.perf_counter() - t0) / reps * 1e6
    fuse_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # ============================================================
    # split4 benchmark — needs a parent block in in_big to read from
    # ============================================================
    # Put a synthetic parent block at (ix0, iy0). For benchmarking purposes
    # the values don't matter; for numerical-equivalence we'll use the same
    # in_big across backends.
    parent = torch.randn(S_out, S_out, device=device, dtype=torch.float32)
    in_big_split = torch.zeros(H, W, device=device, dtype=torch.float32)
    ix0 = int(fuse_offsets[4, 0])
    iy0 = int(fuse_offsets[4, 1])
    in_big_split[iy0:iy0 + S_out, ix0:ix0 + S_out] = parent

    out_big_split = torch.zeros(H, W, device=device, dtype=torch.float16)
    for _ in range(warmup):
        out_big_split.zero_()
        backend.split4(in_big_split, out_big_split, split_offsets, S, O)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(reps):
        out_big_split.zero_()
        backend.split4(in_big_split, out_big_split, split_offsets, S, O)
    torch.cuda.synchronize()
    split_us = (time.perf_counter() - t0) / reps * 1e6
    split_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # ============================================================
    # Numerical equivalence vs cuda_bigbuf (only for non-cuda backends)
    # ============================================================
    fuse_diff = 0.0
    split_diff = 0.0
    if backend_name != "cuda_bigbuf":
        cuda = get_backend("cuda_bigbuf")

        # fuse diff
        out_cuda = torch.zeros(H, W, device=device, dtype=torch.float16)
        out_test = torch.zeros(H, W, device=device, dtype=torch.float16)
        cuda.fuse4(in_big, out_cuda, fuse_offsets, S, O)
        backend.fuse4(in_big, out_test, fuse_offsets, S, O)
        region_cuda = out_cuda[iy0:iy0 + S_out, ix0:ix0 + S_out].float()
        region_test = out_test[iy0:iy0 + S_out, ix0:ix0 + S_out].float()
        fuse_diff = (region_cuda - region_test).abs().max().item()

        # split diff (compute over the whole bigbuf for simplicity)
        out_cuda_split = torch.zeros(H, W, device=device, dtype=torch.float16)
        out_test_split = torch.zeros(H, W, device=device, dtype=torch.float16)
        cuda.split4(in_big_split, out_cuda_split, split_offsets, S, O)
        backend.split4(in_big_split, out_test_split, split_offsets, S, O)
        split_diff = (out_cuda_split.float() - out_test_split.float()).abs().max().item()

    return BackendResult(
        name=backend_name,
        fuse_us_per_call=fuse_us,
        split_us_per_call=split_us,
        fuse_vram_mb=fuse_vram_mb,
        split_vram_mb=split_vram_mb,
        fuse_max_diff_vs_cuda=fuse_diff,
        split_max_diff_vs_cuda=split_diff,
        fuse_pass=fuse_diff < atol,
        split_pass=split_diff < atol,
    )


def main():
    parser = argparse.ArgumentParser(description="v8 fuse4/split4 backend ablation")
    parser.add_argument(
        "--backends",
        type=str,
        default="pytorch_inplace,cuda_bigbuf,triton_bigbuf",
        help="comma-separated backend names",
    )
    parser.add_argument("--reps", type=int, default=50, help="timed reps per backend")
    parser.add_argument("--warmup", type=int, default=5, help="warmup calls before timing")
    parser.add_argument("--S", type=int, default=1024, help="sub-block side length")
    parser.add_argument("--O", type=int, default=64, help="overlap")
    parser.add_argument("--out-json", type=str, default="", help="persist results JSON to this path")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("v8 ablation requires CUDA")

    backends: List[str] = [b.strip() for b in args.backends.split(",")]

    print(f"\n{'='*78}")
    print(f"v8 fuse4/split4 backend ablation")
    print(f"{'='*78}")
    print(f"S={args.S}, O={args.O}, P={args.S - args.O}, S_out={args.S + (args.S - args.O)}")
    print(f"backends: {backends}")
    print(f"warmup={args.warmup}, reps={args.reps}")
    print(f"{'-'*78}")

    # IMPORTANT: benchmark cuda_bigbuf first so the .so is compiled before
    # any non-cuda backend's numerical-equivalence check needs it.
    if "cuda_bigbuf" in backends:
        backends.remove("cuda_bigbuf")
        backends.insert(0, "cuda_bigbuf")

    results: List[BackendResult] = []
    for name in backends:
        print(f"\n[{name}] benchmarking...")
        result = _benchmark_backend(
            backend_name=name,
            S=args.S,
            O=args.O,
            warmup=args.warmup,
            reps=args.reps,
        )
        results.append(result)
        print(f"  fuse4: {result.fuse_us_per_call:7.1f} us/call | "
              f"VRAM {result.fuse_vram_mb:6.1f} MB | "
              f"diff vs cuda {result.fuse_max_diff_vs_cuda:.4f} "
              f"({'PASS' if result.fuse_pass else 'FAIL'})")
        print(f"  split4: {result.split_us_per_call:7.1f} us/call | "
              f"VRAM {result.split_vram_mb:6.1f} MB | "
              f"diff vs cuda {result.split_max_diff_vs_cuda:.4f} "
              f"({'PASS' if result.split_pass else 'FAIL'})")

    print(f"\n{'='*78}")
    print(f"Summary")
    print(f"{'='*78}")
    print(f"{'Backend':<18} {'fuse4 µs':>10} {'split4 µs':>10} "
          f"{'fuse vs cuda':>14} {'split vs cuda':>14}")
    print(f"{'-'*78}")
    for r in results:
        fuse_diff_str = "baseline" if r.name == "cuda_bigbuf" else f"{r.fuse_max_diff_vs_cuda:.4f}"
        split_diff_str = "baseline" if r.name == "cuda_bigbuf" else f"{r.split_max_diff_vs_cuda:.4f}"
        print(f"{r.name:<18} {r.fuse_us_per_call:>10.1f} {r.split_us_per_call:>10.1f} "
              f"{fuse_diff_str:>14} {split_diff_str:>14}")
    print(f"{'='*78}\n")

    if args.out_json:
        _write_out_json(args.out_json, results, args, git_commit=_git_commit())


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, cwd=str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))).stdout.strip()
    except Exception:
        return "?"


def _write_out_json(path: str, results: List[BackendResult], args: argparse.Namespace,
                    git_commit: str) -> None:
    import triton
    payload = {
        "script": "baseline/test_baseline_v8_bigbuf_ablation.py",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": git_commit,
        "gpu": torch.cuda.get_device_name(0),
        "env": {"torch": torch.__version__, "cuda": torch.version.cuda,
                "triton": triton.__version__},
        "config": {
            "canvas": "4096x4096",          # _benchmark_backend H=W=4096 defaults
            "S": args.S, "O": args.O,
            "warmup": args.warmup, "reps": args.reps,
        },
        "backends": [
            {
                "name": r.name,
                "fuse4_us_per_call": r.fuse_us_per_call,
                "split4_us_per_call": r.split_us_per_call,
                "fuse4_vram_mb": r.fuse_vram_mb,
                "split4_vram_mb": r.split_vram_mb,
                "fuse4_diff_vs_cuda": r.fuse_max_diff_vs_cuda,
                "split4_diff_vs_cuda": r.split_max_diff_vs_cuda,
                "fuse4_pass": r.fuse_pass,
                "split4_pass": r.split_pass,
            }
            for r in results
        ],
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] wrote {path}")


if __name__ == "__main__":
    main()
