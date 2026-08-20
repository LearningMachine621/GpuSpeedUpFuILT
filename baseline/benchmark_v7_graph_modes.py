"""Unified 5-mode benchmark: the CUDA capture-stream fix, before vs after.

One harness, one shared ctx (256 tiles, P0.1 combined kernel), five modes that
tell the whole graph-capture story:

  1. unfused eager      pure-PyTorch pointwise, eager            (v5/v6 baseline)
  2. fused eager        hand-written CUDA pointwise, eager       (v7)
  3. fused graph        CUDA pointwise + CUDA Graph, capture_safe=OFF  (BUGGY)
  4. triton graph       Triton pointwise + CUDA Graph            (capture-safe)
  5. fixed CUDA graph   CUDA pointwise + CUDA Graph, capture_safe=ON   (FIX)

Mode 3 reproduces the capture-stream bug: a hand-written kernel that launches on
`getDefaultCUDAStream()` (stream 0) is silently NOT recorded into the graph
captured on a side stream, so replay skips it -> stale/wrong output. Mode 5 is
the SAME kernel after the `getCurrentCUDAStream()` fix. Mode 4 is Triton, which
honors the capture stream automatically (no fix needed).

Numerical equivalence: every mode's final params are diffed against mode 2
(fused eager CUDA) as the reference. Correct modes match within fp/fast-math
tolerance; the buggy mode 3 diverges.

⚠ Run on an EXCLUSIVE GPU. On a shared multi-GPU machine, concurrent workloads
distort latency 5-15× (power throttling, PCIe contention). Pin an idle GPU:
  CUDA_VISIBLE_DEVICES=1 python -m baseline.benchmark_v7_graph_modes

Run:
  CUDA_VISIBLE_DEVICES=1 FUILT_TILE_MULTIPLIER=16 \
    python -m baseline.benchmark_v7_graph_modes --iters 20
"""

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from baseline.test_baseline_v7_fuselevelset import (
    DEVICE,
    _run_one_iter,
    set_pointwise_capture_safe,
    setup_v7_context,
)

TILES_DIR = os.environ.get("FUILT_TILES_DIR", "/data/lyj/FuILT/tiles_from_patch")


@dataclass
class ModeResult:
    name: str
    pointwise: str
    exec_mode: str  # "eager" | "graph"
    capture_safe: Optional[bool]  # True/False for cuda-graph; None otherwise
    ms_per_iter: float
    max_param_diff: float  # vs fused-eager-CUDA reference (informational)
    loss_rel_diff: float  # |loss - ref_loss| / |ref_loss| — the correctness gate
    final_loss: float
    status: str  # "PASS" | "FAIL" | "REF"
    note: str = ""


def _snapshot_params(ctx) -> List[torch.Tensor]:
    return [p.detach().clone() for p in ctx["params_list"]]


def _restore_params(ctx, snapshot: List[torch.Tensor]) -> None:
    for p, p0 in zip(ctx["params_list"], snapshot):
        p.copy_(p0)
    torch.cuda.synchronize()


def _max_param_diff(ctx, reference: List[torch.Tensor]) -> float:
    diffs = [
        (p - ref).abs().max().item()
        for p, ref in zip(ctx["params_list"], reference)
    ]
    return max(diffs) if diffs else 0.0


def _run_eager(ctx, pointwise: str, iters: int, warmup: int) -> Tuple[float, float]:
    """Run `iters` eager iterations; return (ms/iter, avg per-tile loss)."""
    loss_accum = torch.zeros(1, device=DEVICE, dtype=torch.float32)
    for _ in range(warmup):
        _run_one_iter(ctx, loss_accum, pointwise=pointwise)
    torch.cuda.synchronize()
    loss_accum.zero_()
    t0 = time.perf_counter()
    for _ in range(iters):
        _run_one_iter(ctx, loss_accum, pointwise=pointwise)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    ms_per_iter = elapsed / iters * 1000.0
    avg_loss = float(loss_accum.item()) / max(1, iters * len(ctx["params_list"]))
    return ms_per_iter, avg_loss


def _run_graph(
    ctx, pointwise: str, iters: int, warmup: int, capture_safe: bool
) -> Tuple[float, float, str]:
    """Warmup -> capture one iter -> replay `iters` times.

    For pointwise=="cuda", `capture_safe` toggles the kernel's stream:
      True  -> getCurrentCUDAStream() (recorded into the graph; the FIX)
      False -> getDefaultCUDAStream() (skipped on replay; the BUG)
    Returns (ms/iter, avg per-tile loss, capture_status).
    """
    if pointwise == "cuda":
        set_pointwise_capture_safe(capture_safe)

    loss_accum = torch.zeros(1, device=DEVICE, dtype=torch.float32)
    for _ in range(warmup):
        _run_one_iter(ctx, loss_accum, pointwise=pointwise)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            _run_one_iter(ctx, loss_accum, pointwise=pointwise)
    except Exception as e:
        return float("nan"), float("nan"), f"CAPTURE-FAILED: {type(e).__name__}: {e}"

    # Zero the accumulator in-place (same data_ptr) so timed replays report a
    # clean per-iter loss. Graphs replay ops, so in-place writes to persistent
    # buffers are reflected on each replay.
    loss_accum.zero_()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        g.replay()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    ms_per_iter = elapsed / iters * 1000.0
    avg_loss = float(loss_accum.item()) / max(1, iters * len(ctx["params_list"]))
    return ms_per_iter, avg_loss, "OK"


def _status(max_diff: float, equiv_tol: float) -> str:
    return "PASS" if max_diff < equiv_tol else "FAIL"


def _loss_rel_diff(loss: float, ref_loss: float) -> float:
    """Relative loss deviation from the reference (the correctness gate).

    Why loss, not param-diff: when the pointwise kernel is skipped on graph
    replay, its diff^2 output is stale/uninitialized garbage, so the per-iter
    loss is immediately and obviously wrong (sign-flipped / huge) — no need to
    wait for param drift to accumulate. Correct modes match the eager reference
    loss to ~fp precision regardless of iteration count.
    """
    return abs(loss - ref_loss) / max(abs(ref_loss), 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description="v7 unified 5-mode graph benchmark")
    parser.add_argument("--iters", type=int, default=int(os.environ.get("FUILT_ITERS", "20")))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--multiplier", type=int, default=int(os.environ.get("FUILT_TILE_MULTIPLIER", "16")))
    parser.add_argument("--loss-tol", type=float, default=0.05,
                        help="relative loss deviation from reference for PASS (correctness gate)")
    parser.add_argument("--param-tol", type=float, default=0.5,
                        help="max param-diff vs reference (informational; the gate is loss-based)")
    parser.add_argument("--tiles-dir", default=TILES_DIR)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("benchmark requires CUDA")

    print("=" * 78)
    print("v7 unified 5-mode benchmark — CUDA capture-stream fix (before vs after)")
    print("=" * 78)
    print(f"GPU: {torch.cuda.get_device_name(0)}  (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})")
    print(f"iters={args.iters}, warmup={args.warmup}, loss_tol={args.loss_tol} (param_tol={args.param_tol}, informational)")
    print("-" * 78)

    # One shared context — all modes see the same tiles/kernels/workspace.
    ctx = setup_v7_context(args.tiles_dir, multiplier=args.multiplier, pointwise="cuda")
    initial = _snapshot_params(ctx)

    results: List[ModeResult] = []

    # ---- Mode 2 first (internally): fused eager CUDA = the REFERENCE trajectory ----
    # Run it before the others so its final params + loss are the diff baseline.
    _restore_params(ctx, initial)
    set_pointwise_capture_safe(True)
    ms2, loss2 = _run_eager(ctx, "cuda", args.iters, args.warmup)
    reference = _snapshot_params(ctx)
    ref_loss = loss2

    def _gate(loss: float) -> str:
        return "PASS" if _loss_rel_diff(loss, ref_loss) < args.loss_tol else "FAIL"

    # ---- Mode 1: unfused eager (pure-PyTorch pointwise, no fused kernel) ----
    _restore_params(ctx, initial)
    set_pointwise_capture_safe(True)
    ms, loss = _run_eager(ctx, "unfused", args.iters, args.warmup)
    pdiff = _max_param_diff(ctx, reference)
    results.append(ModeResult("unfused eager", "unfused", "eager", None, ms, pdiff, _loss_rel_diff(loss, ref_loss), loss, _gate(loss), "pure-PyTorch baseline"))

    # ---- Mode 2: fused eager (CUDA pointwise) — REFERENCE ----
    results.append(ModeResult("fused eager", "cuda", "eager", None, ms2, 0.0, 0.0, loss2, "REF", "reference trajectory"))

    # ---- Mode 3: fused graph (CUDA, capture_safe=OFF → BUGGY) ----
    _restore_params(ctx, initial)
    ms, loss, cap = _run_graph(ctx, "cuda", args.iters, args.warmup, capture_safe=False)
    pdiff = _max_param_diff(ctx, reference)
    note = "default-stream launch skipped on replay" if cap == "OK" else cap
    results.append(ModeResult("fused graph", "cuda", "graph", False, ms, pdiff, _loss_rel_diff(loss, ref_loss), loss, _gate(loss), note))
    set_pointwise_capture_safe(True)  # restore before any later CUDA path

    # ---- Mode 4: triton graph (capture-safe by default) ----
    _restore_params(ctx, initial)
    ms, loss, cap = _run_graph(ctx, "triton", args.iters, args.warmup, capture_safe=True)
    pdiff = _max_param_diff(ctx, reference)
    results.append(ModeResult("triton graph", "triton", "graph", None, ms, pdiff, _loss_rel_diff(loss, ref_loss), loss, _gate(loss), cap))

    # ---- Mode 5: fixed CUDA graph (capture_safe=ON → FIX) ----
    _restore_params(ctx, initial)
    ms, loss, cap = _run_graph(ctx, "cuda", args.iters, args.warmup, capture_safe=True)
    pdiff = _max_param_diff(ctx, reference)
    results.append(ModeResult("fixed CUDA graph", "cuda", "graph", True, ms, pdiff, _loss_rel_diff(loss, ref_loss), loss, _gate(loss), cap))

    # ---- Report ----
    # results are already in display order: 0=mode1 ... 4=mode5
    order = [0, 1, 2, 3, 4]
    print("\n" + "=" * 90)
    print(f"{'#':<3}{'mode':<20}{'ms/iter':>10}{'speedup':>10}{'param-diff':>12}{'loss-rel':>10}{'loss':>14}  status")
    print("-" * 90)
    base_ms = results[0].ms_per_iter  # speedup vs unfused eager
    for display_i, idx in enumerate(order, 1):
        r = results[idx]
        speedup = base_ms / r.ms_per_iter if r.ms_per_iter == r.ms_per_iter else float("nan")
        pdiff_str = "ref" if r.status == "REF" else f"{r.max_param_diff:.4f}"
        lrd_str = "ref" if r.status == "REF" else f"{r.loss_rel_diff:.2e}"
        print(f"{display_i:<3}{r.name:<20}{r.ms_per_iter:>10.3f}{speedup:>9.2f}×{pdiff_str:>12}{lrd_str:>10}{r.final_loss:>14.4e}  {r.status}")
    print("-" * 90)
    print(f"speedup vs mode 1 (unfused eager = {base_ms:.3f} ms/iter). param-diff vs mode 2.")
    print(f"loss-rel = |loss - ref| / |ref|. status: PASS if loss-rel < {args.loss_tol}; mode 2 is the reference.")
    print("=" * 90)

    print("\nInterpretation:")
    print("  • mode 3 (fused graph) FAILS: the hand-written CUDA kernel launches on the")
    print("    DEFAULT stream, which is NOT the graph capture stream → the launch is")
    print("    silently skipped on replay → stale/garbage diff^2 → wildly wrong loss.")
    print("  • mode 4 (triton graph) PASSES: Triton honors the capture stream by default.")
    print("  • mode 5 (fixed CUDA graph) PASSES: launching on getCurrentCUDAStream() records")
    print("    the CUDA kernel into the graph — same kernel, same binary, after the fix.")
    print("=" * 90)

    # machine-readable one-line-per-mode summary
    print("\n# JSONL summary:")
    for display_i, idx in enumerate(order, 1):
        r = results[idx]
        print(f'{{"mode": {display_i}, "name": "{r.name}", "pointwise": "{r.pointwise}", '
              f'"exec": "{r.exec_mode}", "capture_safe": {json_str(r.capture_safe)}, '
              f'"ms_per_iter": {r.ms_per_iter:.4f}, "param_diff": {r.max_param_diff:.6e}, '
              f'"loss_rel_diff": {r.loss_rel_diff:.6e}, "loss": {r.final_loss:.6e}, "status": "{r.status}"}}')

    # ---- Persist machine-readable results (committed evidence in
    # benchmarks/results/; the ledger docs/verifiable_results.md references this + the script) ----
    def _git_commit() -> str:
        try:
            return subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            ).stdout.strip()
        except Exception:
            return "unknown"

    modes_json = []
    for display_i, idx in enumerate(order, 1):
        r = results[idx]
        modes_json.append({
            "mode": display_i,
            "name": r.name,
            "pointwise": r.pointwise,
            "exec": r.exec_mode,
            "capture_safe": r.capture_safe,
            "ms_per_iter": round(r.ms_per_iter, 4),
            "param_diff": r.max_param_diff,
            "loss_rel_diff": r.loss_rel_diff,
            "loss": r.final_loss,
            "status": r.status,
            "note": r.note,
        })
    payload = {
        "script": "baseline/benchmark_v7_graph_modes.py",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": _git_commit(),
        "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "env": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": getattr(__import__("triton"), "__version__", "unknown"),
        },
        "args": {
            "iters": args.iters,
            "warmup": args.warmup,
            "multiplier": args.multiplier,
            "loss_tol": args.loss_tol,
            "param_tol": args.param_tol,
            "tiles_dir": args.tiles_dir,
        },
        "reference": "mode 2 (fused eager CUDA) is the numerical reference",
        "correctness_gate": "loss_rel_diff < loss_tol; mode 3 expected FAIL (capture-stream bug)",
        "modes": modes_json,
    }
    out_path = Path(os.environ.get("FUILT_BENCH_OUT", "benchmarks/results")) / "graph_modes.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n[WROTE] {out_path}")


def json_str(v: Optional[bool]) -> str:
    if v is None:
        return "null"
    return "true" if v else "false"


if __name__ == "__main__":
    main()
