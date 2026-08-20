"""v7 graph capture POC (P1 follow-up): can we capture one full v7 iteration?

The v7 main loop mixes CPU sync ops (loss.item(), print) with GPU compute,
which prevents direct graph capture. This POC strips the CPU bits and tests
whether the pure-GPU path (LevelSet + Fuse + Split + Apply) is
capture-friendly.

If this works, the path to full integration is:
  1. Add a `static_loss_accum` GPU tensor (replace `loss_sum += float(...)`)
  2. Move print/eval outside the captured region
  3. Add `FUILT_USE_GRAPH` env var to v7 main loop
"""
import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fuilt.fusion_split.write_grads_inplace_v1 import StaticFusionSplitManager
from fuilt.ilt.func import simple as lithosim
from fuilt.ilt.func import evaluation  # noqa: F401 (ensures import works)
from fuilt.pre_work.read_oas2mask import (
    parse_tile_info_from_filename,
    read_oas_to_real_size_mask,
)
from baseline.test_baseline_v7_fuselevelset import (
    _build_kernel_fft,
    _cpu_init_single_tile,
    _pytorch_fft_grad_loss,
    discover_tile_files,
    infer_grid_size,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"


def setup_pipeline(oas_dir: str, multiplier: int = 16):
    """Mimics v7 setup: load tiles, SDF, H2D, build kernels, workspace."""
    tile_files = discover_tile_files(oas_dir)
    first_info = parse_tile_info_from_filename(tile_files[0])
    tile_size = int(first_info.get("tile_size") or 1024)
    overlap = int(first_info.get("overlap") or 64)

    print(f"Loading {len(tile_files)} tiles...")
    target_masks = []
    for f in tile_files:
        mask, _ = read_oas_to_real_size_mask(f, target_layers=[(23, 100)], target_size=tile_size, dtype=torch.float32)
        target_masks.append(mask)
    target_masks = target_masks * multiplier
    base_grid = infer_grid_size(len(target_masks))
    print(f"  Expanded to {len(target_masks)} tiles ({base_grid}x{base_grid} grid)")

    print("CPU SDF compute (ThreadPool)...")
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
        results = list(ex.map(_cpu_init_single_tile, target_masks))

    print("H2D...")
    params_list: List[torch.Tensor] = []
    target_cuda_list: List[torch.Tensor] = []
    for target_np, params_np in results:
        target_cuda_list.append(torch.from_numpy(target_np).pin_memory().to(DEVICE, dtype=torch.float32, non_blocking=True).contiguous())
        params_list.append(torch.from_numpy(params_np).pin_memory().to(DEVICE, dtype=torch.float32, non_blocking=True).contiguous())

    litho = lithosim.LithoSim(str(CONFIG_DIR / "lithosimple.txt"))
    opt_kernels = litho._kernels["focus"].kernels.to(DEVICE, dtype=torch.float32).contiguous()
    kernel_scales = litho._kernels["focus"].scales.to(DEVICE, dtype=torch.float32).contiguous()
    kernel_fft = _build_kernel_fft(opt_kernels, tile_size, tile_size)

    # P0.1 combined kernel
    combined_kernel_fft = (kernel_scales.view(-1, 1, 1).to(kernel_fft.dtype) * kernel_fft).sum(dim=0)

    target_density = float(litho._config["TargetDensity"])
    print_steepness = float(litho._config["PrintSteepness"])

    workspace = StaticFusionSplitManager(base_grid=base_grid, s0=tile_size, overlap=overlap, device=DEVICE)

    return {
        "params_list": params_list,
        "target_cuda_list": target_cuda_list,
        "workspace": workspace,
        "kernel_fft": kernel_fft,
        "kernel_scales": kernel_scales,
        "combined_kernel_fft": combined_kernel_fft,
        "target_density": target_density,
        "print_steepness": print_steepness,
        "tile_size": tile_size,
        "learning_rate": 0.025,
    }


def run_one_iter(ctx: dict, loss_accum: torch.Tensor) -> None:
    """One v7 iteration, pure GPU (no .item(), no print).

    loss_accum: a GPU tensor that we add per-tile loss to (replaces loss_sum).
    """
    workspace = ctx["workspace"]
    workspace.reset_canvas()

    for tile_idx, (params, target) in enumerate(zip(ctx["params_list"], ctx["target_cuda_list"])):
        grad, loss_gpu = _pytorch_fft_grad_loss(
            params, target, ctx["kernel_fft"], ctx["kernel_scales"],
            ctx["target_density"], ctx["print_steepness"],
            combined_kernel_fft=ctx["combined_kernel_fft"],
            return_gpu_loss=True,
        )
        workspace.add_tile_grad_inplace(tile_idx, grad.to(torch.float32))
        loss_accum.add_(loss_gpu)  # GPU accumulator, no .item() sync

    fused = workspace.finalize_fuse()
    split_grads = workspace.split(fused)
    for i in range(len(ctx["params_list"])):
        ctx["params_list"][i].add_(-ctx["learning_rate"] * split_grads[i])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles-dir", default=os.environ.get("FUILT_TILES_DIR", "/data/lyj/FuILT/tiles_from_patch"))
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--multiplier", type=int, default=16)
    args = parser.parse_args()

    print("=" * 60)
    print("v7 Graph Capture POC")
    print("=" * 60)

    ctx = setup_pipeline(args.tiles_dir, args.multiplier)
    loss_accum = torch.zeros(1, device=DEVICE, dtype=torch.float32)

    # === Eager mode baseline ===
    print(f"\n[Eager] {args.iters} iters...")
    # Reset params to deterministic state for fair comparison
    torch.manual_seed(42)
    for p in ctx["params_list"]:
        p.normal_()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        run_one_iter(ctx, loss_accum)
    torch.cuda.synchronize()
    eager_s = time.perf_counter() - t0
    print(f"  Eager: {eager_s:.3f} s  ({eager_s/args.iters*1000:.2f} ms/iter)")

    # Snapshot final params (for graph equivalence check)
    eager_final = [p.clone() for p in ctx["params_list"]]

    # === Graph capture mode ===
    # Reset params to the SAME initial state (so we can diff against eager_final)
    torch.manual_seed(42)
    for p in ctx["params_list"]:
        p.normal_()
    torch.cuda.synchronize()

    print(f"\n[Graph] warmup {args.warmup} iters (must precede capture)...")
    for _ in range(args.warmup):
        run_one_iter(ctx, loss_accum)
    torch.cuda.synchronize()

    print("[Graph] capturing one iter...")
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run_one_iter(ctx, loss_accum)
        print("  Capture: OK")
    except Exception as e:
        print(f"  Capture FAILED: {type(e).__name__}: {e}")
        return

    print(f"[Graph] replay {args.iters} times...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        g.replay()
    torch.cuda.synchronize()
    graph_s = time.perf_counter() - t0
    print(f"  Graph replay: {graph_s:.3f} s  ({graph_s/args.iters*1000:.2f} ms/iter)")
    print(f"  Speedup: {eager_s/graph_s:.2f}×")

    # === Numerical equivalence check ===
    print("\n[Equivalence] graph vs eager final params...")
    max_diff = 0.0
    for p_graph, p_eager in zip(ctx["params_list"], eager_final):
        d = (p_graph - p_eager).abs().max().item()
        max_diff = max(max_diff, d)
    print(f"  max abs diff: {max_diff:.6e}")
    print(f"  equivalence: {'PASS' if max_diff < 1e-3 else 'FAIL'}")

    print("=" * 60)
    print(f"Eager:   {eager_s/args.iters*1000:>7.2f} ms/iter")
    print(f"Graph:   {graph_s/args.iters*1000:>7.2f} ms/iter")
    print(f"Speedup: {eager_s/graph_s:>7.2f}×")
    print("=" * 60)


if __name__ == "__main__":
    main()
