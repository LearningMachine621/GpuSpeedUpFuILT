"""v8 hierarchical fuse/split pipeline (v4-style multi-level tree).

Different from v7's flat N×N fuse (StaticFusionSplitManager) which puts all
256 tiles onto a single 16×16 canvas in one launch. Hierarchical does
4-tuple fuse4 in a tree:

    256 tiles → 64 parents → 16 grandparents → 4 → 1 final canvas

Each level calls the v8 backend's `fuse4` once per 4-tuple. Total launches:
64 + 16 + 4 + 1 = 85 calls per full fuse.

This benchmark answers: is hierarchical viable on single-GPU, or is flat
strictly better here (hierarchical wins on multi-GPU where each level is
independent)?

Run:
  python -m baseline.test_baseline_v8_hierarchical_pipeline
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fuilt.fusion_split.backends import get_backend
from fuilt.fusion_split.write_grads_inplace_v1 import StaticFusionSplitManager

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fuse4_tensors(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor,
    backend, S: int, O: int,
    bigbuf_in: torch.Tensor, bigbuf_out: torch.Tensor,
) -> torch.Tensor:
    """Fuse 4 [S, S] fp32 sub-blocks into 1 [S_out, S_out] fp16 parent.

    Uses pre-allocated bigbufs to avoid per-call allocation. The bigbuf
    layout is:
      - input: A B / C D in a 2S × 2S region at top-left
      - output: S_out × S_out region starting at (2S, 2S)
    """
    P = S - O
    S_out = S + P

    # Place sub-blocks (reset only the regions we touch)
    bigbuf_in.zero_()
    bigbuf_in[0:S, 0:S] = a
    bigbuf_in[0:S, S:2*S] = b
    bigbuf_in[S:2*S, 0:S] = c
    bigbuf_in[S:2*S, S:2*S] = d

    # Offsets: A, B, C, D, OUT top-left positions
    offsets = torch.tensor(
        [[0, 0], [S, 0], [0, S], [S, S], [2*S, 2*S]],
        dtype=torch.int32,
    )

    backend.fuse4(bigbuf_in, bigbuf_out, offsets, S, O)

    # Extract parent (clone to detach from bigbuf_out before next call reuses it)
    return bigbuf_out[2*S:2*S + S_out, 2*S:2*S + S_out].clone()


def hierarchical_fuse(
    tiles: List[torch.Tensor],
    backend,
    S: int, O: int,
) -> torch.Tensor:
    """Multi-level fuse4 tree. N must be a power of 4."""
    n = len(tiles)
    assert n >= 4 and (n & (n - 1) == 0) and (n.bit_length() - 1) % 2 == 0, (
        f"N={n} must be a power of 4 (4, 16, 64, 256, ...)"
    )

    # Pre-allocate bigbufs sized for the deepest level
    S_curr = S
    max_S_out = 0
    s_temp = S
    while s_temp >= S:
        s_temp = 2 * s_temp - O
        max_S_out = max(max_S_out, s_temp)
        if s_temp >= 4 * S:  # safety
            break
    # Actually just size for the largest level (last fuse4 → biggest S_out)
    # Simpler: walk the tree, find max S
    s_walk = S
    levels = []
    while True:
        s_next = 2 * s_walk - O
        levels.append((s_walk, s_next))
        s_walk = s_next
        if (len(tiles) // (4 ** len(levels))) == 1:
            break
    biggest_S = max(s for s, _ in levels)

    H = W = 2 * biggest_S + (2 * biggest_S - O)  # room for 4 sub-blocks + parent
    bigbuf_in = torch.zeros(H, W, device=DEVICE, dtype=torch.float32)
    bigbuf_out = torch.zeros(H, W, device=DEVICE, dtype=torch.float16)

    current = list(tiles)
    s_curr = S
    while len(current) > 1:
        next_level = []
        for i in range(0, len(current), 4):
            parent = fuse4_tensors(
                current[i], current[i+1], current[i+2], current[i+3],
                backend, s_curr, O, bigbuf_in, bigbuf_out,
            )
            next_level.append(parent)
        current = next_level
        s_curr = 2 * s_curr - O
    return current[0]


def benchmark_hierarchical(backend_name: str, n_tiles: int, S: int, O: int,
                            reps: int = 5, warmup: int = 2) -> float:
    """Time hierarchical fuse of n_tiles (power of 4) with given backend."""
    backend = get_backend(backend_name)
    torch.manual_seed(0)
    tiles = [torch.randn(S, S, device=DEVICE, dtype=torch.float32) for _ in range(n_tiles)]

    # Warmup
    for _ in range(warmup):
        _ = hierarchical_fuse(tiles, backend, S, O)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(reps):
        _ = hierarchical_fuse(tiles, backend, S, O)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3  # ms


def benchmark_flat(n_tiles: int, S: int, O: int,
                   reps: int = 5, warmup: int = 2) -> float:
    """Time flat N×N fuse with StaticFusionSplitManager (v7 baseline)."""
    grid = int(n_tiles ** 0.5)
    assert grid * grid == n_tiles, f"n_tiles={n_tiles} must be perfect square"

    workspace = StaticFusionSplitManager(base_grid=grid, s0=S, overlap=O, device=DEVICE)
    torch.manual_seed(0)
    tiles = [torch.randn(S, S, device=DEVICE, dtype=torch.float32) for _ in range(n_tiles)]

    # Warmup
    for _ in range(warmup):
        workspace.reset_canvas()
        for i, t in enumerate(tiles):
            workspace.add_tile_grad_inplace(i, t)
        _ = workspace.finalize_fuse()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(reps):
        workspace.reset_canvas()
        for i, t in enumerate(tiles):
            workspace.add_tile_grad_inplace(i, t)
        _ = workspace.finalize_fuse()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--S", type=int, default=1024)
    parser.add_argument("--O", type=int, default=64)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--out-json", type=str, default="", help="persist results JSON to this path")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA")

    print("=" * 72)
    print(f"Hierarchical (v4-style) vs Flat N×N (v7 StaticFusionSplitManager)")
    print(f"S={args.S}, O={args.O}, reps={args.reps}, GPU isolated")
    print("=" * 72)
    print(f"\n{'N tiles':<10} {'Flat ms':>10} {'hier cuda ms':>14} {'hier triton ms':>16} {'hier pytorch ms':>17}")
    print("-" * 72)

    rows: List[dict] = []
    for n_tiles in [4, 16, 64, 256]:
        try:
            flat_ms = benchmark_flat(n_tiles, args.S, args.O, reps=args.reps)
        except Exception as e:
            flat_ms = float('nan')
            print(f"  flat failed at N={n_tiles}: {e}", file=sys.stderr)

        hier_results = {}
        for backend_name in ["cuda_bigbuf", "triton_bigbuf", "pytorch_inplace"]:
            try:
                hier_results[backend_name] = benchmark_hierarchical(
                    backend_name, n_tiles, args.S, args.O, reps=args.reps
                )
            except Exception as e:
                hier_results[backend_name] = float('nan')
                print(f"  hier {backend_name} failed at N={n_tiles}: {e}", file=sys.stderr)

        rows.append({
            "n_tiles": n_tiles,
            "flat_ms": flat_ms,
            "hier_cuda_ms": hier_results["cuda_bigbuf"],
            "hier_triton_ms": hier_results["triton_bigbuf"],
            "hier_pytorch_ms": hier_results["pytorch_inplace"],
        })
        print(f"{n_tiles:<10} {flat_ms:>10.2f} {hier_results['cuda_bigbuf']:>14.2f} "
              f"{hier_results['triton_bigbuf']:>16.2f} {hier_results['pytorch_inplace']:>17.2f}")

    print("=" * 72)
    print("\nInterpretation:")
    print("  - Flat wins on single-GPU (1 launch vs N/4 × log4(N) launches)")
    print("  - Hierarchical wins on multi-GPU (each level independent, halo-exchange-friendly)")
    print("  - This is why v7 picked flat, and the multi-GPU sharded ILT future work")
    print("    in README uses hierarchical decomposition.")

    if args.out_json:
        _write_out_json(args.out_json, rows, args)


def _write_out_json(path: str, rows: List[dict], args: argparse.Namespace) -> None:
    import triton
    try:
        git_commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                    capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent)).stdout.strip()
    except Exception:
        git_commit = "?"
    payload = {
        "script": "baseline/test_baseline_v8_hierarchical_pipeline.py",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": git_commit,
        "gpu": torch.cuda.get_device_name(0),
        "env": {"torch": torch.__version__, "cuda": torch.version.cuda,
                "triton": triton.__version__},
        "config": {"S": args.S, "O": args.O, "reps": args.reps},
        "rows_ms": rows,
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] wrote {path}")


if __name__ == "__main__":
    main()
