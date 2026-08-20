"""Multi-GPU band-exchange micro-benchmark: halo exchange vs AllReduce.

Models the sharded-ILT communication pattern from README "Future work —
Multi-GPU sharded ILT": each rank owns one patch (``--patch-tiles`` tiles of
1024x1024 fp32). After a local update, only the patch boundary band
(``--band-px`` wide) needs to cross ranks. Four communication modes:

  full_allreduce      naive DDP baseline: NCCL AllReduce the whole patch
                      canvas (1 GiB at 256 tiles).
  band_allreduce      perimeter bands only (post-local-fuse semantics):
                      pack the 4 boundary bands into one buffer, AllReduce.
  band_allreduce_raw  README reference design: AllReduce ALL per-tile bands
                      (no local flat-fuse first) — validates the README
                      "~240 MB/rank @ O=64" projection directly.
  p2p_neighbor        true halo exchange: isend/irecv only with actual grid
                      neighbors (corner ranks 2, edge ranks 3, interior 4).
                      Reported in three phases: pack / comm / unpack — the
                      P0.4 lesson: packing can dominate, so measure it.

Per mode: wall ms (max across ranks = sync time), wire bytes per GPU,
effective GB/s, ratio vs full_allreduce, % of per-iter compute
(0.124 ms/tile x patch_tiles, headline u-bench).

Correctness: p2p fills every send band with the sender's rank id; after one
exchange each recv band must equal the neighbor's id (cross-rank PASS/FAIL).

Run when the host is IDLE (numbers on a busy host are 5-15x off):

  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
      baseline/multigpu_band_exchange.py --patch-tiles 256 --band-px 96 \
      --out-json benchmarks/results/multigpu_band.json
"""

import argparse
import datetime
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Dict, List

import torch
import torch.distributed as dist

TILE = 1024                                   # px per tile side
DIRS = ("up", "down", "left", "right")
# vertical links tag 0, horizontal tag 1: each rank pair shares at most one
# link per category, so tags cannot cross-match between directions.
LINK_TAG = {"up": 0, "down": 0, "left": 1, "right": 1}


def grid_shape(n: int) -> (int, int):
    cols = math.isqrt(n)
    while n % cols:
        cols -= 1
    return n // cols, cols


def neighbors(rank: int, rows: int, cols: int, world: int) -> Dict[str, int]:
    r, c = divmod(rank, cols)
    nb: Dict[str, int] = {}
    if r > 0:
        nb["up"] = rank - cols
    if r + 1 < rows and rank + cols < world:
        nb["down"] = rank + cols
    if c > 0:
        nb["left"] = rank - 1
    if c + 1 < cols and rank + 1 < world:
        nb["right"] = rank + 1
    return nb


def bench(fn, warmup: int, reps: int) -> List[float]:
    """CUDA-event timing per rep; returns the max-across-ranks time (sync time)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    out: List[float] = []
    for _ in range(reps):
        dist.barrier()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        fn()
        t1.record()
        torch.cuda.synchronize()
        t_max = torch.tensor([t0.elapsed_time(t1)], device="cuda")  # ms
        dist.all_reduce(t_max, op=dist.ReduceOp.MAX)
        out.append(float(t_max.item()))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-GPU band-exchange micro-bench")
    p.add_argument("--patch-tiles", type=int, default=256, help="tiles per rank (16 -> 4096px patch)")
    p.add_argument("--band-px", type=int, default=96, help="patch perimeter band width in px")
    p.add_argument("--raw-overlap", type=int, default=64, help="tile overlap O for band_allreduce_raw")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--ms-per-tile", type=float, default=0.124, help="compute anchor (headline u-bench)")
    p.add_argument("--modes", type=str, default="full,band,raw,p2p")
    p.add_argument("--out-json", type=str, default="")
    p.add_argument("--note", type=str, default="")
    args = p.parse_args()

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise SystemExit(
            "launch via torchrun, e.g.:\n"
            "  CUDA_VISIBLE_DEVICES=0..7 torchrun --nproc_per_node=8 "
            "baseline/multigpu_band_exchange.py --patch-tiles 256 --band-px 96"
        )
    dist.init_process_group("nccl", device_id=torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))))
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)

    g = int(round(math.sqrt(args.patch_tiles)))
    P, B, S, O = g * TILE, args.band_px, TILE, args.raw_overlap
    rows, cols = grid_shape(world)
    nb = neighbors(rank, rows, cols, world)

    canvas = torch.randn(P, P, device=dev, dtype=torch.float32)
    band_bytes = B * P * 4                      # bytes per directional band
    send = {d: torch.full(tuple([B, P] if d in ("up", "down") else [P, B]),
                          float(rank), device=dev) for d in DIRS}
    recv = {d: torch.zeros_like(send[d]) for d in DIRS}
    bandbuf = torch.zeros(4 * B * P, device=dev)          # padded 4-dir buffer
    raw_band_px = S * S - (S - 2 * O) ** 2
    rawbuf = torch.randn(args.patch_tiles * raw_band_px, device=dev)

    def band_slices():
        return {"up": canvas[:B, :], "down": canvas[-B:, :],
                "left": canvas[:, :B], "right": canvas[:, -B:]}

    def pack():
        for d, sl in band_slices().items():
            send[d].copy_(sl)

    def unpack():
        for d, sl in band_slices().items():
            sl.copy_(recv[d])

    def p2p_comm():
        # batched submit: one communicator for all ops on this rank pair set
        ops = []
        for d in DIRS:
            if d in nb:
                ops.append(dist.P2POp(dist.isend, send[d], nb[d], tag=LINK_TAG[d]))
        for d in DIRS:
            if d in nb:
                ops.append(dist.P2POp(dist.irecv, recv[d], nb[d], tag=LINK_TAG[d]))
        for req in dist.batch_isend_irecv(ops):
            req.wait()

    def band_allreduce():
        off = 0
        for d in DIRS:
            bandbuf[off:off + B * P].copy_(send[d].reshape(-1))
            off += B * P
        dist.all_reduce(bandbuf)

    full_wire = 2 * (world - 1) / world * P * P * 4
    band_wire = 2 * (world - 1) / world * 4 * band_bytes
    raw_bytes = args.patch_tiles * raw_band_px * 4
    raw_wire = 2 * (world - 1) / world * raw_bytes
    p2p_wire = len(nb) * band_bytes             # per-GPU unidirectional bytes
    compute_ms = args.patch_tiles * args.ms_per_tile

    modes: Dict[str, Dict] = {}

    def record(name: str, fn, wire_bytes: float, extra: Dict = None):
        times = bench(fn, args.warmup, args.reps)
        st = {
            **{k: round(v, 4) for k, v in zip(("mean_ms", "std_ms", "min_ms", "max_ms"),
                                              _stats(times))},
            "wire_mb_per_gpu": round(wire_bytes / 1e6, 2),
        }
        st["eff_gb_s"] = round(wire_bytes / 1e6 / st["mean_ms"], 2) if st["mean_ms"] else 0.0
        st["pct_of_compute"] = round(100.0 * st["mean_ms"] / compute_ms, 2)
        if extra:
            st.update(extra)
        modes[name] = st
        if rank == 0:
            print(f"  {name:<18} {st['mean_ms']:9.3f} ±{st['std_ms']:6.3f} ms"
                  f"  wire {st['wire_mb_per_gpu']:8.1f} MB  {st['eff_gb_s']:7.1f} GB/s"
                  f"  {st['pct_of_compute']:5.1f}% of compute")

    if rank == 0:
        print(f"band exchange: world={world} grid={rows}x{cols} patch={P}x{P}px "
              f"band={B}px tiles/rank={args.patch_tiles} compute={compute_ms:.1f}ms/iter")
        print(f"  neighbors per rank: mine={sorted(nb.items())} (corner=2 edge=3 interior=4)")

    sel = {m.strip() for m in args.modes.split(",")}

    if "full" in sel:
        record("full_allreduce", lambda: dist.all_reduce(canvas), full_wire)

    if "band" in sel:
        record("band_allreduce", band_allreduce, band_wire,
               extra={"note": "perimeter only (post-local-fuse), padded to 4 dirs"})

    if "raw" in sel:
        record("band_allreduce_raw", lambda: dist.all_reduce(rawbuf), raw_wire,
               extra={"note": f"all per-tile bands at O={O} — README reference design"})

    if "p2p" in sel:
        record("p2p_pack", pack, 0.0, extra={"note": "D2D slice copy into send buffers"})
        record("p2p_comm", p2p_comm, p2p_wire,
               extra={"neighbors": {k: int(v) for k, v in nb.items()}})
        record("p2p_unpack", unpack, 0.0, extra={"note": "write reduced band back to canvas"})
        # correctness: re-fill with rank ids (pack bench overwrote them), then check
        for d in DIRS:
            send[d].fill_(float(rank))
        p2p_comm()
        ok = torch.tensor([1 if all(torch.all(recv[d] == float(nb[d])).item()
                                    for d in nb) else 0], device=dev)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        if rank == 0:
            print(f"  p2p correctness: {'PASS' if ok.item() else 'FAIL'}"
                  f" (each recv band == sender rank id)")

    if rank == 0 and args.out_json:
        peer = {str(d): bool(torch.cuda.can_device_access_peer(local_rank, d))
                for d in range(torch.cuda.device_count()) if d != local_rank}
        out = {
            "script": "baseline/multigpu_band_exchange.py",
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "git_commit": _git_commit(),
            "gpu": torch.cuda.get_device_name(0),
            "world_size": world,
            "grid": f"{rows}x{cols}",
            "nccl": ".".join(map(str, torch.cuda.nccl.version())),
            "peer_access_from_local0": peer,
            "config": vars(args),
            "compute_ms_per_iter": compute_ms,
            "analytic": {
                "patch_canvas_mb": round(P * P * 4 / 1e6, 1),
                "band_mb_per_dir": round(band_bytes / 1e6, 2),
                "raw_band_mb_per_rank": round(raw_bytes / 1e6, 1),
                "p2p_send_mb_per_gpu": round(p2p_wire / 1e6, 2),
            },
            "modes": modes,
        }
        if args.note:
            out["note"] = args.note
        path = Path(args.out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {path}")


def _stats(times: List[float]) -> List[float]:
    t = torch.tensor(times)
    return [float(t.mean()), float(t.std() if len(times) > 1 else 0.0),
            float(t.min()), float(t.max())]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
