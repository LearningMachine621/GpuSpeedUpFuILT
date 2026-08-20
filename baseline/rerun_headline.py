#!/usr/bin/env python3
"""Rerun the README headline E2E table with persisted raw evidence.

Runs each baseline / v1..v7 / P0.x variant `--repeats` times on ONE isolated
GPU (no concurrent workloads), parses each script's stdout, and writes a JSON
per-variant raw reps + mean/std + full metadata to benchmarks/results/.

This is the reproducible evidence for the README headline table. It is a thin
driver: it does NOT re-implement any pipeline math, it launches the existing
`baseline/test_baseline_v*.py` scripts as subprocesses exactly like a human
would, and records the timing lines they already print.

Usage (from repo root):
    PYTHONPATH=. python -m baseline.rerun_headline [--gpu 7] [--repeats 3]
    # or pin the GPU via CUDA_VISIBLE_DEVICES

Config honored (env, passed through to every child):
    FUILT_TILES_DIR  FUILT_ITERS  FUILT_EVAL_INTERVAL  FUILT_LR
    FUILT_BENCH_WARMUP  FUILT_TILE_MULTIPLIER
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
BENCH_OUT = ROOT / "benchmarks" / "results"

# ---------------------------------------------------------------------------
# Parse patterns — identical to run_benchmark_matrix.py so evidence is comparable
# ---------------------------------------------------------------------------
RE_PURE = re.compile(r"流水线总纯耗时\s*\(剔除Eval\):\s*([0-9]+(?:\.[0-9]+)?)\s*s")
RE_TOTAL = re.compile(r"脚本端到端总耗时\s*\(含Eval\):\s*([0-9]+(?:\.[0-9]+)?)\s*s")
RE_VRAM = re.compile(r"峰值显存.*?:\s*([0-9]+(?:\.[0-9]+)?)\s*MB")
RE_MICRO = re.compile(r"Micro-Bench\):\s*([0-9]+(?:\.[0-9]+)?)\s*ms\s*/\s*Tile")
RE_LOSS = re.compile(r"avg_loss=([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)")
RE_TILES_SPEC = re.compile(r"Tile 规格:\s*(\d+)\s*个\s*\((\d+)x(\d+)")
RE_TILES_CLONE = re.compile(r"强制克隆为\s*(\d+)\s*个\s*Tiles")


def _parse_float(regex: re.Pattern, text: str) -> Optional[float]:
    m = regex.search(text)
    return float(m.group(1)) if m else None


def _parse_last_float(regex: re.Pattern, text: str) -> Optional[float]:
    ms = regex.findall(text)
    return float(ms[-1]) if ms else None


# ---------------------------------------------------------------------------
# Variant matrix (headline table + the v5 AMP-state audit point)
# ---------------------------------------------------------------------------
@dataclass
class Variant:
    name: str
    module: str
    mode: Optional[str]  # None => script takes no --mode
    env: Dict[str, str] = field(default_factory=dict)
    note: str = ""


VARIANTS: List[Variant] = [
    Variant("v1", "baseline.test_baseline_v1", None, {},
            "naive eager per-tile baseline (hardcoded 256 tiles, no warmup)"),
    Variant("v3", "baseline.test_baseline_v3_H2D_init", "pytorch_fft", {},
            "H2D / pinned-memory overlap"),
    Variant("v5_amp_on", "baseline.test_baseline_v5_amp_graph", "pytorch_fft",
            {"FUILT_USE_AMP": "1"}, "AMP fp16 + CUDA graph capture, FUILT_USE_AMP=1 (README claim)"),
    Variant("v5_amp_off", "baseline.test_baseline_v5_amp_graph", "pytorch_fft", {},
            "v5 script default: pytorch_fft mode => FUILT_USE_AMP=0 (recorded state)"),
    Variant("v6", "baseline.test_baseline_v6_full_batch", "pytorch_fft", {},
            "full-batch CUDA graph capture"),
    Variant("v7_cuda", "baseline.test_baseline_v7_fuselevelset", "pytorch_fft",
            {"FUILT_USE_TRITON_POINTWISE": "0"}, "v7 fused pointwise (hand CUDA kernel)"),
    Variant("v7_triton", "baseline.test_baseline_v7_fuselevelset", "pytorch_fft",
            {"FUILT_USE_TRITON_POINTWISE": "1"}, "v7 fused pointwise (Triton port)"),
    Variant("p01", "baseline.test_baseline_v7_fuselevelset", "pytorch_fft",
            {"FUILT_USE_COMBINED_KERNEL": "1"}, "v7 + P0.1 combined-kernel FFT (24 ifft2 -> 1)"),
    Variant("p01p03", "baseline.test_baseline_v7_fuselevelset", "pytorch_fft",
            {"FUILT_USE_COMBINED_KERNEL": "1", "FUILT_USE_ASYNC_D2H": "1"},
            "v7 + P0.1 + P0.3 async D2H overlap (headline 22.8x)"),
    Variant("p04", "baseline.test_baseline_v7_fuselevelset", "pytorch_fft",
            {"FUILT_USE_COMBINED_KERNEL": "1", "FUILT_USE_ASYNC_D2H": "1",
             "FUILT_USE_BATCH_H2D": "1"},
            "v7 + P0.1 + P0.3 + P0.4 batch H2D (regressed — kept as cautionary tale)"),
]


# ---------------------------------------------------------------------------
def _pick_gpu(preferred: Optional[str]) -> str:
    """Return a CUDA device id; default = first idle GPU (respects host state)."""
    if preferred:
        return preferred
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        ).stdout
        for line in out.strip().splitlines():
            idx, util = [p.strip() for p in line.split(",")]
            if int(util) == 0:
                return idx
        return "0"
    except Exception:
        return "0"


def _gpu_meta(gpu: str) -> Dict[str, Any]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,utilization.gpu,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        ).stdout
        for line in out.strip().splitlines():
            idx, name, total, util, used = [p.strip() for p in line.split(",")]
            if idx == gpu:
                return {"index": int(gpu), "name": name,
                        "vram_total_mb": int(float(total.split()[0])),
                        "util_pct_at_start": int(util.split()[0]), "used_mb": int(used.split()[0])}
    except Exception:
        pass
    return {"index": int(gpu), "name": "unknown"}


def _run_one(module: str, mode: Optional[str], env: Dict[str, str], cwd: Path,
             timeout_s: int = 900) -> Dict[str, Any]:
    cmd = [sys.executable, "-m", module]
    if mode:
        cmd += ["--mode", mode]
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True,
                          timeout=timeout_s)
    text = proc.stdout + "\n" + proc.stderr
    if proc.returncode != 0:
        # Keep going — some variants may be expected to fail (e.g. P0.4 under OOM);
        # the raw stdout is preserved so the failure is verifiable, not hidden.
        return {
            "returncode": proc.returncode,
            "error_tail": proc.stderr[-2000:] if proc.stderr else proc.stdout[-2000:],
            "raw": text[-8000:],
        }
    tiles_m = RE_TILES_SPEC.search(text) or RE_TILES_CLONE.search(text)
    return {
        "returncode": 0,
        "pure_pipeline_s": _parse_float(RE_PURE, text),
        "total_script_s": _parse_float(RE_TOTAL, text),
        "peak_vram_mb": _parse_float(RE_VRAM, text),
        "micro_ms_per_tile": _parse_float(RE_MICRO, text),
        "final_loss": _parse_last_float(RE_LOSS, text),
        "num_tiles": int(tiles_m.group(1)) if tiles_m else None,
        "graph_replay": bool(re.search(r"replay|极速回放", text, re.IGNORECASE)),
    }


def _mean_std(values: List[float]) -> Dict[str, Any]:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": round(statistics.mean(vals), 6),
            "std": round(statistics.stdev(vals), 6) if len(vals) >= 2 else 0.0,
            "n": len(vals)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun headline E2E table, persist raw JSON.")
    parser.add_argument("--gpu", type=str, default=os.environ.get("FUILT_GPU", ""))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=str,
                        default=str(BENCH_OUT / "headline_e2e.json"))
    parser.add_argument("--only", type=str, default="",
                        help="comma-separated variant names to run (default: all)")
    args = parser.parse_args()

    gpu = _pick_gpu(args.gpu)
    gpu_meta = _gpu_meta(gpu)
    print(f"[GPU] using device {gpu} ({gpu_meta.get('name')}), isolated via CUDA_VISIBLE_DEVICES")

    import torch  # noqa: E402
    import triton  # noqa: E402

    iters = int(os.environ.get("FUILT_ITERS", "20"))
    warmup = int(os.environ.get("FUILT_BENCH_WARMUP", "10"))
    multiplier = int(os.environ.get("FUILT_TILE_MULTIPLIER", "16"))
    tiles_dir = os.environ.get("FUILT_TILES_DIR", "/data/lyj/FuILT/tiles_from_patch")
    mode = "pytorch_fft"

    git_commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, cwd=str(ROOT)).stdout.strip()

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    variants = [v for v in VARIANTS if not only or v.name in only]

    results: List[Dict[str, Any]] = []
    for variant in variants:
        print(f"\n===== {variant.name} ({variant.module}) =====")
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": gpu,
            "PYTHONPATH": str(ROOT),
            "FUILT_TILES_DIR": tiles_dir,
            "FUILT_ITERS": str(iters),
            "FUILT_EVAL_INTERVAL": os.environ.get("FUILT_EVAL_INTERVAL", "10"),
            "FUILT_LR": os.environ.get("FUILT_LR", "0.025"),
            "FUILT_BENCH_WARMUP": str(warmup),
            "FUILT_TILE_MULTIPLIER": str(multiplier),
            # Redirect v7's committed v7_profile.json so per-rep runs don't clobber it.
            "FUILT_BENCH_OUT": "/tmp/fuilt_headline_bench",
            "FUILT_USE_GRAPH": "0",  # headline rows are eager
        })
        env.update(variant.env)

        reps: List[Dict[str, Any]] = []
        for rep in range(1, args.repeats + 1):
            t0 = time.time()
            r = _run_one(variant.module, variant.mode, env, ROOT)
            r["rep"] = rep
            r["wall_s"] = round(time.time() - t0, 2)
            reps.append(r)
            pure = r.get("pure_pipeline_s")
            loss = r.get("final_loss")
            print(f"  rep{rep}: pure={pure}s vram={r.get('peak_vram_mb')} "
                  f"micro={r.get('micro_ms_per_tile')} loss={loss} ({r.get('wall_s')}s wall)")

        failed = [r for r in reps if r.get("returncode", 0) != 0]
        entry: Dict[str, Any] = {
            "name": variant.name,
            "module": variant.module,
            "mode": mode,
            "env_gates": variant.env,
            "note": variant.note,
            "n_reps": len(reps),
            "reps": reps,
            "mean_std": {
                "pure_pipeline_s": _mean_std([r.get("pure_pipeline_s") for r in reps]),
                "peak_vram_mb": _mean_std([r.get("peak_vram_mb") for r in reps]),
                "micro_ms_per_tile": _mean_std([r.get("micro_ms_per_tile") for r in reps]),
            },
            "n_failed": len(failed),
            "failed_reps": [r["rep"] for r in failed],
        }
        results.append(entry)
        pure_vals = [r.get("pure_pipeline_s") for r in reps if r.get("pure_pipeline_s")]
        if pure_vals:
            m = statistics.mean(pure_vals)
            s = statistics.stdev(pure_vals) if len(pure_vals) >= 2 else 0.0
            print(f"  => {variant.name}: pure={m:.3f} ± {s:.3f} s ({len(pure_vals)} reps)")

    # Speedup vs v1 baseline (mean pure pipeline time).
    v1_entry = next((e for e in results if e["name"] == "v1"), None)
    v1_mean = (v1_entry["mean_std"]["pure_pipeline_s"]["mean"] if v1_entry else None)
    for entry in results:
        e = entry["mean_std"]["pure_pipeline_s"]
        entry["speedup_vs_v1"] = (round(v1_mean / e["mean"], 3)
                                  if v1_mean and e.get("mean") else None)

    payload: Dict[str, Any] = {
        "script": "baseline/rerun_headline.py",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": git_commit,
        "gpu": gpu_meta,
        "cuda_visible_devices": gpu,
        "env": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": triton.__version__,
            "python": sys.version.split()[0],
        },
        "args": {
            "iters": iters,
            "warmup": warmup,
            "multiplier": multiplier,
            "mode": mode,
            "tiles_dir": tiles_dir,
            "repeats": args.repeats,
            "eager": True,
        },
        "note": ("Independent re-run of the README headline table. Each variant is the "
                 "existing baseline/test_baseline_v*.py script launched via subprocess; "
                 "timing lines are parsed from stdout. v5 measured in BOTH AMP states "
                 "(README claims fp16; the script default for pytorch_fft mode is AMP=0)."),
        "variants": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] wrote {out_path}")

    # Human summary table.
    print(f"\n{'variant':<12}{'pure(s)':>10}{'±':>8}{'vram(MB)':>10}{'speedup':>9}")
    for e in results:
        ms = e["mean_std"]["pure_pipeline_s"]
        if ms.get("mean") is None:
            print(f"{e['name']:<12}{'FAILED':>10}")
            continue
        vram = e["mean_std"]["peak_vram_mb"]
        sp = e.get("speedup_vs_v1")
        print(f"{e['name']:<12}{ms['mean']:>10.3f}{ms['std']:>8.3f}"
              f"{str(vram.get('mean')):>10}{str(sp):>9}")


if __name__ == "__main__":
    main()
