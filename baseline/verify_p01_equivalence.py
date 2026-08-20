"""Independent A/B numerical check for P0.1 (no shared code with the pipeline).

    OLD: aerial_old = Σ_k s[k]·Re{IFFT2(mask_fft ⊙ K_k)}    (24× ifft2 + add_)
    NEW: aerial_new = Re{IFFT2(mask_fft ⊙ Σ_k s[k]·K_k)}    (1× ifft2)

Mathematically strictly equal (IFFT linearity, real scales — proof in
docs/p01_ifft_linearity_proof.md). Bit-equality is NOT implied: the two paths
sum in different domains (image vs frequency) with different rounding orders,
and the FFT count differs. This script measures the actual difference.

Data mimics the pipeline structure: K=24 real non-negative PSFs (hermitian
FFT), real dose scales in [0,1], real mask in [-1,1], norm="forward",
complex64 (fp32) — plus a complex128 (fp64) arm to confirm the math
(difference should collapse toward ~1e-12 relative) and a 1-ulp perturbation
sanity arm proving the comparator is not structurally zero.

Correctness checks are valid on a busy GPU (contention affects latency, not
values). Regenerate:
  CUDA_VISIBLE_DEVICES=<any> python -m baseline.verify_p01_equivalence \
      --out-json benchmarks/results/p01_equivalence.json
"""

import argparse
import datetime
import json
import os
import subprocess
from pathlib import Path

import torch

K = 24
N = 1024


def build_inputs(seed: int, cdtype: torch.dtype, rdtype: torch.dtype):
    g = torch.Generator(device="cuda").manual_seed(seed)
    psf = torch.rand(K, N, N, device="cuda", generator=g)            # real, >= 0
    kern_fft = torch.fft.fft2(psf.to(cdtype), norm="forward")        # hermitian
    scales = torch.rand(K, device="cuda", generator=g).to(rdtype)    # real dose weights
    mask = (torch.rand(N, N, device="cuda", generator=g) * 2 - 1).to(rdtype)
    return mask, kern_fft, scales


def old_path(mask_fft, kern_fft, scales):
    """24× ifft2 + image-domain accumulation (pre-P0.1)."""
    aerial = torch.zeros_like(mask_fft.real)
    for k in range(kern_fft.shape[0]):
        conv = torch.fft.ifft2(mask_fft * kern_fft[k], norm="forward").real
        aerial.add_(conv, alpha=float(scales[k].item()))
    return aerial


def new_path(mask_fft, kern_fft, scales):
    """frequency-domain sum + 1× ifft2 (P0.1)."""
    combined = (scales.view(-1, 1, 1) * kern_fft).sum(dim=0)
    return torch.fft.ifft2(mask_fft * combined, norm="forward").real


def run_arm(cdtype, rdtype, seeds, perturb=False):
    rows = []
    for seed in seeds:
        mask, kern_fft, scales = build_inputs(seed, cdtype, rdtype)
        mask_fft = torch.fft.fft2(mask.to(cdtype), norm="forward")
        a_old = old_path(mask_fft, kern_fft, scales)
        a_new = new_path(mask_fft, kern_fft, scales)
        if perturb:
            flat = a_new.flatten()
            i = a_new.numel() // 2
            nxt = torch.nextafter(flat[i], torch.full_like(flat[i:i + 1], float("inf")))
            flat[i] = nxt
        diff = (a_old - a_new).abs()
        scale = a_old.abs().max().item()
        rows.append({
            "seed": seed,
            "max_abs_diff": float(diff.max().item()),
            "n_nonzero": int((diff > 0).sum().item()),
            "n_elements": a_old.numel(),
            "rel_vs_max_abs": float(diff.max().item() / scale) if scale else 0.0,
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--out-json", type=str, default="")
    args = p.parse_args()
    assert torch.cuda.is_available(), "needs a CUDA device"
    torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES picks the physical card
    dev = torch.cuda.current_device()
    print(f"device: {torch.cuda.get_device_name(dev)}  K={K} N={N} norm=forward seeds={args.seeds}")

    seeds = list(range(args.seeds))
    fp32 = run_arm(torch.complex64, torch.float32, seeds)
    fp64 = run_arm(torch.complex128, torch.float64, seeds)
    sanity = run_arm(torch.complex64, torch.float32, seeds[:1], perturb=True)

    def fmt(rows):
        return " | ".join(f"s{r['seed']}: diff={r['max_abs_diff']:.3e} "
                          f"nz={r['n_nonzero']}/{r['n_elements']}" for r in rows)

    print(f"fp32 (complex64) : {fmt(fp32)}")
    print(f"fp64 (complex128): {fmt(fp64)}")
    ok = all(r["max_abs_diff"] > 0 for r in sanity)
    print(f"sanity (1-ulp perturbation on NEW): {fmt(sanity)} -> comparator "
          f"{'OK (detects nonzero)' if ok else 'BROKEN'}")

    out = {
        "script": "baseline/verify_p01_equivalence.py",
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": _git_commit(),
        "gpu": torch.cuda.get_device_name(dev),
        "torch": torch.__version__,
        "config": {"K": K, "N": N, "norm": "forward", "seeds": args.seeds},
        "fp32_complex64": fp32,
        "fp64_complex128": fp64,
        "sanity_ulp_perturbation": sanity,
        "conclusion": (
            "OLD (24x ifft2, image-domain accumulation) vs NEW (frequency-domain "
            "sum, 1x ifft2): strictly equal in exact arithmetic by IFFT linearity; "
            "measured differences below are finite-precision only. Bit-equality is "
            "a per-data measurement, not a theorem."
        ),
    }
    if args.out_json:
        path = Path(args.out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {path}")


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
