import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from baseline import test_baseline_v2 as v2mod
from fuilt.ilt.func import initializer
from fuilt.ilt.func import simple as lithosim
from fuilt.ilt.func.levelset import LevelSetCfg, LevelSetILT
from fuilt.pre_work.read_oas2mask import parse_tile_info_from_filename, read_oas_to_real_size_mask


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"


def _discover_tiles(tiles_dir: str) -> List[str]:
    return v2mod.discover_tile_files(tiles_dir)


def _resolve_one_tile(tiles_dir: Optional[str], oas_file: Optional[str], tile_index: int) -> str:
    if oas_file:
        return oas_file
    if not tiles_dir:
        raise ValueError("必须提供 --tiles-dir 或 --oas-file")
    files = _discover_tiles(tiles_dir)
    if not files:
        raise FileNotFoundError(f"目录中没有 oas 文件: {tiles_dir}")
    if tile_index < 0 or tile_index >= len(files):
        raise IndexError(f"tile_index 越界: {tile_index}, 总数={len(files)}")
    return files[tile_index]


def _load_target_and_init_params(oas_file: str) -> Tuple[torch.Tensor, torch.Tensor, int]:
    info = parse_tile_info_from_filename(os.path.basename(oas_file))
    tile_size = int(info.get("tile_size") or 1024)

    target_mask, _ = read_oas_to_real_size_mask(
        oas_file=oas_file,
        target_layers=[(23, 100)],
        target_size=tile_size,
        dtype=torch.float32,
    )

    target_gpu = target_mask.to(DEVICE, dtype=torch.float32)
    target_norm = target_gpu / 255.0 if target_gpu.max() > 1.5 else target_gpu

    _, params_np = initializer.LevelSetImageInit().run(target_norm.detach().cpu().numpy())
    if not isinstance(params_np, torch.Tensor):
        params_np = torch.tensor(params_np, dtype=torch.float32)
    params = params_np.detach().to(DEVICE, dtype=torch.float32).contiguous()
    return target_norm.contiguous(), params, tile_size


def _orig_levelset_loss_grad(
    target_norm: torch.Tensor,
    params: torch.Tensor,
    levelset_cfg_path: str,
    litho_cfg_path: str,
) -> Dict[str, Any]:
    cfg = LevelSetCfg(levelset_cfg_path)
    litho = lithosim.LithoSim(litho_cfg_path)
    solver = LevelSetILT(cfg, litho, device=DEVICE)

    result = solver.compute_grad(target_norm, params.clone())
    grad = result["gradient"].detach()

    return {
        "loss": float(result["loss"]),
        "l2loss": float(result["l2loss"]),
        "pvbl2": float(result["pvbl2"]),
        "pvbloss": float(result["pvbloss"]),
        "grad": grad,
    }


def _v2_loss_grad(
    mode: str,
    target_norm: torch.Tensor,
    params: torch.Tensor,
    litho_cfg_path: str,
    tile_size: int,
) -> Dict[str, Any]:
    litho = lithosim.LithoSim(litho_cfg_path)
    opt_kernels = litho._kernels["focus"].kernels.to(DEVICE, dtype=torch.float32).contiguous()
    kernel_scales = litho._kernels["focus"].scales.to(DEVICE, dtype=torch.float32).contiguous()

    target_density = float(litho._config["TargetDensity"])
    print_steepness = float(litho._config["PrintSteepness"])

    if mode == "pytorch_fft":
        kernel_fft = v2mod._build_kernel_fft(opt_kernels, tile_size, tile_size)
        grad, loss = v2mod._pytorch_fft_grad_loss(
            params.clone().contiguous(),
            target_norm,
            kernel_fft,
            kernel_scales,
            target_density,
            print_steepness,
        )
        return {"loss": float(loss), "grad": grad.detach()}

    if mode == "cuda_op":
        ext = v2mod._load_cuda_extension()
        grad = ext.levelset_step(
            params.clone().contiguous(),
            target_norm,
            opt_kernels,
            kernel_scales,
            target_density,
            print_steepness,
        )
        kernel_fft = v2mod._build_kernel_fft(opt_kernels, tile_size, tile_size)
        _, loss = v2mod._pytorch_fft_grad_loss(
            params.clone().contiguous(),
            target_norm,
            kernel_fft,
            kernel_scales,
            target_density,
            print_steepness,
        )
        return {"loss": float(loss), "grad": grad.detach()}

    raise ValueError(f"未知 mode: {mode}")


def _grad_stats(g: torch.Tensor) -> Dict[str, float]:
    ga = g.abs()
    return {
        "grad_mean_abs": float(ga.mean().item()),
        "grad_max_abs": float(ga.max().item()),
        "grad_min_abs": float(ga.min().item()),
        "grad_l2": float(torch.linalg.norm(g).item()),
    }


def _cosine_similarity(g1: torch.Tensor, g2: torch.Tensor) -> float:
    v1 = g1.reshape(-1)
    v2 = g2.reshape(-1)
    denom = torch.linalg.norm(v1) * torch.linalg.norm(v2)
    if float(denom.item()) < 1e-20:
        return 0.0
    return float((torch.dot(v1, v2) / denom).item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare single-tile original LevelSet loss vs v2 optimized chain loss")
    parser.add_argument("--tiles-dir", type=str, default=os.environ.get("FUILT_TILES_DIR", ""))
    parser.add_argument("--oas-file", type=str, default="")
    parser.add_argument("--tile-index", type=int, default=0)
    parser.add_argument("--mode", type=str, default="pytorch_fft", choices=["pytorch_fft", "cuda_op"])
    parser.add_argument("--litho-cfg", type=str, default=str(CONFIG_DIR / "lithosimple.txt"))
    parser.add_argument("--levelset-cfg", type=str, default=str(CONFIG_DIR / "pylevelset1024.txt"))
    parser.add_argument("--save-json", type=str, default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("该脚本需要 CUDA 环境")

    oas_file = _resolve_one_tile(args.tiles_dir, args.oas_file or None, args.tile_index)
    target_norm, params, tile_size = _load_target_and_init_params(oas_file)

    orig = _orig_levelset_loss_grad(
        target_norm=target_norm,
        params=params,
        levelset_cfg_path=args.levelset_cfg,
        litho_cfg_path=args.litho_cfg,
    )
    opt = _v2_loss_grad(
        mode=args.mode,
        target_norm=target_norm,
        params=params,
        litho_cfg_path=args.litho_cfg,
        tile_size=tile_size,
    )

    loss_orig = float(orig["loss"])
    loss_opt = float(opt["loss"])
    abs_diff = abs(loss_orig - loss_opt)
    rel_diff = abs_diff / max(1e-12, abs(loss_orig))

    orig_grad = orig["grad"]
    opt_grad = opt["grad"]

    result = {
        "tile": {
            "oas_file": oas_file,
            "tile_size": tile_size,
            "mode": args.mode,
        },
        "loss": {
            "original_levelset": loss_orig,
            "v2_optimized": loss_opt,
            "abs_diff": abs_diff,
            "rel_diff_vs_original": rel_diff,
        },
        "grad": {
            "original_stats": _grad_stats(orig_grad),
            "v2_stats": _grad_stats(opt_grad),
            "cosine_similarity": _cosine_similarity(orig_grad, opt_grad),
        },
        "original_detail": {
            "l2loss": float(orig["l2loss"]),
            "pvbl2": float(orig["pvbl2"]),
            "pvbloss": float(orig["pvbloss"]),
        },
        "note": "original_levelset 的 loss 含 PVB 项；v2 优化链路主要对应 nominal 打印项，数值通常不会严格一致。",
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.save_json:
        out = Path(args.save_json)
    else:
        out = ROOT / "outputs" / "compare_single_tile" / f"compare_{Path(oas_file).stem}_{args.mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

'''
python -m baseline.compare_single_tile_levelset_vs_v2 \
  --tiles-dir /data/lyj/FuILT/tiles_from_patch \
  --tile-index 0 \
  --mode pytorch_fft
python -m baseline.compare_single_tile_levelset_vs_v2 \
  --tiles-dir /data/lyj/FuILT/tiles_from_patch \
  --tile-index 0 \
  --mode cuda_op
python -m baseline.compare_single_tile_levelset_vs_v2 \
  --tiles-dir /data/lyj/FuILT/tiles_from_patch \
  --tile-index 0 \
  --mode pytorch_fft \
  --save-json outputs/compare_single_tile/my_compare.json

'''