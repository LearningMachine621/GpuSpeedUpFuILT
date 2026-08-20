import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from baseline import test_baseline_v2 as v2mod
from fuilt.ilt.func import initializer
from fuilt.ilt.func import simple
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


def _load_target_and_init_params(oas_file: str) -> Tuple[torch.Tensor, torch.Tensor]:
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
    return target_norm.contiguous(), params


def _grad_image(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    image4d = image.view([-1, 1, image.shape[-2], image.shape[-1]])
    padded = F.pad(image4d, (1, 1, 1, 1), mode="replicate")[:, 0].detach()
    grad_x = (padded[:, 2:, 1:-1] - padded[:, :-2, 1:-1]) / 2.0
    grad_y = (padded[:, 1:-1, 2:] - padded[:, 1:-1, :-2]) / 2.0
    return grad_x.view(image4d.shape), grad_y.view(image4d.shape)


class _EquivalentBinarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, levelset: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(levelset)
        mask = torch.zeros_like(levelset)
        mask[levelset < 0] = 1.0
        return mask

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor) -> torch.Tensor:
        grad_output = grad_outputs[0]
        (levelset,) = ctx.saved_tensors
        grad_x, grad_y = _grad_image(levelset)
        l2norm = torch.sqrt(grad_x**2 + grad_y**2)
        return -l2norm * grad_output


def _build_filter(cfg: LevelSetCfg, device: torch.device) -> torch.Tensor:
    flt = torch.zeros([cfg["TileSizeX"], cfg["TileSizeY"]], dtype=torch.float32, device=device)
    flt[
        cfg["OffsetX"] : cfg["OffsetX"] + cfg["ILTSizeX"],
        cfg["OffsetY"] : cfg["OffsetY"] + cfg["ILTSizeY"],
    ] = 1.0
    return flt


def _orig_levelset_loss_grad(
    target_norm: torch.Tensor,
    params: torch.Tensor,
    levelset_cfg_path: str,
    litho_cfg_path: str,
) -> Dict[str, Any]:
    cfg = LevelSetCfg(levelset_cfg_path)
    litho_model = simple.LithoSim(litho_cfg_path)
    solver = LevelSetILT(cfg, litho_model, device=DEVICE)

    result = solver.compute_grad(target_norm, params.clone())
    return {
        "loss": float(result["loss"]),
        "l2loss": float(result["l2loss"]),
        "pvbl2": float(result["pvbl2"]),
        "pvbloss": float(result["pvbloss"]),
        "grad": result["gradient"].detach(),
    }


def _fft_equivalent_loss_grad(
    target_norm: torch.Tensor,
    params: torch.Tensor,
    levelset_cfg_path: str,
    litho_cfg_path: str,
) -> Dict[str, Any]:
    cfg = LevelSetCfg(levelset_cfg_path)
    litho_model = simple.LithoSim(litho_cfg_path)

    target_norm = target_norm.to(dtype=torch.float32, device=DEVICE)
    flt = _build_filter(cfg, DEVICE)

    x = params.clone().detach().requires_grad_(True)
    masked = x * flt
    mask = _EquivalentBinarize.apply(masked)

    k = litho_model._kernels
    c = litho_model._config

    aerial_nom = simple._LithoSim.apply(
        mask,
        c["DoseNom"],
        k["focus"].kernels,
        k["focus"].scales,
        c["KernelNum"],
        k["combo CT focus"].kernels,
        k["combo CT focus"].scales,
        1,
        k["combo focus"].kernels,
        k["combo focus"].scales,
        1,
    )
    aerial_max = simple._LithoSim.apply(
        mask,
        c["DoseMax"],
        k["focus"].kernels,
        k["focus"].scales,
        c["KernelNum"],
        k["combo CT focus"].kernels,
        k["combo CT focus"].scales,
        1,
        k["combo focus"].kernels,
        k["combo focus"].scales,
        1,
    )
    aerial_min = simple._LithoSim.apply(
        mask,
        c["DoseMin"],
        k["defocus"].kernels,
        k["defocus"].scales,
        c["KernelNum"],
        k["combo CT defocus"].kernels,
        k["combo CT defocus"].scales,
        1,
        k["combo defocus"].kernels,
        k["combo defocus"].scales,
        1,
    )

    printed_nom = torch.sigmoid(c["PrintSteepness"] * (aerial_nom - c["TargetDensity"]))
    printed_max = torch.sigmoid(c["PrintSteepness"] * (aerial_max - c["TargetDensity"]))
    printed_min = torch.sigmoid(c["PrintSteepness"] * (aerial_min - c["TargetDensity"]))

    l2loss = F.mse_loss(printed_nom, target_norm, reduction="sum")
    pvbl2 = F.mse_loss(printed_max, target_norm, reduction="sum") + F.mse_loss(printed_min, target_norm, reduction="sum")
    pvbloss = F.mse_loss(printed_max, printed_min, reduction="sum")

    loss = l2loss + cfg["WeightPVBL2"] * pvbl2 + cfg["WeightPVBand"] * pvbloss
    loss.backward()

    if x.grad is None:
        raise RuntimeError("FFT 等价梯度计算失败：x.grad is None")
    grad = x.grad.detach()
    return {
        "loss": float(loss.item()),
        "l2loss": float(l2loss.item()),
        "pvbl2": float(pvbl2.item()),
        "pvbloss": float(pvbloss.item()),
        "grad": grad,
    }


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


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare original LevelSet compute_grad and FFT-equivalent compute_grad on a single tile"
    )
    parser.add_argument("--tiles-dir", type=str, default=os.environ.get("FUILT_TILES_DIR", ""))
    parser.add_argument("--oas-file", type=str, default="")
    parser.add_argument("--tile-index", type=int, default=0)
    parser.add_argument("--litho-cfg", type=str, default=str(CONFIG_DIR / "lithosimple.txt"))
    parser.add_argument("--levelset-cfg", type=str, default=str(CONFIG_DIR / "pylevelset1024.txt"))
    parser.add_argument("--save-json", type=str, default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("该脚本需要 CUDA 环境")

    oas_file = _resolve_one_tile(args.tiles_dir, args.oas_file or None, args.tile_index)
    target_norm, params = _load_target_and_init_params(oas_file)

    orig = _orig_levelset_loss_grad(
        target_norm=target_norm,
        params=params,
        levelset_cfg_path=args.levelset_cfg,
        litho_cfg_path=args.litho_cfg,
    )
    fft_eq = _fft_equivalent_loss_grad(
        target_norm=target_norm,
        params=params,
        levelset_cfg_path=args.levelset_cfg,
        litho_cfg_path=args.litho_cfg,
    )

    loss_orig = float(orig["loss"])
    loss_fft = float(fft_eq["loss"])
    loss_abs_diff = abs(loss_orig - loss_fft)
    loss_rel_diff = loss_abs_diff / max(1e-12, abs(loss_orig))

    g_orig = orig["grad"]
    g_fft = fft_eq["grad"]

    result = {
        "tile": {
            "oas_file": oas_file,
            "tile_index": args.tile_index,
        },
        "loss_compare": {
            "original_levelset": loss_orig,
            "fft_equivalent": loss_fft,
            "abs_diff": loss_abs_diff,
            "rel_diff_vs_original": loss_rel_diff,
        },
        "detail_compare": {
            "l2loss_original": float(orig["l2loss"]),
            "l2loss_fft": float(fft_eq["l2loss"]),
            "pvbl2_original": float(orig["pvbl2"]),
            "pvbl2_fft": float(fft_eq["pvbl2"]),
            "pvbloss_original": float(orig["pvbloss"]),
            "pvbloss_fft": float(fft_eq["pvbloss"]),
        },
        "grad_compare": {
            "original_stats": _grad_stats(g_orig),
            "fft_stats": _grad_stats(g_fft),
            "cosine_similarity": _cosine_similarity(g_orig, g_fft),
            "max_abs_diff": _max_abs_diff(g_orig, g_fft),
        },
        "note": "该 FFT 版本按原版 compute_grad 的 forward/backward 公式重写，理论上应与原版高度一致。",
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.save_json:
        out = Path(args.save_json)
    else:
        out = ROOT / "outputs" / "compare_single_tile" / f"compare_levelset_vs_fft_{Path(oas_file).stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
'''
python -m baseline.compare_single_tile_levelset_vs_fft_equivalent \
  --tiles-dir /data/lyj/FuILT/tiles_from_patch \
  --tile-index 0
python -m baseline.compare_single_tile_levelset_vs_fft_equivalent \
  --tiles-dir /data/lyj/FuILT/tiles_from_patch \
  --tile-index 0 \
  --save-json outputs/compare_single_tile/levelset_vs_fft_eq_tile0.json
'''