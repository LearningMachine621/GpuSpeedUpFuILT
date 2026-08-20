import glob
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import cv2
import numpy as np
import torch

from fuilt.pre_work.read_oas2mask import (
    read_oas_to_real_size_mask,
    parse_tile_info_from_filename
)

from fuilt.fusion_split.fuse_grads import hierarchical_fuse_grads
from fuilt.fusion_split.split_grads import hierarchical_split_grads

from fuilt.ilt.func.levelset import LevelSetILT, LevelSetCfg
from fuilt.ilt.func import simple as lithosim
from fuilt.ilt.func import initializer
from fuilt.ilt.func import evaluation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parent.parent
FU_OVERLAP_CONFIG_DIR = ROOT / "config"


def _cuda_sync_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_peak_vram_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def _extract_row_col(path: str) -> Tuple[int, int]:
    base = os.path.basename(path)
    m = re.search(r"tile_r(\d+)_c(\d+)", base)
    if not m:
        return 10**9, 10**9
    return int(m.group(1)), int(m.group(2))

def discover_tile_files(oas_dir: str) -> List[str]:
    files = glob.glob(os.path.join(oas_dir, "*.oas"))
    files = sorted(files, key=lambda x: _extract_row_col(x))
    return files

def infer_grid_size(num_tiles: int) -> int:
    grid = int(round(math.sqrt(num_tiles)))
    if grid * grid != num_tiles:
        raise ValueError(f"tile 数量必须是完全平方数，当前: {num_tiles}")
    if grid <= 0 or (grid & (grid - 1)) != 0:
        raise ValueError(f"grid 必须是 2 的幂，当前: {grid}")
    return grid

def _save_tile_masks(params_list: List[torch.Tensor], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for i, p in enumerate(params_list):
        mask = (p < 0).to(torch.uint8).detach().cpu().numpy() * 255
        cv2.imwrite(os.path.join(out_dir, f"tile_{i:03d}_mask.png"), mask)
        torch.save(p.detach().cpu(), os.path.join(out_dir, f"tile_{i:03d}_params.pt"))

def _build_large_mask_from_params(
    params_list: List[torch.Tensor],
    base_grid: int,
    overlap_ratio: float,
) -> torch.Tensor:
    tile_masks = [((p < 0).to(torch.float32)) for p in params_list]
    fused = hierarchical_fuse_grads(tile_masks, base_grid=base_grid, overlap_ratio=overlap_ratio)
    return cast(torch.Tensor, fused["top"])

def gradient_fusion_split_optimization_tiles(
    oas_dir: str,
    target_layers: Optional[List[Tuple[int, int]]] = None,
    iterations: int = 50,
    learning_rate: float = 0.025,
    eval_interval: int = 10,
    levelset_cfg_path: Optional[str] = None,
    litho_cfg_path: Optional[str] = None,
) -> Dict[str, Any]:
    
    # ================= [PROFILING 计时器初始化] =================
    t_io_raster = 0.0      # 1. OAS读取与CPU光栅化
    t_h2d_init = 0.0       # 2. 拷贝到GPU及初始化
    t_levelset = 0.0       # 3.1 迭代内：LevelSet计算
    t_fuse = 0.0           # 3.2 迭代内：Fusion
    t_split = 0.0          # 3.3 迭代内：Split
    t_apply = 0.0          # 3.4 迭代内：Apply update
    t_eval = 0.0           # (额外) 评估时间，避免污染纯计算耗时
    t_d2h_save = 0.0       # 4. 传回CPU与保存生成大图
    
    _reset_peak_vram_if_needed()
    # ==========================================================

    tile_files = discover_tile_files(oas_dir)
    if not tile_files:
        raise FileNotFoundError(f"目录中没有 oas 文件: {oas_dir}")

    first_info = parse_tile_info_from_filename(os.path.basename(tile_files[0]))
    tile_size = int(first_info.get("tile_size") or 1024)
    overlap = int(first_info.get("overlap") or 64)
    overlap_ratio = float(overlap) / float(tile_size)
    base_grid = infer_grid_size(len(tile_files))

    print(f"发现 tile: {len(tile_files)} 个，grid={base_grid}x{base_grid}, s={tile_size}, overlap={overlap}")

    # ================= 1. OAS读取与光栅化 (CPU端为主) =================
    target_masks: List[torch.Tensor] = []
    _t_start = time.time()
    for idx, oas_file in enumerate(tile_files):
        mask, info = read_oas_to_real_size_mask(
            oas_file=oas_file, target_layers=target_layers, target_size=tile_size, dtype=torch.float32,
        )
        target_masks.append(mask)
    t_io_raster += (time.time() - _t_start)

    # ================= [压测克隆魔法] =================
    # 假设你想测试 256 个 Tile (16x16 grid)
    MULTIPLIER = 16 
    target_masks = target_masks * MULTIPLIER 
    
    # 强制覆盖基线网格大小
    base_grid = infer_grid_size(len(target_masks))
    print(f"🔥 压测模式开启: 强制克隆为 {len(target_masks)} 个 Tiles, Grid={base_grid}x{base_grid}")
    # ==================================================

    # ================= 2. 数据传输至 GPU (H2D) & 初始化 =================
    _t_start = time.time()
    if levelset_cfg_path is None:
        levelset_cfg_path = str(FU_OVERLAP_CONFIG_DIR / "pylevelset1024.txt")
    if litho_cfg_path is None:
        litho_cfg_path = str(FU_OVERLAP_CONFIG_DIR / "lithosimple.txt")

    cfg = LevelSetCfg(levelset_cfg_path)
    litho = lithosim.LithoSim(litho_cfg_path)
    levelset_solver = LevelSetILT(cfg, litho, device=DEVICE)

    params_list: List[torch.Tensor] = []
    for i in range(len(target_masks)):
        target_masks[i] = target_masks[i].to(DEVICE) # Mask 上 GPU
        target_norm = target_masks[i] / 255.0 if target_masks[i].max() > 1.5 else target_masks[i]
        _, params = initializer.LevelSetImageInit().run(target_norm.detach().cpu().numpy())
        if not isinstance(params, torch.Tensor):
            params = torch.tensor(params, dtype=torch.float32)
        params_list.append(params.detach().to(DEVICE)) # Params 上 GPU
        
    _cuda_sync_if_needed()
    t_h2d_init += (time.time() - _t_start)

    # ================= 3. 主循环 (GPU 密集计算) =================
    last_avg_loss = 0.0
    for it in range(iterations):
        grads: List[torch.Tensor] = []
        loss_sum = 0.0

        # --- 3.1 LevelSetILT 计算梯度 ---
        _cuda_sync_if_needed()
        _t_ls_start = time.time()
        for tile_idx, (params, target) in enumerate(zip(params_list, target_masks)):
            target_norm = target / 255.0 if target.max() > 1.5 else target
            debug_tile0 = (tile_idx == 0)
            result = levelset_solver.compute_grad(
                target_norm,
                params,
                debug=debug_tile0,
                debug_tag=f"[iter={it+1:03d}][tile=0]" if debug_tile0 else "",
            )
            grads.append(result["gradient"].to(torch.float32))
            loss_sum += float(result["loss"])
        _cuda_sync_if_needed()
        t_levelset += (time.time() - _t_ls_start)

        # --- 3.2 Fusion ---
        _cuda_sync_if_needed()
        _t_fuse_start = time.time()
        fused = hierarchical_fuse_grads(grads=grads, base_grid=base_grid, overlap_ratio=overlap_ratio)
        fused_grad = cast(torch.Tensor, fused["top"])
        _cuda_sync_if_needed()
        t_fuse += (time.time() - _t_fuse_start)

        if fused_grad.numel() > 0:
            fg_abs = fused_grad.abs()
            print(
                f"[FusionDBG][iter={it+1:03d}] "
                f"top_shape={tuple(fused_grad.shape)}, "
                f"top_grad_mean={fg_abs.mean().item():.6e}, "
                f"top_grad_max={fg_abs.max().item():.6e}"
            )

        # --- 3.3 Split ---
        _cuda_sync_if_needed()
        _t_split_start = time.time()
        split_grads = hierarchical_split_grads(fused_top=fused_grad, base_grid=base_grid, overlap_ratio=overlap_ratio, s0=tile_size)
        _cuda_sync_if_needed()
        t_split += (time.time() - _t_split_start)

        if split_grads and split_grads[0].numel() > 0:
            sg0_abs = split_grads[0].abs()
            print(
                f"[SplitDBG][iter={it+1:03d}][tile=0] "
                f"grad_shape={tuple(split_grads[0].shape)}, "
                f"grad_mean={sg0_abs.mean().item():.6e}, "
                f"grad_max={sg0_abs.max().item():.6e}"
            )

        # --- 3.4 Apply update ---
        _cuda_sync_if_needed()
        _t_apply_start = time.time()
        for i in range(len(params_list)):
            debug_tile0 = (i == 0)
            params_list[i] = levelset_solver.apply_gradient(
                params_list[i],
                split_grads[i],
                lr=learning_rate,
                debug=debug_tile0,
                debug_tag=f"[iter={it+1:03d}][tile=0]" if debug_tile0 else "",
            )
        _cuda_sync_if_needed()
        t_apply += (time.time() - _t_apply_start)

        last_avg_loss = loss_sum / len(params_list)
        
        # --- (额外) 评估代码 (剔除出核心性能耗时) ---
        if (it + 1) % max(1, eval_interval) == 0 or (it + 1) == iterations:
            _cuda_sync_if_needed()
            _t_eval_start = time.time()
            l2s, pvbs, epes = [], [], []
            with torch.no_grad():
                for params, target in zip(params_list, target_masks):
                    mask = (params < 0).to(torch.float32)
                    target_norm = target / 255.0 if target.max() > 1.5 else target
                    l2, pvb, epe, _ = evaluation.evaluate(mask, target_norm, litho, scale=1, shots=False)
                    l2s.append(l2)
                    pvbs.append(pvb)
                    epes.append(epe)
            _cuda_sync_if_needed()
            t_eval += (time.time() - _t_eval_start)
            print(f"[Iter {it+1:03d}/{iterations}] avg_loss={last_avg_loss:.4f} | Eval L2={np.mean(l2s):.2f}, EPE={np.mean(epes):.2f}")
        else:
            print(f"[Iter {it+1:03d}/{iterations}] avg_loss={last_avg_loss:.4f}")

    # ================= 4. D2H 传回与结果生成 (CPU端) =================
    _cuda_sync_if_needed()
    _t_start = time.time()
    out_tile_dir = str(ROOT / "tile_masks")
    _save_tile_masks(params_list, out_tile_dir)
    large_mask = _build_large_mask_from_params(params_list=params_list, base_grid=base_grid, overlap_ratio=overlap_ratio)
    large_mask_img = (large_mask.detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
    large_out_dir = ROOT / "large_mask_results"
    large_out_dir.mkdir(exist_ok=True)
    large_mask_path = str(large_out_dir / f"large_mask_baseline_{large_mask_img.shape[1]}x{large_mask_img.shape[0]}.png")
    cv2.imwrite(large_mask_path, large_mask_img)
    t_d2h_save += (time.time() - _t_start)

    # ================= [获取最终峰值显存] =================
    peak_memory_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0

    return {
        "num_tiles": len(tile_files),
        "grid": base_grid,
        "tile_size": tile_size,
        "overlap": overlap,
        "final_loss": last_avg_loss,
        "tile_masks_dir": out_tile_dir,
        "large_mask_path": large_mask_path,
        "large_mask": large_mask,
        "profile_times": {
            "1. IO_Raster": t_io_raster,
            "2. H2D_Init": t_h2d_init,
            "3.1 LevelSet": t_levelset,
            "3.2 Fuse": t_fuse,
            "3.3 Split": t_split,
            "3.4 Apply": t_apply,
            "4. D2H_Save": t_d2h_save,
            "X. Eval(Excluded)": t_eval
        },
        "peak_vram_mb": peak_memory_mb
    }

def main() -> None:
    default_tiles = "/data/lyj/FuILT/tiles_from_patch"
    tiles_dir = os.environ.get("FUILT_TILES_DIR", default_tiles)
    
    total_t0 = time.time()
    result = gradient_fusion_split_optimization_tiles(
        oas_dir=tiles_dir,
        target_layers=[(23, 100)],
        iterations=int(os.environ.get("FUILT_ITERS", "20")),
        learning_rate=float(os.environ.get("FUILT_LR", "0.025")),
        eval_interval=int(os.environ.get("FUILT_EVAL_INTERVAL", "10")),
    )
    total_t1 = time.time()

    print("\n" + "="*50)
    print("🚀 Baseline Profiling Report (基线性能分析报告)")
    print("="*50)
    print(f"📌 Tile 规格: {result['num_tiles']} 个 ({result['grid']}x{result['grid']}), 大小 {result['tile_size']}x{result['tile_size']}")
    print(f"🔥 峰值显存 (Peak VRAM): {result['peak_vram_mb']:.2f} MB")
    
    times = cast(Dict[str, float], result['profile_times'])
    total_valid_time = (
        times["1. IO_Raster"]
        + times["2. H2D_Init"]
        + times["3.1 LevelSet"]
        + times["3.2 Fuse"]
        + times["3.3 Split"]
        + times["3.4 Apply"]
        + times["4. D2H_Save"]
    )
    
    print("-" * 50)
    print(f"{'阶段 (Pipeline Stage)':<25} | {'耗时 (s)':<10} | {'占比 (%)'}")
    print("-" * 50)
    for stage, t in times.items():
        if "Eval" in stage: continue # 评估时间不计入主流水线占比
        pct = (t / total_valid_time) * 100 if total_valid_time > 0 else 0.0
        print(f"{stage:<25} | {t:>8.3f} s | {pct:>5.1f} %")
    print("-" * 50)
    print(f"💡 流水线总纯耗时 (剔除Eval): {total_valid_time:.3f} s")
    print(f"⏱️ 脚本端到端总耗时 (含Eval): {total_t1 - total_t0:.3f} s")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()