import glob
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
# 项目根目录与配置目录（若运行时未设置，请使用仓库根的 config/）
ROOT = Path(__file__).resolve().parent.parent
FU_OVERLAP_CONFIG_DIR = ROOT / "config"

# Note: `origin` 信息来自 OAS 文件或文件名解析；若缺失，默认会返回 (0,0)。
# 如果你需要真实的 tile 原点，可以通过文件名或外部元数据传入。


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
	return fused["top"]


def gradient_fusion_split_optimization_tiles(
	oas_dir: str,
	target_layers: Optional[List[Tuple[int, int]]] = None,
	iterations: int = 50,
	learning_rate: float = 0.025,
	eval_interval: int = 10,
	levelset_cfg_path: Optional[str] = None,
	litho_cfg_path: Optional[str] = None,
) -> Dict[str, object]:
	tile_files = discover_tile_files(oas_dir)
	if not tile_files:
		raise FileNotFoundError(f"目录中没有 oas 文件: {oas_dir}")

	# 由首个 tile 推断参数
	first_info = parse_tile_info_from_filename(os.path.basename(tile_files[0]))
	tile_size = int(first_info.get("tile_size") or 1024)
	overlap = int(first_info.get("overlap") or 64)
	overlap_ratio = float(overlap) / float(tile_size)
	base_grid = infer_grid_size(len(tile_files))

	print(f"发现 tile: {len(tile_files)} 个，grid={base_grid}x{base_grid}, s={tile_size}, overlap={overlap}")

	# 1) OAS -> mask
	target_masks: List[torch.Tensor] = []
	for idx, oas_file in enumerate(tile_files):
		mask, info = read_oas_to_real_size_mask(
			oas_file=oas_file,
			target_layers=target_layers,
			target_size=tile_size,
			dtype=torch.float32,
		)
		mask = mask.to(DEVICE)
		target_masks.append(mask)
		if idx < 3:
			print(
				f"  tile[{idx}] polygons={info.get('polygons', 0)}, "
				f"bbox={info.get('oas_bbox', None)}, origin={info.get('origin_used', None)}"
			)

	# 2) LevelSetILT 初始化
	if levelset_cfg_path is None:
		levelset_cfg_path = str(FU_OVERLAP_CONFIG_DIR / "pylevelset1024.txt")
	if litho_cfg_path is None:
		litho_cfg_path = str(FU_OVERLAP_CONFIG_DIR / "lithosimple.txt")

	cfg = LevelSetCfg(levelset_cfg_path)
	litho = lithosim.LithoSim(litho_cfg_path)
	levelset_solver = LevelSetILT(cfg, litho, device=DEVICE)

	params_list: List[torch.Tensor] = []
	for target in target_masks:
		target_norm = target / 255.0 if target.max() > 1.5 else target
		_, params = initializer.LevelSetImageInit().run(target_norm.detach().cpu().numpy())
		if not isinstance(params, torch.Tensor):
			params = torch.tensor(params, dtype=torch.float32)
		params = params.detach().to(DEVICE)
		params_list.append(params)

	# 3) 主循环: grad -> fuse -> split -> update
	last_avg_loss = 0.0
	for it in range(iterations):
		t0 = time.time()
		grads: List[torch.Tensor] = []
		loss_sum = 0.0

		for params, target in zip(params_list, target_masks):
			target_norm = target / 255.0 if target.max() > 1.5 else target
			result = levelset_solver.compute_grad(target_norm, params)
			grads.append(result["gradient"].to(torch.float32))
			loss_sum += float(result["loss"])

		fused = hierarchical_fuse_grads(
			grads=grads,
			base_grid=base_grid,
			overlap_ratio=overlap_ratio,
		)
		fused_grad = fused["top"]

		split_grads = hierarchical_split_grads(
			fused_top=fused_grad,
			base_grid=base_grid,
			overlap_ratio=overlap_ratio,
			s0=tile_size,
		)

		if len(split_grads) != len(params_list):
			raise RuntimeError(f"split 后梯度数量错误: {len(split_grads)} vs {len(params_list)}")

		for i in range(len(params_list)):
			params_list[i] = levelset_solver.apply_gradient(params_list[i], split_grads[i], lr=learning_rate)

		last_avg_loss = loss_sum / len(params_list)
		t1 = time.time()
		print(f"[Iter {it+1:03d}/{iterations}] avg_loss={last_avg_loss:.4f}, time={t1 - t0:.2f}s")

		if (it + 1) % max(1, eval_interval) == 0 or (it + 1) == iterations:
			l2s, pvbs, epes = [], [], []
			with torch.no_grad():
				for params, target in zip(params_list, target_masks):
					mask = (params < 0).to(torch.float32)
					target_norm = target / 255.0 if target.max() > 1.5 else target
					l2, pvb, epe, _ = evaluation.evaluate(mask, target_norm, litho, scale=1, shots=False)
					l2s.append(l2)
					pvbs.append(pvb)
					epes.append(epe)
			print(
				f"  Eval -> L2={np.mean(l2s):.2f}, PVBand={np.mean(pvbs):.2f}, EPE={np.mean(epes):.2f}"
			)

	# 4) 保存结果
	out_tile_dir = str(ROOT / "tile_masks")
	_save_tile_masks(params_list, out_tile_dir)

	large_mask = _build_large_mask_from_params(
		params_list=params_list,
		base_grid=base_grid,
		overlap_ratio=overlap_ratio,
	)
	large_mask_img = (large_mask.detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
	large_out_dir = ROOT / "large_mask_results"
	large_out_dir.mkdir(exist_ok=True)
	large_mask_path = str(large_out_dir / f"large_mask_baseline_{large_mask_img.shape[1]}x{large_mask_img.shape[0]}.png")
	cv2.imwrite(large_mask_path, large_mask_img)

	return {
		"num_tiles": len(tile_files),
		"grid": base_grid,
		"tile_size": tile_size,
		"overlap": overlap,
		"final_loss": last_avg_loss,
		"tile_masks_dir": out_tile_dir,
		"large_mask_path": large_mask_path,
		"large_mask": large_mask,
	}


def main() -> None:
	# 按你的当前目录结构默认读取 tiles_from_patch
	default_tiles = "/data/lyj/FuILT/tiles_from_patch"
	tiles_dir = os.environ.get("FUILT_TILES_DIR", default_tiles)

	result = gradient_fusion_split_optimization_tiles(
		oas_dir=tiles_dir,
		target_layers=[(23, 100)],
		iterations=int(os.environ.get("FUILT_ITERS", "20")),
		learning_rate=float(os.environ.get("FUILT_LR", "0.025")),
		eval_interval=int(os.environ.get("FUILT_EVAL_INTERVAL", "10")),
	)

	print("\n===== Baseline 完成 =====")
	print(f"tile 数: {result['num_tiles']} ({result['grid']}x{result['grid']})")
	print(f"tile 尺寸: {result['tile_size']}, overlap: {result['overlap']}")
	print(f"最终 loss: {result['final_loss']:.4f}")
	print(f"tile 输出目录: {result['tile_masks_dir']}")
	print(f"大图输出: {result['large_mask_path']}")


if __name__ == "__main__":
	main()

