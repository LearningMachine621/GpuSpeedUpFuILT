import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

try:
	import klayout.db as kdb
except Exception:
	import pya as kdb


def _safe_int(v: str, default: Optional[int] = None) -> Optional[int]:
	try:
		return int(v)
	except Exception:
		return default


def parse_tile_info_from_filename(filename: str) -> Dict[str, Optional[int]]:
	"""
	解析 tile 文件名中的坐标与尺寸信息。

	支持格式（示例）:
	- tile_r0_c0_s1024_o64_rel0_0_abs100000_100000_1024x1024.oas
	- tile_r0_c0_s1024_o64_orig0_0_1024x1024.oas
	- ...global(100000,100000)...rel1024x1024...
	- ...global_100000_100000...rel1024x1024...
	"""
	base = os.path.basename(filename)

	info: Dict[str, Optional[int]] = {
		"row": None,
		"col": None,
		"tile_size": None,
		"overlap": None,
		"rel_x": None,
		"rel_y": None,
		"abs_x": None,
		"abs_y": None,
		"real_w": None,
		"real_h": None,
	}

	m_rc = re.search(r"tile_r(\d+)_c(\d+)", base)
	if m_rc:
		info["row"] = _safe_int(m_rc.group(1))
		info["col"] = _safe_int(m_rc.group(2))

	m_so = re.search(r"_s(\d+)_o(\d+)_", base)
	if m_so:
		info["tile_size"] = _safe_int(m_so.group(1))
		info["overlap"] = _safe_int(m_so.group(2))

	m_rel_abs = re.search(r"_rel(-?\d+)_(-?\d+)_abs(-?\d+)_(-?\d+)_", base)
	if m_rel_abs:
		info["rel_x"] = _safe_int(m_rel_abs.group(1))
		info["rel_y"] = _safe_int(m_rel_abs.group(2))
		info["abs_x"] = _safe_int(m_rel_abs.group(3))
		info["abs_y"] = _safe_int(m_rel_abs.group(4))
	else:
		m_orig = re.search(r"_orig(-?\d+)_(-?\d+)_", base)
		if m_orig:
			info["rel_x"] = _safe_int(m_orig.group(1))
			info["rel_y"] = _safe_int(m_orig.group(2))

		m_global = re.search(r"global\((-?\d+),\s*(-?\d+)\)", base)
		if not m_global:
			m_global = re.search(r"global_(-?\d+)_(-?\d+)", base)
		if m_global:
			info["abs_x"] = _safe_int(m_global.group(1))
			info["abs_y"] = _safe_int(m_global.group(2))

	m_wh = re.search(r"_(\d+)x(\d+)\.oas$", base)
	if m_wh:
		info["real_w"] = _safe_int(m_wh.group(1))
		info["real_h"] = _safe_int(m_wh.group(2))
	else:
		m_rel_wh = re.search(r"rel(\d+)x(\d+)", base)
		if m_rel_wh:
			info["real_w"] = _safe_int(m_rel_wh.group(1))
			info["real_h"] = _safe_int(m_rel_wh.group(2))

	return info


def _find_layer_indices(layout: "kdb.Layout", top_cell: "kdb.Cell", target_layers: Optional[Sequence[Tuple[int, int]]]) -> List[int]:
	indices: List[int] = []
	if target_layers is None:
		for idx in range(layout.layers()):
			if len(top_cell.shapes(idx)) > 0:
				indices.append(idx)
		return indices

	for target_layer, target_dtype in target_layers:
		for idx in range(layout.layers()):
			li = layout.get_info(idx)
			if li.layer == target_layer and li.datatype == target_dtype and len(top_cell.shapes(idx)) > 0:
				indices.append(idx)
	return indices


def _infer_transform(
	bbox: "kdb.Box",
	canvas_w: int,
	canvas_h: int,
	abs_x: Optional[int],
	abs_y: Optional[int],
) -> Tuple[int, int]:
	# 情况1: OAS 已是局部坐标
	if bbox.left >= 0 and bbox.bottom >= 0 and bbox.right <= canvas_w and bbox.top <= canvas_h:
		return 0, 0

	# 情况2: OAS 使用全局坐标（文件名可提供绝对起点）
	if abs_x is not None and abs_y is not None:
		return abs_x, abs_y

	# 情况3: 退化为按 bbox 左下角对齐
	return bbox.left, bbox.bottom


def read_oas_to_real_size_mask(
	oas_file: str,
	target_layers: Optional[Sequence[Tuple[int, int]]] = None,
	target_size: int = 1024,
	dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, Dict[str, object]]:
	"""
	读取 OAS tile 并光栅化为 mask。

	返回:
	- mask_tensor: [target_size, target_size]，值域 0/1
	- result_info: 解析与坐标信息
	"""
	filename = os.path.basename(oas_file)
	tile_info = parse_tile_info_from_filename(filename)

	layout = kdb.Layout()
	layout.read(oas_file)
	top_cell = layout.top_cell()
	bbox = top_cell.bbox()

	real_w = tile_info.get("real_w") or bbox.width()
	real_h = tile_info.get("real_h") or bbox.height()
	real_w = int(real_w)
	real_h = int(real_h)

	mask = np.zeros((real_h, real_w), dtype=np.uint8)
	valid_layers = _find_layer_indices(layout, top_cell, target_layers)

	if not valid_layers:
		mask_tensor = torch.zeros((target_size, target_size), dtype=dtype)
		info = {
			"tile_info": tile_info,
			"oas_bbox": (bbox.left, bbox.bottom, bbox.right, bbox.top),
			"mask_size": (real_w, real_h),
			"polygons": 0,
		}
		return mask_tensor, info

	tx = _infer_transform(
		bbox=bbox,
		canvas_w=real_w,
		canvas_h=real_h,
		abs_x=tile_info.get("abs_x"),
		abs_y=tile_info.get("abs_y"),
	)
	origin_x, origin_y = tx

	poly_count = 0
	for layer_idx in valid_layers:
		shapes = top_cell.shapes(layer_idx)
		for shape in shapes.each():
			if shape.is_box():
				box = shape.box
				points = [
					(box.left, box.bottom),
					(box.right, box.bottom),
					(box.right, box.top),
					(box.left, box.top),
				]
			elif shape.is_polygon():
				points = [(p.x, p.y) for p in shape.polygon.each_point_hull()]
			elif shape.is_path():
				poly = shape.path.polygon()
				points = [(p.x, p.y) for p in poly.each_point_hull()]
			else:
				continue

			if len(points) < 3:
				continue

			pix = []
			for x, y in points:
				px = int(x - origin_x)
				py = int(y - origin_y)
				px = max(0, min(real_w - 1, px))
				py = max(0, min(real_h - 1, py))
				pix.append((px, py))

			if len(pix) >= 3:
				cv2.fillPoly(mask, [np.array(pix, dtype=np.int32)], 1)
				poly_count += 1

	# 统一到 target_size，保持与 1024 tile pipeline 对齐
	if mask.shape[0] > target_size or mask.shape[1] > target_size:
		mask = mask[:target_size, :target_size]
	if mask.shape[0] < target_size or mask.shape[1] < target_size:
		pad_h = target_size - mask.shape[0]
		pad_w = target_size - mask.shape[1]
		mask = np.pad(mask, ((0, max(0, pad_h)), (0, max(0, pad_w))), mode="constant")

	mask_tensor = torch.from_numpy(mask).to(dtype=dtype).contiguous()
	info = {
		"tile_info": tile_info,
		"oas_bbox": (bbox.left, bbox.bottom, bbox.right, bbox.top),
		"mask_size": (real_w, real_h),
		"origin_used": (origin_x, origin_y),
		"polygons": poly_count,
	}
	return mask_tensor, info


def read_oas_to_mask(
	oas_file: str,
	target_layers: Optional[Sequence[Tuple[int, int]]] = None,
	target_size: int = 1024,
	dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
	mask, _ = read_oas_to_real_size_mask(
		oas_file=oas_file,
		target_layers=target_layers,
		target_size=target_size,
		dtype=dtype,
	)
	return mask

