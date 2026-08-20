import math
from typing import Dict, List, Tuple

import torch


Offsets5 = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]


def _assert_power_of_two(v: int) -> None:
	if v <= 0 or (v & (v - 1)) != 0:
		raise ValueError(f"值必须是 2 的幂，当前: {v}")


def derive_levels(s0: int, levels: int, ratio: float = 1.0 / 16.0) -> List[Dict[str, int]]:
	"""
	返回每层参数:
	- S: 子块边长
	- O: overlap
	- P: stride = S - O
	- S_next: 父块边长 = S + P
	"""
	out: List[Dict[str, int]] = []
	s = int(s0)
	for _ in range(levels):
		o = int(round(s * ratio))
		o = max(0, min(o, s - 1))
		p = s - o
		s_next = s + p
		out.append({"S": s, "O": o, "P": p, "S_next": s_next})
		s = s_next
	out.append({"S": s, "O": int(round(s * ratio)), "P": s - int(round(s * ratio)), "S_next": None})
	return out


def canvas_size_for_level0(grid0: int, s0: int, p0: int) -> int:
	return s0 + (grid0 - 1) * p0


def precompute_offsets_up(levels: List[Dict[str, int]], base_grid: int) -> List[Dict[str, object]]:
	all_levels: List[Dict[str, object]] = []
	grid_in = base_grid
	for l in range(len(levels) - 1):
		p_in = int(levels[l]["P"])
		s_out = int(levels[l]["S_next"])
		o_out = int(levels[l + 1]["O"])
		p_out = s_out - o_out

		grid_out = grid_in // 2
		offsets_level: List[Offsets5] = []
		for r in range(grid_out):
			for c in range(grid_out):
				ax0, ay0 = (2 * c) * p_in, (2 * r) * p_in
				bx0, by0 = (2 * c + 1) * p_in, (2 * r) * p_in
				cx0, cy0 = (2 * c) * p_in, (2 * r + 1) * p_in
				dx0, dy0 = (2 * c + 1) * p_in, (2 * r + 1) * p_in
				ox0, oy0 = c * p_out, r * p_out
				offsets_level.append(((ax0, ay0), (bx0, by0), (cx0, cy0), (dx0, dy0), (ox0, oy0)))
		all_levels.append({"grid_out": grid_out, "offsets": offsets_level})
		grid_in = grid_out
	return all_levels


def _add_patch(dst: torch.Tensor, src: torch.Tensor, x0: int, y0: int) -> None:
	h, w = src.shape
	dst[y0:y0 + h, x0:x0 + w] += src


def _add_count(cnt: torch.Tensor, h: int, w: int, x0: int, y0: int) -> None:
	cnt[y0:y0 + h, x0:x0 + w] += 1.0


def fuse4_from_bigbuf_python(
	in_big: torch.Tensor,
	out_big: torch.Tensor,
	offsets: Offsets5,
	s: int,
	o: int,
) -> torch.Tensor:
	"""
	纯 Python 版本，几何规则对齐 FusionSplit/fuse4_bigbuf_ext.py。
	offsets: [A, B, C, D, O]，每个元素为 (x0, y0)
	"""
	p = s - o
	s_out = s + p

	(ax0, ay0), (bx0, by0), (cx0, cy0), (dx0, dy0), (ox0, oy0) = offsets

	work = torch.zeros((s_out, s_out), dtype=torch.float32, device=in_big.device)
	cnt = torch.zeros((s_out, s_out), dtype=torch.float32, device=in_big.device)

	a = in_big[ay0:ay0 + s, ax0:ax0 + s].to(torch.float32)
	b = in_big[by0:by0 + s, bx0:bx0 + s].to(torch.float32)
	c = in_big[cy0:cy0 + s, cx0:cx0 + s].to(torch.float32)
	d = in_big[dy0:dy0 + s, dx0:dx0 + s].to(torch.float32)

	_add_patch(work, a, 0, 0)
	_add_count(cnt, s, s, 0, 0)

	_add_patch(work, b, p, 0)
	_add_count(cnt, s, s, p, 0)

	_add_patch(work, c, 0, p)
	_add_count(cnt, s, s, 0, p)

	_add_patch(work, d, p, p)
	_add_count(cnt, s, s, p, p)

	fused = torch.where(cnt > 0, work / cnt, torch.zeros_like(work))
	out_big[oy0:oy0 + s_out, ox0:ox0 + s_out] = fused.to(out_big.dtype)
	return out_big


def hierarchical_fuse_grads(
	grads: List[torch.Tensor],
	base_grid: int,
	overlap_ratio: float = 1.0 / 16.0,
) -> Dict[str, object]:
	"""
	对 grid x grid 个梯度做层级融合，返回顶层梯度与元信息。
	grads 顺序: row-major（先行后列）。
	"""
	_assert_power_of_two(base_grid)
	if len(grads) != base_grid * base_grid:
		raise ValueError(f"grads 数量与 base_grid 不匹配: {len(grads)} vs {base_grid}x{base_grid}")
	if not grads:
		raise ValueError("grads 不能为空")

	s0_h, s0_w = grads[0].shape[-2], grads[0].shape[-1]
	if s0_h != s0_w:
		raise ValueError(f"当前仅支持正方形 tile，得到: {s0_h}x{s0_w}")
	s0 = int(s0_h)

	for g in grads:
		if g.shape[-2:] != (s0, s0):
			raise ValueError("所有梯度尺寸必须一致")

	levels_fuse = int(math.log2(base_grid))
	levels = derive_levels(s0=s0, levels=levels_fuse, ratio=overlap_ratio)
	p0 = int(levels[0]["P"])
	canvas0 = canvas_size_for_level0(base_grid, s0, p0)
	s_top = int(levels[-1]["S"])

	up = precompute_offsets_up(levels=levels, base_grid=base_grid)

	device = grads[0].device

	# 纯张量版本：每层单独新建张量，不复用 ping-pong buffer
	current_level_tensor = torch.zeros((canvas0, canvas0), dtype=torch.float32, device=device)

	# level0 拼装
	idx = 0
	for r in range(base_grid):
		for c in range(base_grid):
			y0 = r * p0
			x0 = c * p0
			current_level_tensor[y0:y0 + s0, x0:x0 + s0] = grads[idx].to(torch.float32)
			idx += 1

	# up-pass
	for l in range(levels_fuse):
		s_in = int(levels[l]["S"])
		o_in = int(levels[l]["O"])
		s_out = int(levels[l]["S_next"])
		grid_out = int(up[l]["grid_out"])
		p_out = int(levels[l + 1]["P"])
		canvas_out = s_out + (grid_out - 1) * p_out

		next_level_tensor = torch.zeros((canvas_out, canvas_out), dtype=torch.float32, device=device)
		for offsets in up[l]["offsets"]:
			fuse4_from_bigbuf_python(current_level_tensor, next_level_tensor, offsets, s=s_in, o=o_in)
		current_level_tensor = next_level_tensor

	top = current_level_tensor[0:s_top, 0:s_top].clone()

	return {
		"top": top,
		"meta": {
			"base_grid": base_grid,
			"overlap_ratio": overlap_ratio,
			"levels": levels,
			"s0": s0,
		},
	}

