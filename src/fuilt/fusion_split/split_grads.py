import math
from typing import Dict, List, Tuple

import torch


Offsets5 = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]


def _assert_power_of_two(v: int) -> None:
	if v <= 0 or (v & (v - 1)) != 0:
		raise ValueError(f"值必须是 2 的幂，当前: {v}")


def derive_levels(s0: int, levels: int, ratio: float = 1.0 / 16.0) -> List[Dict[str, int]]:
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


def precompute_offsets_down(levels: List[Dict[str, int]]) -> List[Dict[str, object]]:
	"""
	Down-pass 从顶层(1x1)逐层切回 level0(base_grid x base_grid)。
	offsets: [A, B, C, D, I(parent)]
	"""
	all_levels: List[Dict[str, object]] = []
	grid_parent = 1
	for l in range(len(levels) - 2, -1, -1):
		p_child = int(levels[l]["P"])
		s_parent = int(levels[l]["S_next"])
		o_parent = int(levels[l + 1]["O"])
		p_parent = s_parent - o_parent

		offsets_level: List[Offsets5] = []
		for r in range(grid_parent):
			for c in range(grid_parent):
				ax0, ay0 = (2 * c) * p_child, (2 * r) * p_child
				bx0, by0 = (2 * c + 1) * p_child, (2 * r) * p_child
				cx0, cy0 = (2 * c) * p_child, (2 * r + 1) * p_child
				dx0, dy0 = (2 * c + 1) * p_child, (2 * r + 1) * p_child
				ix0, iy0 = c * p_parent, r * p_parent
				offsets_level.append(((ax0, ay0), (bx0, by0), (cx0, cy0), (dx0, dy0), (ix0, iy0)))

		all_levels.append({"grid_parent": grid_parent, "offsets": offsets_level})
		grid_parent *= 2
	return all_levels


def split4_to_bigbuf_python(
	in_big: torch.Tensor,
	out_big: torch.Tensor,
	offsets: Offsets5,
	s: int,
	o: int,
) -> torch.Tensor:
	"""
	纯 Python 版本，几何规则对齐 FusionSplit/split4_bigbuf_ext.py。
	offsets: [A, B, C, D, I(parent)]，每个元素为 (x0, y0)
	"""
	p = s - o
	(ax0, ay0), (bx0, by0), (cx0, cy0), (dx0, dy0), (ix0, iy0) = offsets

	parent = in_big[iy0:iy0 + s + p, ix0:ix0 + s + p].to(out_big.dtype)

	out_big[ay0:ay0 + s, ax0:ax0 + s] = parent[0:s, 0:s]
	out_big[by0:by0 + s, bx0:bx0 + s] = parent[0:s, p:p + s]
	out_big[cy0:cy0 + s, cx0:cx0 + s] = parent[p:p + s, 0:s]
	out_big[dy0:dy0 + s, dx0:dx0 + s] = parent[p:p + s, p:p + s]
	return out_big


def hierarchical_split_grads(
	fused_top: torch.Tensor,
	base_grid: int,
	overlap_ratio: float = 1.0 / 16.0,
	s0: int = 1024,
) -> List[torch.Tensor]:
	"""
	将顶层融合梯度切回 level0 的 grid x grid tile 梯度。
	返回顺序: row-major。
	"""
	_assert_power_of_two(base_grid)

	levels_fuse = int(math.log2(base_grid))
	levels = derive_levels(s0=s0, levels=levels_fuse, ratio=overlap_ratio)
	p0 = int(levels[0]["P"])

	down = precompute_offsets_down(levels=levels)

	# 纯张量版本：每层动态新建张量，不复用 ping-pong buffer
	current_level_tensor = fused_top.to(torch.float32)

	for i, l in enumerate(range(levels_fuse - 1, -1, -1)):
		s_child = int(levels[l]["S"])
		o_child = int(levels[l]["O"])
		p_child = int(levels[l]["P"])

		grid_child = int(down[i]["grid_parent"]) * 2
		canvas_size = s_child + (grid_child - 1) * p_child
		next_level_tensor = torch.zeros((canvas_size, canvas_size), dtype=torch.float32, device=fused_top.device)

		for offsets in down[i]["offsets"]:
			split4_to_bigbuf_python(current_level_tensor, next_level_tensor, offsets, s=s_child, o=o_child)

		current_level_tensor = next_level_tensor

	level0 = current_level_tensor
	out: List[torch.Tensor] = []
	for r in range(base_grid):
		for c in range(base_grid):
			y0 = r * p0
			x0 = c * p0
			out.append(level0[y0:y0 + s0, x0:x0 + s0].clone())
	return out

