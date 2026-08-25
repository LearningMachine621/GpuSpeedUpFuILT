import argparse
import concurrent.futures
import glob
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import cv2
import numpy as np
import torch
from torch.utils.cpp_extension import load

from fuilt.fusion_split.fuse_grads import hierarchical_fuse_grads
from fuilt.fusion_split.write_grads_inplace_v1 import StaticFusionSplitManager
from fuilt.ilt.func import evaluation
from fuilt.ilt.func import simple as lithosim
from fuilt.ilt.func.triton_fused_pointwise import fused_pointwise_forward_triton
from fuilt.pre_work.read_oas2mask import parse_tile_info_from_filename, read_oas_to_real_size_mask

# Switch the pointwise kernel backend via env var.
#   FUILT_USE_TRITON_POINTWISE=1  → Triton port (default for v8+)
#   FUILT_USE_TRITON_POINTWISE=0  → original CUDA kernel (baseline)
_USE_TRITON_POINTWISE = bool(int(os.environ.get("FUILT_USE_TRITON_POINTWISE", "0")))

# Switch the FFT path via env var.
#   FUILT_USE_COMBINED_KERNEL=1  → pre-combined kernel_fft (24 ifft2 → 1)
#   FUILT_USE_COMBINED_KERNEL=0  → original 24-ifft2 loop (baseline)
#
# Math: by IFFT linearity,
#   sum_k(scale[k] * ifft2(mask_fft * kernel_fft[k]).real)
#   == ifft2(mask_fft * sum_k(scale[k] * kernel_fft[k])).real
# The combined kernel is invariant across iters, so we compute it once at init.
_USE_COMBINED_KERNEL = bool(int(os.environ.get("FUILT_USE_COMBINED_KERNEL", "0")))

# Switch the D2H path via env var.
#   FUILT_USE_ASYNC_D2H=1  → pinned memory + non_blocking + overlap with fuse
#   FUILT_USE_ASYNC_D2H=0  → original synchronous .detach().cpu() loop (baseline)
#
# Why: 256 sequential .cpu() calls incur per-call sync overhead. pinned +
# non_blocking lets the D2H run concurrently with the GPU `workspace.fuse()`
# call, hiding ~50% of D2H time behind compute.
_USE_ASYNC_D2H = bool(int(os.environ.get("FUILT_USE_ASYNC_D2H", "0")))

# Switch the H2D path via env var.
#   FUILT_USE_BATCH_H2D=1  → stack 256 tiles into [256,H,W] + single H2D
#   FUILT_USE_BATCH_H2D=0  → original per-tile H2D loop (baseline)
#
# Why: 256 small H2D calls each pay launch overhead. Stacking lets PCIe
# saturate with one large transfer.
#
# ⚠ This optimization REGRESSED (np.stack + unbind+contiguous add 2GB copy).
# Kept as cautionary tale. See feedback_p04_batch_h2d_failed.md.
_USE_BATCH_H2D = bool(int(os.environ.get("FUILT_USE_BATCH_H2D", "0")))

# Switch the per-tile H2D `.contiguous()` call.
#   FUILT_H2D_NO_CONTIG=1  → skip redundant .contiguous() (P0.5)
#   FUILT_H2D_NO_CONTIG=0  → original path with explicit .contiguous()
#
# Why: `torch.from_numpy(...).pin_memory().to(DEVICE)` already returns a
# contiguous tensor — the explicit `.contiguous()` is a no-op call but
# still pays Python/C++ dispatch overhead per tile.
_USE_H2D_NO_CONTIG = bool(int(os.environ.get("FUILT_H2D_NO_CONTIG", "0")))

# Enable NVTX range annotations for nsys timeline visualization (P6).
#   FUILT_USE_NVTX=1  → wrap each pipeline stage in nvtx.range_push/pop
#   FUILT_USE_NVTX=0  → no annotations (default, zero overhead)
#
# When enabled, run with: `nsys profile -t cuda,nvtx ...` and open the .nsys-rep
# in nsys-ui to see colored per-stage bands on the timeline.
_USE_NVTX = bool(int(os.environ.get("FUILT_USE_NVTX", "0")))

# Capture one full v7 iteration into a CUDA Graph and replay it (POC integration).
#   FUILT_USE_GRAPH=1  -> warmup + torch.cuda.graph capture + replay loop
#   FUILT_USE_GRAPH=0  -> original eager per-iter loop (default)
#
# Why: replaying a captured graph removes Python dispatch + per-kernel launch
# overhead. The captured region must be pure-GPU (no .item()/print/sync), so
# the graph path uses a GPU loss accumulator (_run_one_iter). This is the v7
# graph-capture POC (test_baseline_v7_graph_capture_poc.py) folded into the
# main pipeline.
#
# Prerequisite: every kernel in the captured region must launch on the capture
# stream. Hand-written CUDA kernels do this only after the getCurrentCUDAStream
# fix (see fused_pointwise_kernel.cu); Triton kernels are capture-safe by
# default. With the buggy default-stream launch the replay silently skips the
# kernel -> wrong output.
_USE_GRAPH = bool(int(os.environ.get("FUILT_USE_GRAPH", "0")))


def _nvtx_push(name: str) -> None:
	if _USE_NVTX:
		torch.cuda.nvtx.range_push(name)


def _nvtx_pop() -> None:
	if _USE_NVTX:
		torch.cuda.nvtx.range_pop()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
CUDA_SRC = ROOT / "src" / "fuilt" / "ilt" / "func" / "levelset_simple.cu"
POINTWISE_CUDA_SRC = ROOT / "src" / "fuilt" / "ilt" / "func" / "fused_pointwise_kernel.cu"

_POINTWISE_EXT: Optional[Any] = None


def load_pointwise_ext() -> Any:
	global _POINTWISE_EXT
	if _POINTWISE_EXT is not None:
		return _POINTWISE_EXT
	if not POINTWISE_CUDA_SRC.exists():
		raise FileNotFoundError(f"Pointwise CUDA 源文件不存在: {POINTWISE_CUDA_SRC}")
	build_dir = ROOT / ".torch_extensions" / "fused_pointwise_ext"
	build_dir.mkdir(parents=True, exist_ok=True)
	_POINTWISE_EXT = load(
		name="fused_pointwise_ext",
		sources=[str(POINTWISE_CUDA_SRC)],
		extra_cuda_cflags=["-O3", "--use_fast_math"],
		extra_cflags=["-O3"],
		build_directory=str(build_dir),
		verbose=False,
	)
	return _POINTWISE_EXT


def set_pointwise_capture_safe(enabled: bool) -> None:
	"""Toggle the fused_pointwise CUDA kernel's capture-safe stream launch.

	True (default)  -> launch on getCurrentCUDAStream(): recorded into any
	                   enclosing torch.cuda.graph() (the FIX).
	False           -> launch on getDefaultCUDAStream() (stream 0): silently
	                   skipped on graph replay, leaving output stale (the BUG).
	Used by the 5-mode benchmark to reproduce the pre-fix failure vs the fix
	in one binary. No effect outside graph capture (eager is unchanged).
	"""
	ext = load_pointwise_ext()
	if not hasattr(ext, "set_capture_safe_stream"):
		raise RuntimeError(
			"loaded fused_pointwise_ext predates the capture-safe toggle; "
			"clear .torch_extensions/fused_pointwise_ext and rebuild"
		)
	ext.set_capture_safe_stream(1 if enabled else 0)


def _cuda_sync_if_needed() -> None:
	if torch.cuda.is_available():
		torch.cuda.synchronize()


def _reset_peak_vram_if_needed() -> None:
	if torch.cuda.is_available():
		torch.cuda.reset_peak_memory_stats()


def check_vram(tag: str) -> None:
	"""显存探针：打印当前实时占用与历史峰值。"""
	if not torch.cuda.is_available():
		return
	current_mb = torch.cuda.memory_allocated() / (1024 * 1024)
	peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
	print(f"[{tag:^20}] 实时占用: {current_mb:8.2f} MB | 历史峰值: {peak_mb:8.2f} MB")


def _memory_budget_ok(
	min_headroom_mb: int = 256,
	warn_headroom_mb: int = 1024,
) -> Tuple[bool, str]:
	"""Memory-budget check after setup+warmup: fail fast if the workload sits at
	the VRAM cliff instead of OOMing after the (potentially minutes-long)
	iteration loop.

	'Usable headroom' = current driver-free + what the CUDA caching allocator can
	reclaim via `torch.cuda.empty_cache()` (reserved-but-not-allocated blocks).
	Callers must pair this with an `empty_cache()` before the peak allocation
	(the D2H_Save large-mask) so the reclaimable slack actually becomes free.

	Returns (ok, message). ok=False only when headroom is truly exhausted; a tight
	but workable config returns ok=True with a WARN.
	"""
	if not torch.cuda.is_available():
		return True, "[memory budget] no CUDA, skipped"
	free_bytes, _ = torch.cuda.mem_get_info()
	free_mb = free_bytes / (1024 * 1024)
	reclaimable_mb = (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / (1024 * 1024)
	available_mb = free_mb + reclaimable_mb
	if available_mb < min_headroom_mb:
		return False, (
			f"[memory budget] FAIL: only ~{available_mb:.0f} MB usable "
			f"(driver-free {free_mb:.0f} + reclaimable {reclaimable_mb:.0f}), need "
			f"≥ {min_headroom_mb} MB. This config sits at the VRAM ceiling of this "
			f"GPU — reduce tiles (FUILT_TILE_MULTIPLIER) or shard across GPUs."
		)
	if available_mb < warn_headroom_mb:
		return True, (
			f"[memory budget] WARN: ~{available_mb:.0f} MB usable headroom "
			f"(driver-free {free_mb:.0f} + reclaimable {reclaimable_mb:.0f}) — tight; "
			f"an empty_cache() before the peak D2H_Save step keeps it safe."
		)
	return True, f"[memory budget] OK: ~{available_mb:.0f} MB usable headroom"


def _extract_row_col(path: str) -> Tuple[int, int]:
	base = os.path.basename(path)
	m = re.search(r"tile_r(\d+)_c(\d+)", base)
	if not m:
		return 10**9, 10**9
	return int(m.group(1)), int(m.group(2))


def discover_tile_files(oas_dir: str) -> List[str]:
	files = glob.glob(os.path.join(oas_dir, "*.oas"))
	return sorted(files, key=_extract_row_col)


def infer_grid_size(num_tiles: int) -> int:
	grid = int(round(math.sqrt(num_tiles)))
	if grid * grid != num_tiles:
		raise ValueError(f"tile 数量必须是完全平方数，当前: {num_tiles}")
	if grid <= 0 or (grid & (grid - 1)) != 0:
		raise ValueError(f"grid 必须是 2 的幂，当前: {grid}")
	return grid


def _build_large_mask_from_params(
	params_list: List[torch.Tensor],
	base_grid: int,
	overlap_ratio: float,
) -> torch.Tensor:
	tile_masks = [((p < 0).to(torch.float32)) for p in params_list]
	fused = hierarchical_fuse_grads(tile_masks, base_grid=base_grid, overlap_ratio=overlap_ratio)
	return cast(torch.Tensor, fused["top"])


def _load_cuda_extension() -> Any:
	if not CUDA_SRC.exists():
		raise FileNotFoundError(f"CUDA 源文件不存在: {CUDA_SRC}")
	build_dir = ROOT / ".torch_extensions" / "levelset_simple_ext"
	build_dir.mkdir(parents=True, exist_ok=True)
	return load(
		name="levelset_simple_ext",
		sources=[str(CUDA_SRC)],
		extra_cuda_cflags=[
			"-O3",
			"--use_fast_math",
			"-gencode",
			"arch=compute_89,code=sm_89",
			"-Xptxas",
			"-v",
			"-Xptxas",
			"-warn-spills",
		],
		extra_cflags=["-O3"],
		build_directory=str(build_dir),
		verbose=True,
	)


def _build_kernel_fft(opt_kernels: torch.Tensor, height: int, width: int) -> torch.Tensor:
	k_count = opt_kernels.shape[0]
	kernel_fft_list: List[torch.Tensor] = []
	for k in range(k_count):
		kernel = opt_kernels[k]
		kernel_shifted = torch.fft.ifftshift(kernel)
		padded = torch.zeros((height, width), dtype=torch.float32, device=opt_kernels.device)
		padded[: kernel.shape[0], : kernel.shape[1]] = kernel_shifted
		kernel_fft_list.append(torch.fft.fft2(padded.to(torch.complex64), norm="forward"))
	return torch.stack(kernel_fft_list, dim=0).contiguous()


def _cpu_init_single_tile(target_cpu: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
	mask_np = target_cpu.detach().cpu().numpy()
	if mask_np.max() > 1.5:
		mask_np = mask_np / 255.0
	mask_uint8 = (mask_np > 0.5).astype(np.uint8)
	inner_dist = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
	outer_dist = cv2.distanceTransform(1 - mask_uint8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
	signed_dist = outer_dist - inner_dist
	return mask_uint8.astype(np.float32), signed_dist.astype(np.float32)


def _async_save_single_tile(p_cpu: torch.Tensor, i: int, out_dir: str) -> None:
	mask_img = (p_cpu < 0).to(torch.uint8).numpy() * 255
	cv2.imwrite(os.path.join(out_dir, f"tile_{i:03d}_mask.png"), mask_img)
	torch.save(p_cpu, os.path.join(out_dir, f"tile_{i:03d}_params.pt"))


def _async_save_large_mask(large_mask_cpu: torch.Tensor, large_mask_path: str) -> None:
	# SDF convention: negative = inside mask — same threshold as _async_save_single_tile
	# and _build_large_mask_from_params. (> 0.5) was a display bug: it drew the
	# SDF-positive (outside) region, i.e. the exact complement of the mask.
	large_mask_img = (large_mask_cpu.numpy() < 0).astype(np.uint8) * 255
	cv2.imwrite(large_mask_path, large_mask_img)


@torch.no_grad()
def _compute_pointwise(
	aerial: torch.Tensor,
	target: torch.Tensor,
	pointwise: str,
	target_density: float,
	print_steepness: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
	"""Dispatch the per-element sigmoid-print grad + diff^2 compute.

	pointwise:
	  "auto"    -> env-driven (FUILT_USE_TRITON_POINTWISE); backward compatible
	  "unfused" -> pure-PyTorch discrete ops (v5/v6 baseline; 10 launches measured, no fused kernel)
	  "cuda"    -> hand-written fused CUDA kernel (fused_pointwise_kernel.cu)
	  "triton"  -> Triton port (triton_fused_pointwise.py)

	Returns (grad, diff_sq), same shape/dtype as `aerial`, for all backends.
	"""
	if pointwise == "auto":
		pointwise = "triton" if _USE_TRITON_POINTWISE else "cuda"
	if pointwise == "unfused":
		# Pure-PyTorch reference: identical math to the fused kernels, expressed
		# as discrete ops. This is the eager baseline (v5/v6) the fused kernel
		# replaced — kept here so the benchmark can show unfused-vs-fused under
		# identical FFT/split/apply scaffolding.
		z = print_steepness * (aerial - target_density)
		printed = torch.sigmoid(z)
		diff = printed - target
		grad = 2.0 * diff * print_steepness * printed * (1.0 - printed)
		diff_sq = diff * diff
		return grad, diff_sq
	if pointwise == "cuda":
		ext = load_pointwise_ext()
		return ext.fused_pointwise_forward(aerial, target, target_density, print_steepness)
	if pointwise == "triton":
		return fused_pointwise_forward_triton(aerial, target, target_density, print_steepness)
	raise ValueError(f"unknown pointwise backend: {pointwise!r} (auto|unfused|cuda|triton)")


@torch.no_grad()
def _pytorch_fft_grad_loss(
	params_in: torch.Tensor,
	target_mask: torch.Tensor,
	kernel_fft: torch.Tensor,
	kernel_scales: torch.Tensor,
	target_density: float,
	print_steepness: float,
	combined_kernel_fft: Optional[torch.Tensor] = None,
	return_gpu_loss: bool = False,
	pointwise: str = "auto",
) -> Tuple[torch.Tensor, "torch.Tensor | float"]:
	# 1. 频域变换
	mask_fft = torch.fft.fft2(params_in.to(torch.complex64), norm="forward")

	# 2. 频域卷积 + ifft2 累加到 aerial
	#    两条路径（数学等价，由 IFFT 线性性保证）：
	#      a) 原路径：24 次 ifft2 + 24 次 complex mul + 24 次 add_
	#      b) 优化路径：预 combine kernels，1 次 ifft2 + 1 次 complex mul
	if combined_kernel_fft is not None:
		# 优化路径（P0.1）
		aerial = torch.fft.ifft2(
			mask_fft * combined_kernel_fft, norm="forward"
		).real.to(torch.float32).contiguous()
	else:
		# 原路径
		aerial = torch.zeros_like(params_in, dtype=torch.float32)
		for k in range(kernel_fft.shape[0]):
			conv_freq_k = mask_fft * kernel_fft[k]
			conv_spatial_k = torch.fft.ifft2(conv_freq_k, norm="forward").real
			aerial.add_(conv_spatial_k, alpha=float(kernel_scales[k].item()))

	# 3. 后半场：逐像素计算，交给 Pointwise 算子（unfused / CUDA / Triton）
	grad, diff_sq = _compute_pointwise(
		aerial, target_mask, pointwise, target_density, print_steepness
	)

	# 4. 求和返回 loss
	loss = diff_sq.sum()
	if return_gpu_loss:
		# Graph-capture-friendly path: return GPU tensor, avoid CPU sync.
		# Caller can .item() outside capture if needed.
		return grad, loss
	return grad, float(loss.item())


def setup_v7_context(
	oas_dir: str,
	multiplier: int = 16,
	pointwise: str = "cuda",
	target_layers: Optional[List[Tuple[int, int]]] = None,
	learning_rate: float = 0.025,
	litho_cfg_path: Optional[str] = None,
) -> Dict[str, Any]:
	"""Build a reusable v7 iteration context (shared by the graph path + benchmark).

	Returns the live state (params, targets, workspace, kernels) without running
	the loop. Always uses the P0.1 combined kernel. This is the POC's
	`setup_pipeline` folded into the main module so the main pipeline and the
	5-mode benchmark share one setup path.
	"""
	if target_layers is None:
		target_layers = [(23, 100)]
	if litho_cfg_path is None:
		litho_cfg_path = str(CONFIG_DIR / "lithosimple.txt")

	tile_files = discover_tile_files(oas_dir)
	if not tile_files:
		raise FileNotFoundError(f"目录中没有 oas 文件: {oas_dir}")
	first_info = parse_tile_info_from_filename(os.path.basename(tile_files[0]))
	tile_size = int(first_info.get("tile_size") or 1024)
	overlap = int(first_info.get("overlap") or 64)

	print(f"[setup_v7_context] {len(tile_files)} tiles, pointwise={pointwise}")
	target_masks: List[torch.Tensor] = []
	for oas_file in tile_files:
		mask, _ = read_oas_to_real_size_mask(
			oas_file=oas_file,
			target_layers=target_layers,
			target_size=tile_size,
			dtype=torch.float32,
		)
		target_masks.append(mask)
	if multiplier > 1:
		target_masks = target_masks * multiplier
	base_grid = infer_grid_size(len(target_masks))
	print(f"  扩展为 {len(target_masks)} 个 tiles ({base_grid}x{base_grid} grid)")

	# CPU SDF via ThreadPool (same as the eager path)
	max_workers = min(32, os.cpu_count() or 16)
	with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
		results = list(ex.map(_cpu_init_single_tile, target_masks))

	# Per-tile H2D (pinned + non_blocking), contiguous for the CUDA pointwise kernel
	params_list: List[torch.Tensor] = []
	target_cuda_list: List[torch.Tensor] = []
	for target_np, params_np in results:
		target_cuda_list.append(
			torch.from_numpy(target_np).pin_memory().to(DEVICE, dtype=torch.float32, non_blocking=True).contiguous()
		)
		params_list.append(
			torch.from_numpy(params_np).pin_memory().to(DEVICE, dtype=torch.float32, non_blocking=True).contiguous()
		)

	litho = lithosim.LithoSim(litho_cfg_path)
	opt_kernels = litho._kernels["focus"].kernels.to(DEVICE, dtype=torch.float32).contiguous()
	kernel_scales = litho._kernels["focus"].scales.to(DEVICE, dtype=torch.float32).contiguous()
	kernel_fft = _build_kernel_fft(opt_kernels, tile_size, tile_size)
	# P0.1: pre-combine the K=24 kernels into one (IFFT linearity), one-time.
	combined_kernel_fft = (
		kernel_scales.view(-1, 1, 1).to(kernel_fft.dtype) * kernel_fft
	).sum(dim=0)

	target_density = float(litho._config["TargetDensity"])
	print_steepness = float(litho._config["PrintSteepness"])

	workspace = StaticFusionSplitManager(
		base_grid=base_grid, s0=tile_size, overlap=overlap, device=DEVICE
	)
	_cuda_sync_if_needed()

	return {
		"litho": litho,
		"params_list": params_list,
		"target_cuda_list": target_cuda_list,
		"workspace": workspace,
		"kernel_fft": kernel_fft,
		"kernel_scales": kernel_scales,
		"combined_kernel_fft": combined_kernel_fft,
		"target_density": target_density,
		"print_steepness": print_steepness,
		"tile_size": tile_size,
		"overlap": overlap,
		"base_grid": base_grid,
		"learning_rate": learning_rate,
		"pointwise": pointwise,
	}


def _run_one_iter(ctx: Dict[str, Any], loss_accum: torch.Tensor, pointwise: Optional[str] = None) -> None:
	"""One graph-capture-friendly v7 iteration — pure GPU, no .item()/print/sync.

	`loss_accum` is a persistent GPU scalar that accumulates per-tile loss,
	replacing the eager path's `loss_sum += float(loss.item())` CPU sync (which
	would break capture). All reads/writes hit pre-allocated buffers, so the
	whole body is safe to record into a CUDA Graph.
	"""
	pw = pointwise or ctx.get("pointwise", "cuda")
	workspace = ctx["workspace"]
	workspace.reset_canvas()
	for tile_idx, (params, target) in enumerate(zip(ctx["params_list"], ctx["target_cuda_list"])):
		grad, loss_gpu = _pytorch_fft_grad_loss(
			params, target, ctx["kernel_fft"], ctx["kernel_scales"],
			ctx["target_density"], ctx["print_steepness"],
			combined_kernel_fft=ctx["combined_kernel_fft"],
			return_gpu_loss=True,
			pointwise=pw,
		)
		workspace.add_tile_grad_inplace(tile_idx, grad.to(torch.float32))
		loss_accum.add_(loss_gpu)
	fused = workspace.finalize_fuse()
	split_grads = workspace.split(fused)
	lr = ctx["learning_rate"]
	for i in range(len(ctx["params_list"])):
		ctx["params_list"][i].add_(-lr * split_grads[i])


def gradient_fusion_split_optimization_tiles_v7(
	mode: str,
	oas_dir: str,
	target_layers: Optional[List[Tuple[int, int]]] = None,
	iterations: int = 20,
	learning_rate: float = 0.025,
	eval_interval: int = 10,
	warmup: int = 10,
	levelset_cfg_path: Optional[str] = None,
	litho_cfg_path: Optional[str] = None,
) -> Dict[str, Any]:
	t_io_raster = 0.0
	t_h2d_init = 0.0
	t_levelset = 0.0
	t_levelset_micro_ms = 0.0
	t_fuse = 0.0
	t_split = 0.0
	t_apply = 0.0
	t_eval = 0.0
	t_d2h_save = 0.0
	_reset_peak_vram_if_needed()

	tile_files = discover_tile_files(oas_dir)
	if not tile_files:
		raise FileNotFoundError(f"目录中没有 oas 文件: {oas_dir}")

	first_info = parse_tile_info_from_filename(os.path.basename(tile_files[0]))
	tile_size = int(first_info.get("tile_size") or 1024)
	overlap = int(first_info.get("overlap") or 64)
	base_grid = infer_grid_size(len(tile_files))

	print(f"发现 tile: {len(tile_files)} 个，grid={base_grid}x{base_grid}, s={tile_size}, overlap={overlap}")
	print(f"[Mode] 核心梯度模式: {mode} | pointwise backend: {'triton' if _USE_TRITON_POINTWISE else 'cuda'}")

	target_masks: List[torch.Tensor] = []
	_nvtx_push("IO_Raster")
	t0 = time.time()
	for oas_file in tile_files:
		mask, _ = read_oas_to_real_size_mask(
			oas_file=oas_file,
			target_layers=target_layers,
			target_size=tile_size,
			dtype=torch.float32,
		)
		target_masks.append(mask)
	t_io_raster += time.time() - t0
	_nvtx_pop()
	check_vram("1. 刚读完数据(CPU)")

	multiplier = int(os.environ.get("FUILT_TILE_MULTIPLIER", "16"))
	if multiplier > 1:
		target_masks = target_masks * multiplier
		base_grid = infer_grid_size(len(target_masks))
		print(f"🔥 压测模式开启: 强制克隆为 {len(target_masks)} 个 Tiles, Grid={base_grid}x{base_grid}")

	_nvtx_push("H2D_Init")
	t0 = time.time()
	if litho_cfg_path is None:
		litho_cfg_path = str(CONFIG_DIR / "lithosimple.txt")
	if levelset_cfg_path is None:
		levelset_cfg_path = str(CONFIG_DIR / "pylevelset1024.txt")

	litho = lithosim.LithoSim(litho_cfg_path)
	params_list: List[torch.Tensor] = []
	target_cuda_list: List[torch.Tensor] = []

	print("🚀 启动 CPU 多线程并发计算 SDF...")
	max_workers = min(32, os.cpu_count() or 16)
	with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
		results = list(executor.map(_cpu_init_single_tile, target_masks))

	if _USE_BATCH_H2D:
		# P0.4: stack all tiles into [N, H, W] and do a single batched H2D.
		# Saves N-1 launch overheads vs per-tile H2D loop.
		all_targets_np = np.stack([t for t, _ in results], axis=0)
		all_params_np = np.stack([p for _, p in results], axis=0)

		# pin_memory on the stacked numpy → CUDA transfer goes through DMA.
		all_targets_pinned = torch.from_numpy(all_targets_np).pin_memory()
		all_params_pinned = torch.from_numpy(all_params_np).pin_memory()

		all_targets_gpu = all_targets_pinned.to(
			DEVICE, dtype=torch.float32, non_blocking=True
		).contiguous()
		all_params_gpu = all_params_pinned.to(
			DEVICE, dtype=torch.float32, non_blocking=True
		).contiguous()
		_cuda_sync_if_needed()

		# Unbind back to per-tile list (views into the contiguous batch tensor).
		target_cuda_list = [t.contiguous() for t in all_targets_gpu.unbind(0)]
		params_list = [p.contiguous() for p in all_params_gpu.unbind(0)]
	else:
		# Original per-tile H2D loop.
		for target_np, params_np in results:
			if _USE_H2D_NO_CONTIG:
				# P0.5: skip redundant .contiguous() — torch.from_numpy +
				# pin_memory + to(DEVICE) already returns a contiguous tensor.
				target_gpu = (
					torch.from_numpy(target_np)
					.pin_memory()
					.to(DEVICE, dtype=torch.float32, non_blocking=True)
				)
				params_gpu = (
					torch.from_numpy(params_np)
					.pin_memory()
					.to(DEVICE, dtype=torch.float32, non_blocking=True)
				)
			else:
				target_gpu = (
					torch.from_numpy(target_np)
					.pin_memory()
					.to(DEVICE, dtype=torch.float32, non_blocking=True)
					.contiguous()
				)
				params_gpu = (
					torch.from_numpy(params_np)
					.pin_memory()
					.to(DEVICE, dtype=torch.float32, non_blocking=True)
					.contiguous()
				)
			target_cuda_list.append(target_gpu)
			params_list.append(params_gpu)

	kernel_fft = None
	combined_kernel_fft: Optional[torch.Tensor] = None
	ext = None
	opt_kernels = litho._kernels["focus"].kernels.to(DEVICE, dtype=torch.float32).contiguous()
	kernel_scales = litho._kernels["focus"].scales.to(DEVICE, dtype=torch.float32).contiguous()
	if mode == "pytorch_fft":
		kernel_fft = _build_kernel_fft(opt_kernels, tile_size, tile_size)
		_ = load_pointwise_ext()
		# P0.1 optimization: pre-combine the K=24 kernels into one.
		# Mathematically equivalent (IFFT is linear), saves 23 ifft2 per tile.
		if _USE_COMBINED_KERNEL:
			combined_kernel_fft = (
				kernel_scales.view(-1, 1, 1).to(kernel_fft.dtype) * kernel_fft
			).sum(dim=0)
			print(f"[P0.1] combined_kernel_fft pre-computed: shape={tuple(combined_kernel_fft.shape)}")
	else:
		ext = _load_cuda_extension()

	target_density = float(litho._config["TargetDensity"])
	print_steepness = float(litho._config["PrintSteepness"])

	_cuda_sync_if_needed()
	t_h2d_init += time.time() - t0
	_nvtx_pop()
	check_vram("2. 参数刚上 GPU")

	workspace = StaticFusionSplitManager(
		base_grid=base_grid,
		s0=tile_size,
		overlap=overlap,
		device=DEVICE,
	)
	check_vram("3. 静态管家预分配后")

	if warmup > 0:
		for _ in range(warmup):
			sample_p = params_list[0]
			sample_t = target_cuda_list[0]
			if mode == "pytorch_fft":
				assert kernel_fft is not None
				_ = _pytorch_fft_grad_loss(
					sample_p, sample_t, kernel_fft, kernel_scales,
					target_density, print_steepness,
					combined_kernel_fft=combined_kernel_fft,
				)
			else:
				assert ext is not None
				_ = ext.levelset_step(sample_p, sample_t, opt_kernels, kernel_scales, target_density, print_steepness)
		_cuda_sync_if_needed()

	ok, msg = _memory_budget_ok()
	print(msg)
	if not ok:
		raise RuntimeError(msg)

	last_avg_loss = 0.0
	for it in range(iterations):
		_nvtx_push(f"iter_{it:03d}")
		loss_sum = 0.0
		workspace.reset_canvas()
		if it == 0:
			check_vram("4. [Iter 1] 刚进循环")

		_cuda_sync_if_needed()
		_nvtx_push("LevelSet")
		t_ls_start = time.time()
		ls_start_event = torch.cuda.Event(enable_timing=True)
		ls_end_event = torch.cuda.Event(enable_timing=True)
		stream = torch.cuda.current_stream()
		ls_start_event.record(stream)
		for tile_idx, (params, target_norm) in enumerate(zip(params_list, target_cuda_list)):
			if mode == "pytorch_fft":
				assert kernel_fft is not None
				grad, loss_val = _pytorch_fft_grad_loss(
					params, target_norm, kernel_fft, kernel_scales,
					target_density, print_steepness,
					combined_kernel_fft=combined_kernel_fft,
				)
			else:
				assert ext is not None
				grad = ext.levelset_step(
					params, target_norm, opt_kernels, kernel_scales, target_density, print_steepness
				)
				if kernel_fft is None:
					kernel_fft = _build_kernel_fft(opt_kernels, tile_size, tile_size)
				_, loss_val = _pytorch_fft_grad_loss(
					params, target_norm, cast(torch.Tensor, kernel_fft), kernel_scales,
					target_density, print_steepness,
					combined_kernel_fft=combined_kernel_fft,
				)

			workspace.add_tile_grad_inplace(tile_idx, grad.to(torch.float32))
			loss_sum += float(loss_val)

			if tile_idx == 0:
				grad_abs = grad.abs()
				print(
					f"[LevelSetDBG][iter={it+1:03d}][tile=0] "
					f"loss={loss_val:.6f}, grad_mean={grad_abs.mean().item():.6e}, grad_max={grad_abs.max().item():.6e}"
				)

		ls_end_event.record(stream)
		_cuda_sync_if_needed()
		t_levelset += time.time() - t_ls_start
		t_levelset_micro_ms += ls_start_event.elapsed_time(ls_end_event)
		_nvtx_pop()  # LevelSet
		if it == 0:
			check_vram("5. [Iter 1] LevelSet 后")

		_cuda_sync_if_needed()
		_nvtx_push("Fuse")
		t_fuse_start = time.time()
		fused_grad = workspace.finalize_fuse()
		_cuda_sync_if_needed()
		t_fuse += time.time() - t_fuse_start
		_nvtx_pop()  # Fuse
		if it == 0:
			check_vram("6. [Iter 1] Fuse 后")

		_cuda_sync_if_needed()
		_nvtx_push("Split")
		t_split_start = time.time()
		split_grads = workspace.split(fused_grad)
		_cuda_sync_if_needed()
		t_split += time.time() - t_split_start
		_nvtx_pop()  # Split
		if it == 0:
			check_vram("7. [Iter 1] Split 后")

		_cuda_sync_if_needed()
		_nvtx_push("Apply")
		t_apply_start = time.time()
		with torch.no_grad():
			for i in range(len(params_list)):
				params_list[i].add_(-learning_rate * split_grads[i])
		_cuda_sync_if_needed()
		t_apply += time.time() - t_apply_start
		_nvtx_pop()  # Apply
		if it == 0:
			check_vram("8. [Iter 1] Apply 后")

		last_avg_loss = loss_sum / len(params_list)

		if (it + 1) % max(1, eval_interval) == 0 or (it + 1) == iterations:
			check_vram(f"Eval前 [Iter {it+1}]")
			_cuda_sync_if_needed()
			t_eval_start = time.time()
			l2s, pvbs, epes = [], [], []
			with torch.no_grad():
				for params, target in zip(params_list, target_cuda_list):
					mask = (params < 0).to(torch.float32)
					l2, pvb, epe, _ = evaluation.evaluate(mask, target, litho, scale=1, shots=False)
					l2s.append(l2)
					pvbs.append(pvb)
					epes.append(epe)
			_cuda_sync_if_needed()
			t_eval += time.time() - t_eval_start
			check_vram(f"Eval后 [Iter {it+1}]")
			print(
				f"[Iter {it+1:03d}/{iterations}] avg_loss={last_avg_loss:.4f} | "
				f"Eval L2={np.mean(l2s):.2f}, EPE={np.mean(epes):.2f}"
			)
		else:
			print(f"[Iter {it+1:03d}/{iterations}] avg_loss={last_avg_loss:.4f}")

		_nvtx_pop()  # iter_{it:03d}

	_cuda_sync_if_needed()
	_nvtx_push("D2H_Save")
	t0 = time.time()
	out_tile_dir = str(ROOT / "tile_masks")
	os.makedirs(out_tile_dir, exist_ok=True)
	large_out_dir = ROOT / "large_mask_results"
	large_out_dir.mkdir(exist_ok=True)
	check_vram("9. 准备生成最终大图前")

	if _USE_ASYNC_D2H:
		# P0.3: pinned + non_blocking + overlap params D2H with workspace.fuse().
		# Pre-allocate pinned CPU buffers.
		params_cpu_list = [
			torch.empty_like(p, device="cpu", pin_memory=True)
			for p in params_list
		]
		# Submit all params D2H first (async on default stream).
		for p_gpu, p_pin in zip(params_list, params_cpu_list):
			p_pin.copy_(p_gpu, non_blocking=True)
		# Concurrently compute large_mask on GPU (reads params_list, no write).
		# This is the peak step (full-canvas ~3.8 GB). At 1024 tiles the run sits
		# within ~40 MB of the VRAM cliff, so return the caching allocator's free
		# blocks to the driver FIRST — otherwise a small transient alloc can OOM
		# even though the pool holds ~3.6 GB of reclaimable slack.
		torch.cuda.empty_cache()
		large_mask = workspace.fuse(params_list)
		check_vram("10. 最终大图生成后")
		# Now async D2H large_mask.
		large_mask_cpu = torch.empty_like(large_mask, device="cpu", pin_memory=True)
		large_mask_cpu.copy_(large_mask, non_blocking=True)
		# Wait for all D2H before handing data to ThreadPool.
		_cuda_sync_if_needed()
	else:
		# Original synchronous path.
		# Same as above: return the allocator's free blocks to the driver before
		# the peak large-mask step (see the async-branch comment).
		torch.cuda.empty_cache()
		large_mask = workspace.fuse(params_list)
		check_vram("10. 最终大图生成后")
		params_cpu_list = [p.detach().cpu() for p in params_list]
		large_mask_cpu = large_mask.detach().cpu()
	large_mask_path = str(large_out_dir / f"large_mask_v7_{large_mask_cpu.shape[1]}x{large_mask_cpu.shape[0]}.png")
	t_d2h_save += time.time() - t0
	_nvtx_pop()  # D2H_Save

	print("⏳ 触发后台异步写盘队列...")
	save_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
	for i, p_cpu in enumerate(params_cpu_list):
		save_executor.submit(_async_save_single_tile, p_cpu, i, out_tile_dir)
	save_executor.submit(_async_save_large_mask, large_mask_cpu, large_mask_path)

	peak_memory_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0
	total_calls = iterations * len(params_list)
	avg_micro_ms = t_levelset_micro_ms / max(1, total_calls)

	return {
		"mode": mode,
		"num_tiles": len(params_list),
		"grid": base_grid,
		"tile_size": tile_size,
		"overlap": overlap,
		"iterations": iterations,
		"final_loss": last_avg_loss,
		"tile_masks_dir": out_tile_dir,
		"large_mask_path": large_mask_path,
		"save_executor": save_executor,
		"profile_times": {
			"1. IO_Raster": t_io_raster,
			"2. H2D_Init": t_h2d_init,
			"3.1 LevelSet": t_levelset,
			"3.2 Fuse": t_fuse,
			"3.3 Split": t_split,
			"3.4 Apply": t_apply,
			"4. D2H_Save": t_d2h_save,
			"X. Eval(Excluded)": t_eval,
		},
		"levelset_micro_total_ms": t_levelset_micro_ms,
		"levelset_micro_avg_ms_per_tile": avg_micro_ms,
		"peak_vram_mb": peak_memory_mb,
	}


def _run_graph_mode(
	oas_dir: str,
	iterations: int = 20,
	warmup: int = 5,
	multiplier: int = 16,
	pointwise: str = "triton",
	learning_rate: float = 0.025,
	target_layers: Optional[List[Tuple[int, int]]] = None,
	litho_cfg_path: Optional[str] = None,
) -> Dict[str, Any]:
	"""v7 pipeline with one captured iteration replayed `iterations` times.

	Flow: setup (setup_v7_context) -> warmup (JIT + allocation warmup) ->
	capture one `_run_one_iter` under torch.cuda.graph -> replay N times ->
	final eval + D2H, all OUTSIDE capture. Reports `graph_replay_ms_per_iter`.

	pointwise: "cuda" (hand-written kernel, capture-safe via the stream fix) or
	"triton" (capture-safe by default). "cuda" requires set_pointwise_capture_safe(True).
	"""
	_reset_peak_vram_if_needed()
	if target_layers is None:
		target_layers = [(23, 100)]
	if litho_cfg_path is None:
		litho_cfg_path = str(CONFIG_DIR / "lithosimple.txt")

	t_setup0 = time.perf_counter()
	ctx = setup_v7_context(
		oas_dir, multiplier=multiplier, pointwise=pointwise,
		learning_rate=learning_rate, target_layers=target_layers,
		litho_cfg_path=litho_cfg_path,
	)
	setup_s = time.perf_counter() - t_setup0
	params_list = ctx["params_list"]
	target_cuda_list = ctx["target_cuda_list"]
	workspace = ctx["workspace"]
	litho = ctx["litho"]

	# Hand-written CUDA kernel must launch on the capture stream to be recorded.
	# (Triton honors the capture stream automatically.)
	if pointwise == "cuda":
		set_pointwise_capture_safe(True)

	loss_accum = torch.zeros(1, device=DEVICE, dtype=torch.float32)

	print(f"\n[Graph] warmup {warmup} iters (pointwise={pointwise})...")
	for _ in range(warmup):
		_run_one_iter(ctx, loss_accum, pointwise=pointwise)
	_cuda_sync_if_needed()

	ok, msg = _memory_budget_ok()
	print(msg)
	if not ok:
		raise RuntimeError(msg)

	print("[Graph] capturing one iteration...")
	g = torch.cuda.CUDAGraph()
	capture_t0 = time.time()
	with torch.cuda.graph(g):
		_run_one_iter(ctx, loss_accum, pointwise=pointwise)
	capture_s = time.time() - capture_t0
	print(f"  capture: OK ({capture_s:.3f} s)")

	# Zero the GPU loss accumulator in-place (same data_ptr) so the timed
	# replays accumulate a clean per-iter loss. Graphs replay operations, so
	# in-place ops on persistent buffers reflect the current value on replay.
	loss_accum.zero_()
	print(f"[Graph] replay {iterations} times...")
	_cuda_sync_if_needed()
	t0 = time.perf_counter()
	for _ in range(iterations):
		g.replay()
	_cuda_sync_if_needed()
	replay_s = time.perf_counter() - t0
	replay_ms_per_iter = replay_s / iterations * 1000.0
	# loss_accum now holds sum over (iterations * num_tiles) of per-tile loss.
	last_avg_loss = float(loss_accum.item()) / max(1, iterations * len(params_list))
	print(f"  graph replay: {replay_s:.3f} s ({replay_ms_per_iter:.2f} ms/iter), avg_loss={last_avg_loss:.4f}")

	# Final eval (outside capture) — same evaluator as the eager path.
	print("[Graph] final eval (outside capture)...")
	_cuda_sync_if_needed()
	t_eval_start = time.time()
	l2s, epes = [], []
	with torch.no_grad():
		for params, target in zip(params_list, target_cuda_list):
			mask = (params < 0).to(torch.float32)
			l2, _pvb, epe, _ = evaluation.evaluate(mask, target, litho, scale=1, shots=False)
			l2s.append(l2)
			epes.append(epe)
	_cuda_sync_if_needed()
	t_eval = time.time() - t_eval_start
	print(f"  Eval L2={np.mean(l2s):.2f}, EPE={np.mean(epes):.2f}")

	# D2H + async save (same helpers as the eager path).
	# Same peak-step protection as the eager path: return the allocator's free
	# blocks to the driver before the full-canvas fuse (defensive — graph mode
	# passed 2/2 without it, but keep both paths uniform).
	t_d2h0 = time.perf_counter()
	torch.cuda.empty_cache()
	out_tile_dir = str(ROOT / "tile_masks")
	os.makedirs(out_tile_dir, exist_ok=True)
	large_out_dir = ROOT / "large_mask_results"
	large_out_dir.mkdir(exist_ok=True)
	if _USE_ASYNC_D2H:
		# P0.3 ported into the graph branch (2026-08-26): pinned host buffers +
		# non_blocking copies — params D2H overlaps the GPU fuse, one sync at end.
		params_cpu_list = [torch.empty_like(p, device="cpu", pin_memory=True)
		                   for p in params_list]
		for p_gpu, p_pin in zip(params_list, params_cpu_list):
			p_pin.copy_(p_gpu, non_blocking=True)
		large_mask = workspace.fuse(params_list)
		large_mask_cpu = torch.empty_like(large_mask, device="cpu", pin_memory=True)
		large_mask_cpu.copy_(large_mask, non_blocking=True)
		torch.cuda.synchronize()
	else:
		# original synchronous serial path (baseline; pre-2026-08-26 graph branch)
		large_mask = workspace.fuse(params_list)
		params_cpu_list = [p.detach().cpu() for p in params_list]
		large_mask_cpu = large_mask.detach().cpu()
	d2h_s = time.perf_counter() - t_d2h0
	large_mask_path = str(large_out_dir / f"large_mask_v7graph_{large_mask_cpu.shape[1]}x{large_mask_cpu.shape[0]}.png")
	save_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
	for i, p_cpu in enumerate(params_cpu_list):
		save_executor.submit(_async_save_single_tile, p_cpu, i, out_tile_dir)
	save_executor.submit(_async_save_large_mask, large_mask_cpu, large_mask_path)

	peak_memory_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0

	return {
		"mode": "pytorch_fft",
		"pointwise": pointwise,
		"num_tiles": len(params_list),
		"grid": ctx["base_grid"],
		"tile_size": ctx["tile_size"],
		"iterations": iterations,
		"final_loss": last_avg_loss,
		"graph_capture_s": capture_s,
		"graph_replay_ms_per_iter": replay_ms_per_iter,
		"graph_setup_s": setup_s,
		"graph_d2h_save_s": d2h_s,
		"graph_pure_pipeline_s": setup_s + capture_s + replay_s + d2h_s,
		"eval_l2": float(np.mean(l2s)),
		"eval_epe": float(np.mean(epes)),
		"eval_time_s": t_eval,
		"tile_masks_dir": out_tile_dir,
		"large_mask_path": large_mask_path,
		"save_executor": save_executor,
		"peak_vram_mb": peak_memory_mb,
	}


def _write_profile_json(result: Dict[str, Any], total_s: float, mode_label: str) -> None:
	"""Persist the v7 pipeline's timing profile as JSON (committed evidence in
	benchmarks/results/; the ledger docs/verifiable_results.md references this +
	the rerun command). Defensive: works for both the eager and graph-mode
	result dicts."""
	import json

	profile_times = result.get("profile_times")
	payload = {
		"script": "baseline/test_baseline_v7_fuselevelset.py",
		"date": time.strftime("%Y-%m-%d %H:%M:%S"),
		"mode": mode_label,
		"pointwise": result.get("pointwise", "n/a"),
		"num_tiles": result.get("num_tiles"),
		"grid": result.get("grid"),
		"tile_size": result.get("tile_size"),
		"iterations": result.get("iterations"),
		"final_loss": result.get("final_loss"),
		"total_e2e_s": total_s,
		"peak_vram_mb": result.get("peak_vram_mb"),
		"profile_times_s": profile_times,
		"levelset_micro_avg_ms_per_tile": result.get("levelset_micro_avg_ms_per_tile"),
		"graph_capture_s": result.get("graph_capture_s"),
		"graph_replay_ms_per_iter": result.get("graph_replay_ms_per_iter"),
		"eval_l2": result.get("eval_l2"),
		"eval_epe": result.get("eval_epe"),
	}
	out_path = Path(os.environ.get("FUILT_BENCH_OUT", "benchmarks/results")) / "v7_profile.json"
	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
	print(f"[WROTE] {out_path}")


def main() -> None:
	if not torch.cuda.is_available():
		raise RuntimeError("test_baseline_v7 需要 CUDA 环境")

	parser = argparse.ArgumentParser(description="v7 full-pipeline benchmark with fused pointwise levelset")
	parser.add_argument(
		"--mode",
		type=str,
		default=os.environ.get("FUILT_BENCH_MODE", "pytorch_fft"),
		choices=["pytorch_fft", "cuda_op"],
		help="选择要测试的核心梯度版本",
	)
	args = parser.parse_args()

	# === Graph-capture path (FUILT_USE_GRAPH=1) ===
	# Captures one full v7 iteration and replays it; only the FFT path is wired
	# here. Falls through to the eager path if mode != pytorch_fft.
	if _USE_GRAPH and args.mode == "pytorch_fft":
		tiles_dir = os.environ.get("FUILT_TILES_DIR", "/data/lyj/FuILT/tiles_from_patch")
		iters = int(os.environ.get("FUILT_ITERS", "20"))
		lr = float(os.environ.get("FUILT_LR", "0.025"))
		warmup = int(os.environ.get("FUILT_BENCH_WARMUP", "5"))
		multiplier = int(os.environ.get("FUILT_TILE_MULTIPLIER", "16"))
		pointwise = "triton" if _USE_TRITON_POINTWISE else "cuda"

		total_t0 = time.time()
		result = _run_graph_mode(
			oas_dir=tiles_dir, iterations=iters, warmup=warmup,
			multiplier=multiplier, pointwise=pointwise, learning_rate=lr,
			litho_cfg_path=str(CONFIG_DIR / "lithosimple.txt"),
		)
		total_s = time.time() - total_t0

		print("\n" + "=" * 50)
		print("🚀 v7 Graph-Capture Profiling Report (图捕获性能报告)")
		print("=" * 50)
		print(
			f"📌 Mode: pytorch_fft (graph replay) | pointwise: {result['pointwise']} | "
			f"{result['num_tiles']} 个 ({result['grid']}x{result['grid']}), "
			f"{result['tile_size']}x{result['tile_size']}"
		)
		print(f"🔥 峰值显存 (Peak VRAM): {result['peak_vram_mb']:.2f} MB")
		print(f"🎬 图捕获耗时 (one-shot, amortized): {result['graph_capture_s']:.3f} s")
		print(f"⚡ Graph replay: {result['graph_replay_ms_per_iter']:.3f} ms/iter")
		print(f"💡 流水线总纯耗时 (graph, 剔除Eval): {result['graph_pure_pipeline_s']:.3f} s "
		      f"[setup {result['graph_setup_s']:.3f} + capture {result['graph_capture_s']:.3f} "
		      f"+ replay {result['graph_replay_ms_per_iter'] * result['iterations'] / 1000:.3f} "
		      f"+ d2h {result['graph_d2h_save_s']:.3f}]")
		print(f"📊 Eval L2={result['eval_l2']:.2f}, EPE={result['eval_epe']:.2f} | avg_loss={result['final_loss']:.4f}")
		print(f"⏱️ 端到端总耗时: {total_s:.3f} s")
		print("=" * 50 + "\n")

		_write_profile_json(result, total_s, "pytorch_fft (graph replay)")
		print("⏳ 等待后台写盘任务完成...")
		cast(concurrent.futures.ThreadPoolExecutor, result["save_executor"]).shutdown(wait=True)
		print("✅ 所有图片与权重保存完毕！")
		return

	tiles_dir = os.environ.get("FUILT_TILES_DIR", "/data/lyj/FuILT/tiles_from_patch")
	iters = int(os.environ.get("FUILT_ITERS", "20"))
	lr = float(os.environ.get("FUILT_LR", "0.025"))
	eval_interval = int(os.environ.get("FUILT_EVAL_INTERVAL", "10"))
	warmup = int(os.environ.get("FUILT_BENCH_WARMUP", "10"))

	total_t0 = time.time()
	result = gradient_fusion_split_optimization_tiles_v7(
		mode=args.mode,
		oas_dir=tiles_dir,
		target_layers=[(23, 100)],
		iterations=iters,
		learning_rate=lr,
		eval_interval=eval_interval,
		warmup=warmup,
		levelset_cfg_path=str(CONFIG_DIR / "pylevelset1024.txt"),
		litho_cfg_path=str(CONFIG_DIR / "lithosimple.txt"),
	)
	total_t1 = time.time()

	times = cast(Dict[str, float], result["profile_times"])
	total_valid_time = (
		times["1. IO_Raster"]
		+ times["2. H2D_Init"]
		+ times["3.1 LevelSet"]
		+ times["3.2 Fuse"]
		+ times["3.3 Split"]
		+ times["3.4 Apply"]
		+ times["4. D2H_Save"]
	)

	print("\n" + "=" * 50)
	print("🚀 Baseline Profiling Report (基线性能分析报告)")
	print("=" * 50)
	print(
		f"📌 Mode: {result['mode']} | Tile 规格: {result['num_tiles']} 个 ({result['grid']}x{result['grid']}), "
		f"大小 {result['tile_size']}x{result['tile_size']}"
	)
	print(f"🔥 峰值显存 (Peak VRAM): {result['peak_vram_mb']:.2f} MB")

	print("-" * 50)
	print(f"{'阶段 (Pipeline Stage)':<25} | {'耗时 (s)':<10} | {'占比 (%)'}")
	print("-" * 50)
	for stage, t in times.items():
		if "Eval" in stage:
			continue
		pct = (t / total_valid_time) * 100 if total_valid_time > 0 else 0.0
		print(f"{stage:<25} | {t:>8.3f} s | {pct:>5.1f} %")
	print("-" * 50)
	print(f"💡 流水线总纯耗时 (剔除Eval): {total_valid_time:.3f} s")
	print(f"⏱️ 脚本端到端总耗时 (含Eval): {total_t1 - total_t0:.3f} s")
	print("-" * 50)
	print(f"🔬 核心算子精确耗时 (Micro-Bench): {result['levelset_micro_avg_ms_per_tile']:.3f} ms / Tile")
	print("   (注: 这代表去除了 Python 调度开销后，纯 GPU 的极限速度)")
	print("=" * 50 + "\n")

	_write_profile_json(result, total_t1 - total_t0, str(result["mode"]))
	print("⏳ 性能报告已输出，正在等待后台写盘任务完成...")
	cast(concurrent.futures.ThreadPoolExecutor, result["save_executor"]).shutdown(wait=True)
	print("✅ 所有图片与权重保存完毕！")


if __name__ == "__main__":
	main()

'''
python -m baseline.test_baseline_v7_fuselevelset --mode pytorch_fft
python -m baseline.test_baseline_v7_fuselevelset --mode cuda_op
CUDA_VISIBLE_DEVICES=1 FUILT_TILE_MULTIPLIER=16 python -m baseline.test_baseline_v7_fuselevelset --mode pytorch_fft
'''