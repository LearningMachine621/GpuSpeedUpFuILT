import argparse
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
from fuilt.fusion_split.split_grads import hierarchical_split_grads
from fuilt.ilt.func import evaluation
from fuilt.ilt.func import initializer
from fuilt.ilt.func import simple as lithosim
from fuilt.pre_work.read_oas2mask import parse_tile_info_from_filename, read_oas_to_real_size_mask

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
CUDA_SRC = ROOT / "src" / "fuilt" / "ilt" / "func" / "levelset_simple.cu"


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
    return sorted(files, key=_extract_row_col)


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
            "-gencode", "arch=compute_89,code=sm_89", # 针对 4090 优化，避免架构警告
            "-Xptxas", "-v",                          # 打印寄存器和共享内存使用量
            "-Xptxas", "-warn-spills"                 # 如果发生寄存器溢出，抛出警告
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


def _pytorch_fft_grad_loss(
    params_in: torch.Tensor,
    target_mask: torch.Tensor,
    kernel_fft: torch.Tensor,
    kernel_scales: torch.Tensor,
    target_density: float,
    print_steepness: float,
) -> Tuple[torch.Tensor, float]:
    mask_fft = torch.fft.fft2(params_in.to(torch.complex64), norm="forward")
    conv_freq = mask_fft.unsqueeze(0) * kernel_fft
    conv_spatial = torch.fft.ifft2(conv_freq, norm="forward").real
    aerial = (conv_spatial * kernel_scales.view(-1, 1, 1)).sum(dim=0)

    printed = torch.sigmoid((aerial - target_density) * print_steepness)
    diff = printed - target_mask
    loss = torch.sum(diff * diff)

    dL_dprinted = 2.0 * diff
    dprinted_daerial = print_steepness * printed * (1.0 - printed)
    grad = (dL_dprinted * dprinted_daerial).contiguous()
    return grad, float(loss.item())


def gradient_fusion_split_optimization_tiles_v2(
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
    # ================= [PROFILING 计时器初始化] =================
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
    print(f"[Mode] 核心梯度模式: {mode}")

    # ================= 1. OAS读取与光栅化 =================
    target_masks: List[torch.Tensor] = []
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

    # ================= [压测克隆魔法] =================
    multiplier = int(os.environ.get("FUILT_TILE_MULTIPLIER", "16"))
    if multiplier > 1:
        target_masks = target_masks * multiplier
        base_grid = infer_grid_size(len(target_masks))
        print(f"🔥 压测模式开启: 强制克隆为 {len(target_masks)} 个 Tiles, Grid={base_grid}x{base_grid}")

    # ================= 2. H2D & 初始化 =================
    t0 = time.time()
    if litho_cfg_path is None:
        litho_cfg_path = str(CONFIG_DIR / "lithosimple.txt")
    if levelset_cfg_path is None:
        levelset_cfg_path = str(CONFIG_DIR / "pylevelset1024.txt")

    litho = lithosim.LithoSim(litho_cfg_path)
    params_list: List[torch.Tensor] = []
    target_cuda_list: List[torch.Tensor] = []

    for target in target_masks:
        target_gpu = target.to(DEVICE)
        target_norm = target_gpu / 255.0 if target_gpu.max() > 1.5 else target_gpu
        _, params = initializer.LevelSetImageInit().run(target_norm.detach().cpu().numpy())
        if not isinstance(params, torch.Tensor):
            params = torch.tensor(params, dtype=torch.float32)
        params_list.append(params.detach().to(DEVICE, dtype=torch.float32).contiguous())
        target_cuda_list.append(target_norm.detach().to(DEVICE, dtype=torch.float32).contiguous())

    # 预计算频域核（仅 pytorch_fft）
    kernel_fft = None
    ext = None
    if mode == "pytorch_fft":
        opt_kernels = litho._kernels["focus"].kernels.to(DEVICE, dtype=torch.float32).contiguous()
        kernel_fft = _build_kernel_fft(opt_kernels, tile_size, tile_size)
        kernel_scales = litho._kernels["focus"].scales.to(DEVICE, dtype=torch.float32).contiguous()
    else:
        opt_kernels = litho._kernels["focus"].kernels.to(DEVICE, dtype=torch.float32).contiguous()
        kernel_scales = litho._kernels["focus"].scales.to(DEVICE, dtype=torch.float32).contiguous()
        ext = _load_cuda_extension()

    target_density = float(litho._config["TargetDensity"])
    print_steepness = float(litho._config["PrintSteepness"])

    _cuda_sync_if_needed()
    t_h2d_init += time.time() - t0

    # Warm-up (核心算子)
    if warmup > 0:
        for _ in range(warmup):
            sample_p = params_list[0]
            sample_t = target_cuda_list[0]
            if mode == "pytorch_fft":
                assert kernel_fft is not None
                _ = _pytorch_fft_grad_loss(sample_p, sample_t, kernel_fft, kernel_scales, target_density, print_steepness)
            else:
                assert ext is not None
                _ = ext.levelset_step(sample_p, sample_t, opt_kernels, kernel_scales, target_density, print_steepness)
        _cuda_sync_if_needed()

    # ================= 3. 主循环 =================
    last_avg_loss = 0.0
    for it in range(iterations):
        grads: List[torch.Tensor] = []
        loss_sum = 0.0

        # --- 3.1 LevelSet 核心梯度 ---
        _cuda_sync_if_needed()
        t_ls_start = time.time()
        ls_start_event = torch.cuda.Event(enable_timing=True)
        ls_end_event = torch.cuda.Event(enable_timing=True)

        stream = torch.cuda.current_stream()
        ls_start_event.record(stream)
        for tile_idx, (params, target_norm) in enumerate(zip(params_list, target_cuda_list)):
            if mode == "pytorch_fft":
                assert kernel_fft is not None
                grad, loss_val = _pytorch_fft_grad_loss(
                    params, target_norm, kernel_fft, kernel_scales, target_density, print_steepness
                )
            else:
                assert ext is not None
                grad = ext.levelset_step(
                    params, target_norm, opt_kernels, kernel_scales, target_density, print_steepness
                )
                # 为保持 avg_loss 输出格式，这里用同链路前向补一次 loss（不计入微观计时）。
                assert kernel_fft is not None or True
                if kernel_fft is None:
                    kernel_fft = _build_kernel_fft(opt_kernels, tile_size, tile_size)
                _, loss_val = _pytorch_fft_grad_loss(
                    params, target_norm, cast(torch.Tensor, kernel_fft), kernel_scales, target_density, print_steepness
                )

            grads.append(grad.to(torch.float32))
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

        # --- 3.2 Fusion ---
        _cuda_sync_if_needed()
        t_fuse_start = time.time()
        fused = hierarchical_fuse_grads(grads=grads, base_grid=base_grid, overlap_ratio=overlap_ratio)
        fused_grad = cast(torch.Tensor, fused["top"])
        _cuda_sync_if_needed()
        t_fuse += time.time() - t_fuse_start

        # --- 3.3 Split ---
        _cuda_sync_if_needed()
        t_split_start = time.time()
        split_grads = hierarchical_split_grads(
            fused_top=fused_grad,
            base_grid=base_grid,
            overlap_ratio=overlap_ratio,
            s0=tile_size,
        )
        _cuda_sync_if_needed()
        t_split += time.time() - t_split_start

        # --- 3.4 Apply ---
        _cuda_sync_if_needed()
        t_apply_start = time.time()
        for i in range(len(params_list)):
            params_list[i].add_(-learning_rate * split_grads[i])
        _cuda_sync_if_needed()
        t_apply += time.time() - t_apply_start

        last_avg_loss = loss_sum / len(params_list)

        # --- Eval ---
        if (it + 1) % max(1, eval_interval) == 0 or (it + 1) == iterations:
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
            print(
                f"[Iter {it+1:03d}/{iterations}] avg_loss={last_avg_loss:.4f} | "
                f"Eval L2={np.mean(l2s):.2f}, EPE={np.mean(epes):.2f}"
            )
        else:
            print(f"[Iter {it+1:03d}/{iterations}] avg_loss={last_avg_loss:.4f}")

    # ================= 4. D2H 保存 =================
    _cuda_sync_if_needed()
    t0 = time.time()
    out_tile_dir = str(ROOT / "tile_masks")
    _save_tile_masks(params_list, out_tile_dir)
    large_mask = _build_large_mask_from_params(params_list=params_list, base_grid=base_grid, overlap_ratio=overlap_ratio)
    large_mask_img = (large_mask.detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
    large_out_dir = ROOT / "large_mask_results"
    large_out_dir.mkdir(exist_ok=True)
    large_mask_path = str(large_out_dir / f"large_mask_v2_{large_mask_img.shape[1]}x{large_mask_img.shape[0]}.png")
    cv2.imwrite(large_mask_path, large_mask_img)
    t_d2h_save += time.time() - t0

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


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("test_baseline_v2 需要 CUDA 环境")

    parser = argparse.ArgumentParser(description="v2 full-pipeline benchmark with precise GPU micro timing")
    parser.add_argument(
        "--mode",
        type=str,
        default=os.environ.get("FUILT_BENCH_MODE", "pytorch_fft"),
        choices=["pytorch_fft", "cuda_op"],
        help="选择要测试的核心梯度版本",
    )
    args = parser.parse_args()

    tiles_dir = os.environ.get("FUILT_TILES_DIR", "/data/lyj/FuILT/tiles_from_patch")
    iters = int(os.environ.get("FUILT_ITERS", "20"))
    lr = float(os.environ.get("FUILT_LR", "0.025"))
    eval_interval = int(os.environ.get("FUILT_EVAL_INTERVAL", "10"))
    warmup = int(os.environ.get("FUILT_BENCH_WARMUP", "10"))

    total_t0 = time.time()
    result = gradient_fusion_split_optimization_tiles_v2(
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


if __name__ == "__main__":
    main()
'''
python -m baseline.test_baseline_v2 --mode pytorch_fft
python -m baseline.test_baseline_v2 --mode cuda_op
CUDA_VISIBLE_DEVICES=1 python -m baseline.test_baseline_v1 --mode pytorch_fft
'''