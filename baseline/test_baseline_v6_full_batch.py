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

from fuilt.fusion_split.write_grads_inplace import StaticFusionSplitManager
from fuilt.ilt.func import evaluation
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


def check_vram(tag: str) -> None:
    if not torch.cuda.is_available():
        return
    current_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(f"[{tag:^20}] 实时占用: {current_mb:8.2f} MB | 历史峰值: {peak_mb:8.2f} MB")


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
    # SDF convention: negative = inside mask (see _build_large_mask_from_params).
    # (> 0.5) was a display bug drawing the mask's complement (SDF-positive outside).
    large_mask_img = (large_mask_cpu.numpy() < 0).astype(np.uint8) * 255
    cv2.imwrite(large_mask_path, large_mask_img)


def _pytorch_fft_grad_loss_batched(
    params_in: torch.Tensor,      # [B, H, W]
    target_mask: torch.Tensor,    # [B, H, W]
    kernel_fft: torch.Tensor,     # [K, H, W]
    kernel_scales: torch.Tensor,  # [K]
    target_density: float,
    print_steepness: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 批量 FFT：[B,H,W] -> [B,H,W] complex
    mask_fft = torch.fft.fft2(params_in.to(torch.complex64), norm="forward")

    # 避免构造 [B,K,H,W] 超大张量，按 K 维流式累加
    aerial = torch.zeros_like(params_in, dtype=torch.float32)
    for k in range(kernel_fft.shape[0]):
        conv_freq_k = mask_fft * kernel_fft[k].unsqueeze(0)
        conv_spatial_k = torch.fft.ifft2(conv_freq_k, norm="forward").real
        aerial += conv_spatial_k * kernel_scales[k]

    printed = torch.sigmoid((aerial - target_density) * print_steepness)
    diff = printed - target_mask
    loss = torch.sum(diff * diff)

    dL_dprinted = 2.0 * diff
    dprinted_daerial = print_steepness * printed * (1.0 - printed)
    grad = (dL_dprinted * dprinted_daerial).contiguous()
    return grad, loss


def gradient_fusion_split_optimization_tiles_v6(
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
    print(f"[Mode] 核心梯度模式: {mode}")

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
    check_vram("1. 刚读完数据(CPU)")

    multiplier = int(os.environ.get("FUILT_TILE_MULTIPLIER", "16"))
    if multiplier > 1:
        target_masks = target_masks * multiplier
        base_grid = infer_grid_size(len(target_masks))
        print(f"🔥 压测模式开启: 强制克隆为 {len(target_masks)} 个 Tiles, Grid={base_grid}x{base_grid}")

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

    for target_np, params_np in results:
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

    opt_kernels = litho._kernels["focus"].kernels.to(DEVICE, dtype=torch.float32).contiguous()
    kernel_scales = litho._kernels["focus"].scales.to(DEVICE, dtype=torch.float32).contiguous()
    kernel_fft = _build_kernel_fft(opt_kernels, tile_size, tile_size)

    ext = None
    if mode == "cuda_op":
        ext = _load_cuda_extension()

    target_density = float(litho._config["TargetDensity"])
    print_steepness = float(litho._config["PrintSteepness"])

    _cuda_sync_if_needed()
    t_h2d_init += time.time() - t0
    check_vram("2. 参数刚上 GPU")

    workspace = StaticFusionSplitManager(
        base_grid=base_grid,
        s0=tile_size,
        overlap=overlap,
        device=DEVICE,
    )
    check_vram("3. 静态管家预分配后")

    # ================= 3. 主循环 (全量 Batch + AMP + Graph) =================
    default_amp = "0" if mode == "pytorch_fft" else "1"
    use_amp = bool(int(os.environ.get("FUILT_USE_AMP", default_amp)))
    amp_dtype = torch.float16

    params_batch = torch.stack(params_list)
    target_batch = torch.stack(target_cuda_list)
    split_grads_batch = torch.zeros_like(params_batch)
    static_loss_sum = torch.zeros(1, dtype=torch.float32, device=DEVICE)

    def run_one_iteration() -> None:
        if mode == "pytorch_fft":
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                batched_grads, loss_tensor = _pytorch_fft_grad_loss_batched(
                    params_batch,
                    target_batch,
                    kernel_fft,
                    kernel_scales,
                    target_density,
                    print_steepness,
                )
        else:
            assert ext is not None
            # cuda_op 分支仍然用 batch 结果容器，但算子按 tile 调用后组 batch，保证接口兼容。
            grads_list: List[torch.Tensor] = []
            loss_tensor = torch.zeros(1, dtype=torch.float32, device=DEVICE)
            for i in range(params_batch.shape[0]):
                grad_i = ext.levelset_step(
                    params_batch[i],
                    target_batch[i],
                    opt_kernels,
                    kernel_scales,
                    target_density,
                    print_steepness,
                )
                grads_list.append(grad_i)
                _, li = _pytorch_fft_grad_loss_batched(
                    params_batch[i:i+1],
                    target_batch[i:i+1],
                    kernel_fft,
                    kernel_scales,
                    target_density,
                    print_steepness,
                )
                loss_tensor.add_(li)
            batched_grads = torch.stack(grads_list)

        static_loss_sum.copy_(loss_tensor)

        fused_grad = workspace.fuse_batched(batched_grads)
        workspace.split_batched(fused_grad, split_grads_batch)
        params_batch.add_(split_grads_batch, alpha=-learning_rate)

    print("🔥 正在预热 CUDA Graph...")
    for _ in range(max(1, min(3, warmup))):
        run_one_iteration()
    _cuda_sync_if_needed()

    print("📸 正在捕获计算图...")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run_one_iteration()

    print("🚀 极速回放模式启动！")
    if iterations > 0:
        check_vram("4. [Iter 1] 刚进循环")

    _cuda_sync_if_needed()
    t_main_loop_start = time.time()
    loop_start_event = torch.cuda.Event(enable_timing=True)
    loop_end_event = torch.cuda.Event(enable_timing=True)
    loop_stream = torch.cuda.current_stream()

    loop_start_event.record(loop_stream)
    for it in range(iterations):
        g.replay()
        if it == 0:
            _cuda_sync_if_needed()
            check_vram("8. [Iter 1] Apply 后")

        if (it + 1) % max(1, eval_interval) == 0 or (it + 1) == iterations:
            _cuda_sync_if_needed()
            last_avg_loss = static_loss_sum.item() / max(1, params_batch.shape[0])
            check_vram(f"Eval前 [Iter {it+1}]")

            t_eval_start = time.time()
            l2s, pvbs, epes = [], [], []
            with torch.no_grad():
                for i in range(params_batch.shape[0]):
                    mask = (params_batch[i] < 0).to(torch.float32)
                    l2, pvb, epe, _ = evaluation.evaluate(mask, target_batch[i], litho, scale=1, shots=False)
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
            print(f"[Iter {it+1:03d}/{iterations}] replayed")

    loop_end_event.record(loop_stream)
    _cuda_sync_if_needed()

    t_levelset = time.time() - t_main_loop_start
    t_levelset_micro_ms = loop_start_event.elapsed_time(loop_end_event)
    t_fuse = 0.0
    t_split = 0.0
    t_apply = 0.0

    # ================= 4. D2H 保存 =================
    _cuda_sync_if_needed()
    t0 = time.time()
    out_tile_dir = str(ROOT / "tile_masks")
    os.makedirs(out_tile_dir, exist_ok=True)
    large_out_dir = ROOT / "large_mask_results"
    large_out_dir.mkdir(exist_ok=True)
    check_vram("9. 准备生成最终大图前")

    large_mask = workspace.fuse_batched(params_batch)
    check_vram("10. 最终大图生成后")

    params_cpu_list = list(params_batch.detach().cpu())
    large_mask_cpu = large_mask.detach().cpu()
    large_mask_path = str(large_out_dir / f"large_mask_v6_{large_mask_cpu.shape[1]}x{large_mask_cpu.shape[0]}.png")
    t_d2h_save += time.time() - t0

    print("⏳ 触发后台异步写盘队列...")
    save_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    for i, p_cpu in enumerate(params_cpu_list):
        save_executor.submit(_async_save_single_tile, p_cpu, i, out_tile_dir)
    save_executor.submit(_async_save_large_mask, large_mask_cpu, large_mask_path)

    peak_memory_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0
    total_calls = iterations * len(params_cpu_list)
    avg_micro_ms = t_levelset_micro_ms / max(1, total_calls)
    last_avg_loss = static_loss_sum.item() / max(1, params_batch.shape[0])

    return {
        "mode": mode,
        "num_tiles": len(params_cpu_list),
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


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("test_baseline_v6 需要 CUDA 环境")

    parser = argparse.ArgumentParser(description="v6 full-batch + amp + cuda graph benchmark")
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
    result = gradient_fusion_split_optimization_tiles_v6(
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

    print("⏳ 性能报告已输出，正在等待后台写盘任务完成...")
    cast(concurrent.futures.ThreadPoolExecutor, result["save_executor"]).shutdown(wait=True)
    print("✅ 所有图片与权重保存完毕！")


if __name__ == "__main__":
    main()
