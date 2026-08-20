import argparse
import csv
import glob
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ROOT / "outputs" / "bench_matrix"


SCRIPT_SPECS: Dict[str, Dict[str, Any]] = {
    "baseline": {"module": "baseline.test_baseline", "supports_mode": False},
    "v1": {"module": "baseline.test_baseline_v1", "supports_mode": False},
    "v2": {"module": "baseline.test_baseline_v2", "supports_mode": True},
    "v3": {"module": "baseline.test_baseline_v3_H2D_init", "supports_mode": True},
    "v4": {"module": "baseline.test_baseline_v4_fusesplit", "supports_mode": True},
    "v5": {"module": "baseline.test_baseline_v5_amp_graph", "supports_mode": True},
    "v6": {"module": "baseline.test_baseline_v6_full_batch", "supports_mode": True},
    "v7": {"module": "baseline.test_baseline_v7_fuselevelset", "supports_mode": True},
}

OOM_PATTERNS = [
    r"out of memory",
    r"cuda error:\s*out of memory",
    r"cublas.*alloc",
    r"std::bad_alloc",
]


@dataclass
class RunResult:
    version: str
    module: str
    mode: str
    tiles_target: int
    tiles_effective: Optional[int]
    multiplier: Optional[int]
    warmup: int
    repeat_idx: int
    return_code: int
    oom: bool
    graph_capture_attempted: bool
    graph_capture_success: bool
    elapsed_wall_s: float
    pure_pipeline_s: Optional[float]
    total_script_s: Optional[float]
    peak_vram_mb: Optional[float]
    micro_ms_per_tile: Optional[float]
    throughput_tiles_per_s: Optional[float]
    final_loss: Optional[float]
    grad_mean_last: Optional[float]
    grad_max_last: Optional[float]
    stdout_log: str
    stderr_log: str


def _parse_float(pattern: str, text: str) -> Optional[float]:
    m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _parse_last_float(pattern: str, text: str) -> Optional[float]:
    matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except Exception:
        return None


def _detect_oom(text: str) -> bool:
    lower = text.lower()
    return any(re.search(p, lower) for p in OOM_PATTERNS)


def _discover_base_tile_count(tiles_dir: str) -> int:
    files = glob.glob(os.path.join(tiles_dir, "*.oas"))
    if not files:
        raise FileNotFoundError(f"未在目录找到 .oas 文件: {tiles_dir}")
    return len(files)


def _safe_stdev(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def _run_subprocess(cmd: List[str], env: Dict[str, str], cwd: Path, timeout_s: int) -> Tuple[int, str, str, float]:
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        env=env,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    t1 = time.time()
    return proc.returncode, proc.stdout, proc.stderr, (t1 - t0)


def _to_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return "" if value is None else str(value)


def _build_python_module_cmd(module: str, mode: str, supports_mode: bool) -> List[str]:
    cmd = [sys.executable, "-m", module]
    if supports_mode:
        cmd.extend(["--mode", mode])
    return cmd


def _build_torch_profiler_cmd(module: str, mode: str, supports_mode: bool, trace_path: Path) -> List[str]:
    arg_mode = f"['bench', '--mode', '{mode}']" if supports_mode else "['bench']"
    code = (
        "import sys; "
        f"sys.argv={arg_mode}; "
        "import torch; "
        "import torch.profiler as tp; "
        f"import {module} as m; "
        "activities=[tp.ProfilerActivity.CPU]; "
        "if torch.cuda.is_available(): activities.append(tp.ProfilerActivity.CUDA); "
        "with tp.profile(activities=activities, record_shapes=True, profile_memory=True) as prof: "
        "    m.main(); "
        f"prof.export_chrome_trace(r'{str(trace_path)}')"
    )
    return [sys.executable, "-c", code]


def _wrap_nsys(cmd: List[str], report_prefix: Path) -> List[str]:
    return [
        "nsys",
        "profile",
        "-o",
        str(report_prefix),
        "--force-overwrite",
        "true",
        "--trace",
        "cuda,nvtx,osrt",
    ] + cmd


def _wrap_ncu(cmd: List[str], report_path: Path) -> List[str]:
    return [
        "ncu",
        "--set",
        "full",
        "--target-processes",
        "all",
        "--export",
        str(report_path),
    ] + cmd


def _archive_torch_extensions(dst_dir: Path) -> None:
    src = ROOT / ".torch_extensions"
    if not src.exists():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst_dir / "torch_extensions", dirs_exist_ok=True)


def _ensure_tool_available(tool_name: str) -> None:
    if shutil.which(tool_name) is None:
        raise FileNotFoundError(f"未找到工具 `{tool_name}`，请先安装并加入 PATH")


def _parse_metrics_from_logs(stdout_text: str, stderr_text: str, elapsed_wall_s: float, tiles_effective: Optional[int]) -> Dict[str, Any]:
    combined = stdout_text + "\n" + stderr_text

    pure_pipeline_s = _parse_float(r"流水线总纯耗时\s*\(剔除Eval\):\s*([0-9]+(?:\.[0-9]+)?)\s*s", combined)
    total_script_s = _parse_float(r"脚本端到端总耗时\s*\(含Eval\):\s*([0-9]+(?:\.[0-9]+)?)\s*s", combined)
    peak_vram_mb = _parse_float(r"峰值显存.*?:\s*([0-9]+(?:\.[0-9]+)?)\s*MB", combined)
    micro_ms = _parse_float(r"Micro-Bench\):\s*([0-9]+(?:\.[0-9]+)?)\s*ms\s*/\s*Tile", combined)

    if pure_pipeline_s is not None and tiles_effective and pure_pipeline_s > 0:
        throughput = tiles_effective / pure_pipeline_s
    else:
        throughput = None

    final_loss = _parse_last_float(r"avg_loss=([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)", combined)

    grad_pairs = re.findall(
        r"grad_mean=([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?),\s*grad_max=([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)",
        combined,
    )
    grad_mean_last = float(grad_pairs[-1][0]) if grad_pairs else None
    grad_max_last = float(grad_pairs[-1][1]) if grad_pairs else None

    graph_attempt = bool(re.search(r"捕获计算图|cuda graph|graph", combined, flags=re.IGNORECASE))
    graph_success = bool(re.search(r"极速回放模式启动|replay", combined, flags=re.IGNORECASE))

    tiles_effective_from_log = _parse_last_float(r"强制克隆为\s*([0-9]+)\s*个\s*Tiles", combined)
    if tiles_effective is None and tiles_effective_from_log is not None:
        tiles_effective = int(tiles_effective_from_log)

    return {
        "pure_pipeline_s": pure_pipeline_s,
        "total_script_s": total_script_s if total_script_s is not None else elapsed_wall_s,
        "peak_vram_mb": peak_vram_mb,
        "micro_ms_per_tile": micro_ms,
        "throughput_tiles_per_s": throughput,
        "final_loss": final_loss,
        "grad_mean_last": grad_mean_last,
        "grad_max_last": grad_max_last,
        "graph_capture_attempted": graph_attempt,
        "graph_capture_success": graph_success,
        "tiles_effective": tiles_effective,
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _write_markdown_table(path: Path, rows: List[Dict[str, Any]], max_rows: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("No records.\n", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    take = rows[:max_rows]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in take:
        lines.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
    if len(rows) > max_rows:
        lines.append(f"\n> 仅展示前 {max_rows} 行，完整数据见 CSV。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _log(msg: str, log_file: Path) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _group_key(r: RunResult) -> Tuple[str, str, int, int]:
    return (r.version, r.mode, r.tiles_target, r.warmup)


def _aggregate_runs(
    runs: List[RunResult],
    loss_rtol: float,
    grad_rtol: float,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, int, int], List[RunResult]] = {}
    for r in runs:
        grouped.setdefault(_group_key(r), []).append(r)

    baseline_ref: Dict[Tuple[str, int, int], Dict[str, Optional[float]]] = {}
    for (version, mode, tiles_target, warmup), items in grouped.items():
        if version != "baseline":
            continue
        ok_items = [x for x in items if x.return_code == 0 and not x.oom]
        if not ok_items:
            continue
        baseline_ref[(mode, tiles_target, warmup)] = {
            "loss": _mean_opt([x.final_loss for x in ok_items]),
            "grad_mean": _mean_opt([x.grad_mean_last for x in ok_items]),
            "grad_max": _mean_opt([x.grad_max_last for x in ok_items]),
        }

    rows: List[Dict[str, Any]] = []
    for (version, mode, tiles_target, warmup), items in sorted(grouped.items()):
        elapsed = [x.elapsed_wall_s for x in items]
        pure = [x.pure_pipeline_s for x in items if x.pure_pipeline_s is not None]
        total = [x.total_script_s for x in items if x.total_script_s is not None]
        peak = [x.peak_vram_mb for x in items if x.peak_vram_mb is not None]
        micro = [x.micro_ms_per_tile for x in items if x.micro_ms_per_tile is not None]
        thr = [x.throughput_tiles_per_s for x in items if x.throughput_tiles_per_s is not None]
        loss = [x.final_loss for x in items if x.final_loss is not None]
        grad_mean = [x.grad_mean_last for x in items if x.grad_mean_last is not None]
        grad_max = [x.grad_max_last for x in items if x.grad_max_last is not None]

        first_second_delta = None
        if len(items) >= 2:
            first_second_delta = items[0].elapsed_wall_s - items[1].elapsed_wall_s

        ref = baseline_ref.get((mode, tiles_target, warmup))
        loss_rel_err = None
        grad_mean_rel_err = None
        grad_max_rel_err = None
        within_tol = None

        mean_loss = _mean_opt(loss)
        mean_grad_mean = _mean_opt(grad_mean)
        mean_grad_max = _mean_opt(grad_max)

        if ref is not None:
            loss_rel_err = _rel_err(mean_loss, ref.get("loss"))
            grad_mean_rel_err = _rel_err(mean_grad_mean, ref.get("grad_mean"))
            grad_max_rel_err = _rel_err(mean_grad_max, ref.get("grad_max"))
            checks = []
            if loss_rel_err is not None:
                checks.append(loss_rel_err <= loss_rtol)
            if grad_mean_rel_err is not None:
                checks.append(grad_mean_rel_err <= grad_rtol)
            if grad_max_rel_err is not None:
                checks.append(grad_max_rel_err <= grad_rtol)
            within_tol = all(checks) if checks else None

        attempts = sum(1 for x in items if x.graph_capture_attempted)
        succ = sum(1 for x in items if x.graph_capture_success)
        graph_succ_rate = (succ / attempts) if attempts > 0 else None

        rows.append(
            {
                "version": version,
                "mode": mode,
                "tiles_target": tiles_target,
                "warmup": warmup,
                "runs": len(items),
                "success_runs": sum(1 for x in items if x.return_code == 0 and not x.oom),
                "oom_runs": sum(1 for x in items if x.oom),
                "mean_wall_s": round(statistics.mean(elapsed), 6),
                "std_wall_s": round(_safe_stdev(elapsed), 6),
                "mean_pure_s": _round_opt(_mean_opt(pure)),
                "std_pure_s": _round_opt(_safe_stdev(pure) if pure else None),
                "mean_total_script_s": _round_opt(_mean_opt(total)),
                "mean_peak_vram_mb": _round_opt(_mean_opt(peak)),
                "mean_micro_ms_per_tile": _round_opt(_mean_opt(micro)),
                "mean_throughput_tiles_per_s": _round_opt(_mean_opt(thr)),
                "mean_final_loss": _round_opt(mean_loss),
                "mean_grad_mean": _round_opt(mean_grad_mean),
                "mean_grad_max": _round_opt(mean_grad_max),
                "loss_rel_err_vs_baseline": _round_opt(loss_rel_err),
                "grad_mean_rel_err_vs_baseline": _round_opt(grad_mean_rel_err),
                "grad_max_rel_err_vs_baseline": _round_opt(grad_max_rel_err),
                "within_tolerance": within_tol,
                "first_minus_second_wall_s": _round_opt(first_second_delta),
                "graph_capture_success_rate": _round_opt(graph_succ_rate),
            }
        )

    return rows


def _write_conclusion_markdown(path: Path, report: Dict[str, Any], summary_rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    total = int(report.get("runs_total", 0))
    succ = int(report.get("runs_success", 0))
    oom = int(report.get("runs_oom", 0))
    graph_attempts = int(report.get("graph_capture_attempts", 0))
    graph_succ = int(report.get("graph_capture_successes", 0))
    graph_rate = (graph_succ / graph_attempts) if graph_attempts > 0 else None

    valid_rows = [r for r in summary_rows if r.get("mean_throughput_tiles_per_s") is not None]
    fastest = max(valid_rows, key=lambda r: float(r["mean_throughput_tiles_per_s"])) if valid_rows else None

    tol_rows = [r for r in summary_rows if r.get("within_tolerance") is not None]
    tol_pass = sum(1 for r in tol_rows if bool(r.get("within_tolerance")))
    tol_fail = sum(1 for r in tol_rows if r.get("within_tolerance") is False)

    lines: List[str] = []
    lines.append("# Benchmark 结论摘要")
    lines.append("")
    lines.append("## 总体结论")
    lines.append(f"- 总运行数: {total}")
    lines.append(f"- 成功运行: {succ}")
    lines.append(f"- OOM 运行: {oom}")
    if graph_rate is not None:
        lines.append(f"- Graph Capture 成功率: {graph_succ}/{graph_attempts} = {graph_rate:.2%}")
    else:
        lines.append("- Graph Capture 成功率: 无可用样本")
    lines.append("")

    lines.append("## 误差一致性")
    lines.append(f"- 判定样本数: {len(tol_rows)}")
    lines.append(f"- 通过误差阈值: {tol_pass}")
    lines.append(f"- 超出误差阈值: {tol_fail}")
    lines.append("")

    lines.append("## 性能最佳样本")
    if fastest is None:
        lines.append("- 无可用吞吐数据")
    else:
        lines.append(
            "- 最快配置: "
            f"version={fastest.get('version')}, mode={fastest.get('mode')}, "
            f"tiles={fastest.get('tiles_target')}, warmup={fastest.get('warmup')}"
        )
        lines.append(f"- 平均吞吐: {fastest.get('mean_throughput_tiles_per_s')} tiles/s")
        lines.append(f"- 平均纯流水线耗时: {fastest.get('mean_pure_s')} s")
    lines.append("")

    lines.append("## 产物位置")
    lines.append("- 原始逐次记录: raw_runs.csv")
    lines.append("- 聚合统计: summary.csv")
    lines.append("- 汇总表格: summary.md")
    lines.append("- 全局报告: report.json")
    lines.append("- 运行总日志: runner.log")
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mean_opt(values: Sequence[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if v is not None]
    return statistics.mean(valid) if valid else None


def _rel_err(v: Optional[float], ref: Optional[float]) -> Optional[float]:
    if v is None or ref is None:
        return None
    if abs(ref) < 1e-12:
        return abs(v - ref)
    return abs(v - ref) / abs(ref)


def _round_opt(v: Optional[float], ndigits: int = 6) -> Optional[float]:
    return round(v, ndigits) if v is not None else None


def _validate_tiles(total: int) -> None:
    g = int(round(math.sqrt(total)))
    if g * g != total:
        raise ValueError(f"目标 tiles={total} 不是完全平方数")
    if g <= 0 or (g & (g - 1)) != 0:
        raise ValueError(f"目标 grid={g} 不是 2 的幂")


def _run_kernel_microbench(out_dir: Path, reps: int = 100) -> Dict[str, Any]:
    import torch

    out: Dict[str, Any] = {}
    if not torch.cuda.is_available():
        out["skipped"] = "CUDA not available"
        return out

    device = torch.device("cuda")
    h, w = 1024, 1024
    k = 24

    a = torch.randn((h, w), device=device, dtype=torch.float32)
    t = torch.rand((h, w), device=device, dtype=torch.float32)
    kernels = torch.randn((k, h, w), device=device, dtype=torch.complex64)
    scales = torch.rand((k,), device=device, dtype=torch.float32)

    def time_cuda(fn) -> float:
        fn()  # warmup: JIT compile (Triton), allocator, etc. stay OUT of the timed region
        torch.cuda.synchronize()
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream(device=device)
        starter.record(stream)
        for _ in range(reps):
            fn()
        ender.record(stream)
        torch.cuda.synchronize()
        return float(starter.elapsed_time(ender) / reps)

    def bench_fft2() -> None:
        _ = torch.fft.fft2(a.to(torch.complex64), norm="forward")

    def bench_ifft2_accum() -> None:
        af = torch.fft.fft2(a.to(torch.complex64), norm="forward")
        out_acc = torch.zeros((h, w), device=device, dtype=torch.float32)
        for i in range(k):
            conv = torch.fft.ifft2(af * kernels[i], norm="forward").real
            out_acc.add_(conv, alpha=float(scales[i].item()))

    def bench_pointwise_torch() -> None:
        printed = torch.sigmoid((a - 0.5) * 1.0)
        diff = printed - t
        grad = 2.0 * diff * (printed * (1.0 - printed))
        _ = grad

    out["fft2_ms"] = time_cuda(bench_fft2)
    out["ifft2_accum_ms"] = time_cuda(bench_ifft2_accum)
    out["pointwise_torch_ms"] = time_cuda(bench_pointwise_torch)

    try:
        # NOTE: module is `triton_fused_pointwise`, not `fused_pointwise_triton`.
        from fuilt.ilt.func.triton_fused_pointwise import fused_pointwise_forward_triton as triton_pointwise

        def bench_pointwise_triton() -> None:
            triton_pointwise(a, t, target_density=0.5, print_steepness=1.0)

        out["pointwise_triton_ms"] = time_cuda(bench_pointwise_triton)
    except Exception as e:
        out["pointwise_triton_ms"] = None
        out["pointwise_triton_error"] = str(e)
        print(f"[WARN] triton pointwise micro-bench unavailable: {e}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kernel_microbench.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline/v1..v7 benchmark matrix and summarize results.")
    parser.add_argument("--tiles-dir", type=str, default=os.environ.get("FUILT_TILES_DIR", ""))
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--tile-targets", type=str, default="16,64,256,512,1024")
    parser.add_argument("--warmups", type=str, default="0,10")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--modes", type=str, default="pytorch_fft,cuda_op")
    parser.add_argument("--timeout-s", type=int, default=7200)
    parser.add_argument("--loss-rtol", type=float, default=1e-3)
    parser.add_argument("--grad-rtol", type=float, default=1e-2)
    parser.add_argument("--clean-extensions-first", action="store_true")
    parser.add_argument("--archive-torch-extensions", action="store_true")
    parser.add_argument("--enable-nsys", action="store_true")
    parser.add_argument("--enable-ncu", action="store_true")
    parser.add_argument("--enable-torch-profiler", action="store_true")
    parser.add_argument("--run-kernel-microbench", action="store_true")
    parser.add_argument("--versions", type=str, default="baseline,v1,v2,v3,v4,v5,v6,v7")
    parser.add_argument("--out-dir", type=str, default="")
    args = parser.parse_args()

    if args.enable_nsys and args.enable_ncu:
        raise ValueError("`--enable-nsys` 与 `--enable-ncu` 不能同时开启，请分两次跑。")

    if args.enable_nsys:
        _ensure_tool_available("nsys")
    if args.enable_ncu:
        _ensure_tool_available("ncu")

    if not args.tiles_dir:
        raise ValueError("请通过 --tiles-dir 或环境变量 FUILT_TILES_DIR 指定 .oas 目录")

    out_dir = Path(args.out_dir) if args.out_dir else (OUTPUT_ROOT / time.strftime("%Y%m%d_%H%M%S"))
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    runner_log = out_dir / "runner.log"
    _log(f"output_dir={out_dir}", runner_log)

    base_tiles = _discover_base_tile_count(args.tiles_dir)
    tile_targets = [int(x.strip()) for x in args.tile_targets.split(",") if x.strip()]
    warmups = [int(x.strip()) for x in args.warmups.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    versions = [x.strip() for x in args.versions.split(",") if x.strip()]

    for t in tile_targets:
        _validate_tiles(t)

    runs: List[RunResult] = []

    cleaned_once: Dict[str, bool] = {}

    for version in versions:
        if version not in SCRIPT_SPECS:
            print(f"[WARN] 未知版本标签，跳过: {version}")
            continue
        spec = SCRIPT_SPECS[version]
        module = str(spec["module"])
        supports_mode = bool(spec["supports_mode"])
        version_modes = modes if supports_mode else ["na"]

        for mode in version_modes:
            for target_tiles in tile_targets:
                if target_tiles % base_tiles != 0:
                    print(f"[SKIP] {version}/{mode} target={target_tiles}: 不能整除 base_tiles={base_tiles}")
                    continue
                multiplier = target_tiles // base_tiles

                for warmup in warmups:
                    for repeat_idx in range(args.repeats):
                        env = os.environ.copy()
                        env["FUILT_TILES_DIR"] = args.tiles_dir
                        env["FUILT_ITERS"] = str(args.iters)
                        env["FUILT_EVAL_INTERVAL"] = str(args.eval_interval)
                        env["FUILT_LR"] = str(args.lr)
                        env["FUILT_TILE_MULTIPLIER"] = str(multiplier)
                        env["FUILT_BENCH_WARMUP"] = str(warmup)

                        if args.clean_extensions_first and not cleaned_once.get(version):
                            ext_dir = ROOT / ".torch_extensions"
                            if ext_dir.exists():
                                shutil.rmtree(ext_dir)
                            cleaned_once[version] = True

                        run_tag = f"{version}_{mode}_tiles{target_tiles}_wu{warmup}_r{repeat_idx+1}"
                        stdout_log = logs_dir / f"{run_tag}.stdout.log"
                        stderr_log = logs_dir / f"{run_tag}.stderr.log"

                        if args.enable_torch_profiler:
                            trace_path = logs_dir / f"{run_tag}.torch_trace.json"
                            cmd = _build_torch_profiler_cmd(module, mode, supports_mode, trace_path)
                        else:
                            cmd = _build_python_module_cmd(module, mode, supports_mode)

                        if args.enable_nsys:
                            cmd = _wrap_nsys(cmd, logs_dir / f"{run_tag}.nsys")
                        if args.enable_ncu:
                            cmd = _wrap_ncu(cmd, logs_dir / f"{run_tag}.ncu")

                        print(
                            f"[RUN] version={version} mode={mode} tiles={target_tiles} "
                            f"warmup={warmup} repeat={repeat_idx+1}/{args.repeats}"
                        )
                        _log(
                            f"RUN version={version} mode={mode} tiles={target_tiles} warmup={warmup} repeat={repeat_idx+1}/{args.repeats}",
                            runner_log,
                        )

                        try:
                            code, out, err, elapsed = _run_subprocess(
                                cmd=cmd,
                                env=env,
                                cwd=ROOT,
                                timeout_s=args.timeout_s,
                            )
                        except subprocess.TimeoutExpired as e:
                            code = 124
                            out = _to_text(e.stdout)
                            err = _to_text(e.stderr) + "\n[TIMEOUT]"
                            elapsed = float(args.timeout_s)

                        out = _to_text(out)
                        err = _to_text(err)

                        stdout_log.write_text(out, encoding="utf-8", errors="ignore")
                        stderr_log.write_text(err, encoding="utf-8", errors="ignore")

                        metrics = _parse_metrics_from_logs(
                            stdout_text=out,
                            stderr_text=err,
                            elapsed_wall_s=elapsed,
                            tiles_effective=target_tiles,
                        )

                        oom = _detect_oom(out + "\n" + err)

                        run = RunResult(
                            version=version,
                            module=module,
                            mode=mode,
                            tiles_target=target_tiles,
                            tiles_effective=metrics.get("tiles_effective", target_tiles),
                            multiplier=multiplier,
                            warmup=warmup,
                            repeat_idx=repeat_idx + 1,
                            return_code=code,
                            oom=oom,
                            graph_capture_attempted=bool(metrics.get("graph_capture_attempted", False)),
                            graph_capture_success=bool(metrics.get("graph_capture_success", False)),
                            elapsed_wall_s=elapsed,
                            pure_pipeline_s=metrics.get("pure_pipeline_s"),
                            total_script_s=metrics.get("total_script_s"),
                            peak_vram_mb=metrics.get("peak_vram_mb"),
                            micro_ms_per_tile=metrics.get("micro_ms_per_tile"),
                            throughput_tiles_per_s=metrics.get("throughput_tiles_per_s"),
                            final_loss=metrics.get("final_loss"),
                            grad_mean_last=metrics.get("grad_mean_last"),
                            grad_max_last=metrics.get("grad_max_last"),
                            stdout_log=str(stdout_log),
                            stderr_log=str(stderr_log),
                        )
                        runs.append(run)

                        _log(
                            f"DONE run_tag={run_tag} rc={code} oom={oom} wall_s={elapsed:.3f} "
                            f"loss={metrics.get('final_loss')} peak_vram_mb={metrics.get('peak_vram_mb')}",
                            runner_log,
                        )

                        if args.archive_torch_extensions and repeat_idx == 0:
                            _archive_torch_extensions(out_dir / "artifacts" / run_tag)

    raw_rows = [asdict(r) for r in runs]
    summary_rows = _aggregate_runs(runs, loss_rtol=args.loss_rtol, grad_rtol=args.grad_rtol)

    _write_csv(out_dir / "raw_runs.csv", raw_rows)
    _write_csv(out_dir / "summary.csv", summary_rows)
    _write_markdown_table(out_dir / "summary.md", summary_rows)

    report = {
        "output_dir": str(out_dir),
        "base_tiles": base_tiles,
        "tile_targets": tile_targets,
        "warmups": warmups,
        "modes": modes,
        "versions": versions,
        "runs_total": len(runs),
        "runs_success": sum(1 for r in runs if r.return_code == 0 and not r.oom),
        "runs_oom": sum(1 for r in runs if r.oom),
        "graph_capture_attempts": sum(1 for r in runs if r.graph_capture_attempted),
        "graph_capture_successes": sum(1 for r in runs if r.graph_capture_success),
    }

    if args.run_kernel_microbench:
        report["kernel_microbench"] = _run_kernel_microbench(out_dir / "kernel_microbench")

    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_conclusion_markdown(out_dir / "conclusion.md", report, summary_rows)
    _log("wrote raw_runs.csv/summary.csv/summary.md/report.json/conclusion.md", runner_log)

    print("\n=== Benchmark Matrix Done ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n汇总表: {out_dir / 'summary.md'}")
    print(f"结论页: {out_dir / 'conclusion.md'}")
    print(f"原始记录: {out_dir / 'raw_runs.csv'}")
    print(f"统计记录: {out_dir / 'summary.csv'}")
    print(f"总日志: {runner_log}")


if __name__ == "__main__":
    main()
'''


nohup python -m baseline.run_benchmark_matrix \
  --tiles-dir /data/lyj/FuILT/tiles_from_patch \
  --tile-targets 16,64 \
  --warmups 10 \
  --repeats 1 \
  --enable-nsys \
  --out-dir outputs/bench_matrix/nsys_run \
  > outputs/bench_matrix/nsys_run/console.log 2>&1 &

  nohup python -m baseline.run_benchmark_matrix \
  --tiles-dir /data/lyj/FuILT/tiles_from_patch \
  --tile-targets 16,64 \
  --warmups 10 \
  --repeats 1 \
  --enable-ncu \
  --out-dir outputs/bench_matrix/ncu_run \
  > outputs/bench_matrix/ncu_run/console.log 2>&1 &

nohup python -m baseline.run_benchmark_matrix \
  --tiles-dir /data/lyj/FuILT/tiles_from_patch \
  --tile-targets 16,64 \
  --warmups 10 \
  --repeats 1 \
  --enable-torch-profiler \
  --out-dir outputs/bench_matrix/torchprof_run \
  > outputs/bench_matrix/torchprof_run/console.log 2>&1 &

'''