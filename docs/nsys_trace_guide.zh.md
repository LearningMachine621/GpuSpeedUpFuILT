# Nsight Systems Trace 使用指南

怎么生成、打开、分析项目的 `v7_p013_nvtx.nsys-rep` trace 文件。覆盖三种
访问方式（GUI / 命令行 stats / 无显示器的下载方式）+ 解释 NVTX stage 标注。

---

## 为什么要这个 trace

项目 README 里每个性能声明都有两种背书：

1. **v7 自报告的 `time.time()` 表** —— 快，但是应用自己测自己，没有第三方
   验证。
2. **Nsight Systems trace** —— NVIDIA 官方 profiler；捕获每个 CUDA kernel
   launch、NVTX range、CUDA runtime API call。这是 production-grade 的证据。

trace 文件（`outputs/nsys_traces/v7_p013_nvtx.nsys-rep`，~11 MB）是"亮出工作"
的 artifact。任何人都能复现或下载来验证数字。

---

## 1. 生成 trace

### 一键复现

```bash
CUDA_VISIBLE_DEVICES=<isolated GPU> \
FUILT_TILES_DIR=/data/lyj/FuILT/tiles_from_patch \
FUILT_ITERS=5 FUILT_BENCH_WARMUP=2 FUILT_TILE_MULTIPLIER=16 \
FUILT_USE_COMBINED_KERNEL=1 FUILT_USE_ASYNC_D2H=1 FUILT_USE_NVTX=1 \
nsys profile -t cuda,nvtx,osrt --stats=false \
  -o outputs/nsys_traces/v7_p013_nvtx \
  python -m baseline.test_baseline_v7_fuselevelset --mode pytorch_fft
```

### 关键 env var

| Var | 作用 |
|---|---|
| `FUILT_USE_NVTX=1` | 把每个 pipeline stage 用 `torch.cuda.nvtx.range_push/pop` 包起来，让 timeline 显示彩色 band。 |
| `FUILT_USE_COMBINED_KERNEL=1` | 启用 P0.1（IFFT 线性性重写，3.10× 加速）。 |
| `FUILT_USE_ASYNC_D2H=1` | 启用 P0.3（pinned + non_blocking + overlap，1.20× 加速）。 |
| `FUILT_ITERS=5` | trace 用 5 iter（vs benchmark 20 iter）让文件大小 ~10 MB。 |
| `FUILT_TILE_MULTIPLIER=16` | 总 256 tiles（16 真实 × 16 克隆）。设 64 测 1024-tile batch scaling。 |

### 为什么 `--stats=false`？

`nsys profile` 抓完 trace 后能自动跑 reports。我们禁用，原因：
- 输出冗长且打到 stdout，不进文件
- 我们想用 `nsys stats --report <name>` 灵活选 report

---

## 2. 打开 trace

### 方案 A：X11 forwarding GUI（能用，但慢）

如果 SSH 客户端支持 X11 forwarding：

```bash
# 本地机器：
ssh -X lyj@<server>  # 或 -Y for trusted forwarding
sudo apt install libxcb-cursor0  # 如果报 xcb plugin 错误
nsys-ui outputs/nsys_traces/v7_p013_nvtx.nsys-rep
```

**注意**：nsys-ui 是 OpenGL app。X11 forwarding 下每帧像素都走网络——预期
个位数 FPS。能检查，但导航很痛苦。

### 方案 B：下载到本地（最佳体验）

```bash
# 本地机器：
scp lyj@<server>:/home/lyj/code/GpuSpeedUpFuILT/outputs/nsys_traces/v7_p013_nvtx.nsys-rep ./

# 本地装 Nsight Systems（免费）：
#   macOS:   brew install --cask cuda
#   Windows: https://developer.nvidia.com/nsight-systems
#   Linux:   和服务器同包

nsys-ui v7_p013_nvtx.nsys-rep
```

11 MB 几秒钟传完；本地 GUI 完全流畅。

### 方案 C：纯命令行 stats（无需 GUI）

这是 README 数字用的方式。三个有用的 reports：

```bash
# 按 GPU 时间排名的 top kernels
nsys stats --report cuda_gpu_kern_sum outputs/nsys_traces/v7_p013_nvtx.nsys-rep

# NVTX range 树（per-stage 时间，我们标注的）
nsys stats --report nvtx_pushpop_trace outputs/nsys_traces/v7_p013_nvtx.nsys-rep

# CUDA runtime API 摘要（cudaMemcpy、cudaLaunchKernel 等）
nsys stats --report cuda_api_sum outputs/nsys_traces/v7_p013_nvtx.nsys-rep
```

---

## 3. trace 里有什么

### NVTX stage 标注

trace 标注了 7 个 stage ranges（在 nsys-ui 里显示为彩色 band）：

| Stage | 彩色 band | 覆盖范围 |
|---|---|---|
| `IO_Raster` | (外层) | OAS 文件 → numpy mask 读 |
| `H2D_Init` | (外层) | CPU SDF 计算 + 256× H2D + kernel_fft 构建 + JIT 编译 |
| `iter_NNN` | (per-iter 外层) | 一次优化迭代 |
| ↳ `LevelSet` | (per-iter 内层) | FFT forward + pointwise grad + loss |
| ↳ `Fuse` | (per-iter 内层) | per-tile grad → canvas 累加 |
| ↳ `Split` | (per-iter 内层) | canvas → per-tile 下一 iter 初始值 |
| ↳ `Apply` | (per-iter 内层) | `params -= lr * grad` |
| `D2H_Save` | (外层) | 最终结果 → CPU + ThreadPool 写盘 |

### Stage 时间（5-iter trace）

| Stage | Mean (ms) | Std (ms) | 备注 |
|---|---:|---:|---|
| IO_Raster | 177.34 | — | 一次性 |
| H2D_Init | 6129.76 | — | 一次性，被 first-call JIT 编译 `fused_pointwise_ext` 主导 |
| LevelSet | 52.98 | 13.32 | per-iter；std 被 iter_000 冷启动放大 |
| Fuse | 3.10 | 0.06 | per-iter；极度稳定 |
| Split | 3.60 | 0.02 | per-iter；极度稳定 |
| Apply | 4.96 | 0.10 | per-iter；极度稳定 |
| D2H_Save | 741.17 | — | 一次性 |

> **方法论注记**：trace 时的数字和 headline benchmark 数字
> （v7+P0.1+P0.3 = 2.34s）不同，原因：
> 1. trace 用 5 iter，不是 20
> 2. First-call JIT 编译在 trace 模式下算到 H2D_Init
> 3. NVTX recording 每条 event 有少量 overhead
>
> 用 trace 做 **stage 占比分析**，用隔离的 20-iter benchmark 做
> **绝对 latency 声明**。

### Top GPU kernels（来自 `cuda_gpu_kern_sum`）

完整 kernel breakdown 见主 README 的"GPU kernel breakdown"章节。摘要：
FFT chain 占主导 ~50%，我们优化的 `fused_pointwise_grad_loss` kernel 占 0.3%。

---

## 4. 常见坑

### "qt.qpa.xcb: could not connect to display"

你在无显示器的 SSH session。用以下之一：
- `ssh -X`（X11 forwarding）
- 下载 trace 到本地机器
- 用 `nsys stats` 命令行（方案 C）

### "libxcb-cursor0 needed"

```bash
sudo apt install libxcb-cursor0
```

Qt 6.5+（nsys-ui 的 GUI 框架）需要。

### "OpenGL version is too low"

服务器没有 GPU 加速 OpenGL。Mesa 软件回退能用但慢。要真正 OpenGL，用带
独立 GPU 的本地机器。

### trace 文件太大

- 5-iter trace：~11 MB
- 20-iter trace：~40 MB
- 1024-tile × 20-iter：~200 MB

如果文件大小敏感（比如 CI），减少 `FUILT_ITERS` 或用 `--cuda-um-profiling off`。

### NVTX ranges 缺失

确认：
- `FUILT_USE_NVTX=1` 设了
- `nsys profile -t cuda,nvtx,osrt` 的 `-t` 列表包含 `nvtx`
- Python 进程实际执行了标注的代码路径（用 `--mode pytorch_fft`，不要用
  `cuda_op`，那是另一条路径）

---

## 5. 复现 checklist

从零验证这个 trace：

```bash
# 1. 确认环境
nvidia-smi  # GPU 可用
python -c "import torch; print(torch.__version__)"  # 2.11+
python -c "import triton; print(triton.__version__)"  # 3.6+

# 2. 生成 trace（上面命令）
# 3. 验证 NVTX 抓到了
nsys stats --report nvtx_pushpop_trace outputs/nsys_traces/v7_p013_nvtx.nsys-rep \
  | grep -c ":LevelSet"   # 应该 > 0
# 4. 验证 kernel breakdown
nsys stats --report cuda_gpu_kern_sum outputs/nsys_traces/v7_p013_nvtx.nsys-rep \
  | head -20
```

如果步骤 3 返回 0，说明 NVTX env var 没设或 `-t nvtx` flag 缺失。
