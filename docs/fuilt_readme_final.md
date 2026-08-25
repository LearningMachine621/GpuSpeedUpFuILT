# FuILT — 计算光刻 GPU 流水线优化：53s → 2.36s（22.5×）

> **一句话**：对一个"固定计算图 + 大 batch + 延迟敏感 GPU 路径 + 异步 IO"的计算光刻反演流水线，
> 用 profile 驱动的三层手段（算法级等价重写 / 数据流编排 / kernel 融合）把端到端延迟压缩 **22.5×**，
> 其中最大单笔收益来自一条数学等式（IFFT 线性性），零新 kernel、数学严格等价（实测 fp32 max_abs_diff ≤ 3.7e-9）。
>
> 完整研究叙事：[`optimization_journey.md`](optimization_journey.md) ·
> 证据台账：[`verifiable_results.md`](verifiable_results.md) ·
> 瓶颈迁移图：[`bottleneck_shift.md`](bottleneck_shift.md) ·
> 面试讲法：[`gpu4fuilt_interview.md`](gpu4fuilt_interview.md)

---

## 1. 问题

**计算光刻反演（ILT）**：给定芯片版图，在 GPU 上迭代求最优光刻掩模。每个"请求"是一个 2D tile，
一块芯片是一个大 tile batch，穿过固定计算图：前向 FFT 仿真 → 融合梯度 kernel → tile fuse/split →
参数更新 → D2H 保存。

- 负载：256 tiles × 20 iters，1024×1024 tile，RTX 4090 24GB，PyTorch FFT 基线
- v1 基线端到端 **53.07 s**——直觉是"kernel 算得慢"，profile 证明是**GPU 在等数据**
  （256× H2D 占 H2D_Init 阶段 85%），真正的瓶颈在数据编排与算法表达
- 这个形态与 **LLM serving 同构**（固定图 / batch 输入 / 延迟敏感 GPU 路径 / 异步 IO）——
  本仓库以 serving 工程的方法对待它：每个优化都有 profile、消融和 committed 证据

## 2. 贡献

| # | 贡献 | 结果 |
|---|---|---|
| 1 | **算法级等价重写（P0.1）**：用 IFFT 线性性把每 tile 每 iter 的 24 次 `ifft2` 合并成 1 次（`Σ s·IFFT(Xₖ) = IFFT(Σ s·Xₖ)`，合并核在 init 预计算一次、常驻显存），数学严格等价、零新 kernel | 端到端 **3.20×**，热点阶段 10.5× |
| 2 | **数据流编排**：pinned host memory + async + stream 把 H2D/D2H 藏进计算（DMA 与 SM 是不同引擎；async 的前提是 pinned） | H2D overlap **3.97×**；async D2H 再 1.19× |
| 3 | **融合 kernel 与生产选型**：Triton 融合 pointwise（实测 10 算子 eager 链 → 1 kernel，`pointwise_kernel_count.json`）、v8 bigbuf 地址合并（地址可推导 → 合并访问 + graph friendly）；以"端到端 +2.5%、默认 CUDA-Graph-safe、少 91 行 CUDA 样板"选定 Triton 为生产路径 | kernel 延迟 **2.1×**，fuse4 **3.9×** |
| 4 | **CUDA Graph 捕获 bug 定位与修复**：手写 kernel 裸 `<<<>>>` 走默认流，capture 时被静默跳过（**错而不报错**）；4 处改 `getCurrentCUDAStream()` 后用 5-mode 可控变量实验证明因果，graph capture 回填主流水线 | mode3 loss-rel **1.02 FAIL** → mode5 **0.00 PASS** |
| 5 | **证据工程**：头条数字全表重跑落盘（3 reps）、删除不可复现声明、修正 harness 两个测量 bug；每个数字 = committed 脚本 + raw JSON + 一条命令 | 22.5× 可复现 |

## 3. 核心数字

**端到端（256 tiles × 20 iters，RTX 4090 独占，3 reps mean ± std，2026-08-04 重跑）：**

| 版本 | Pure pipeline | vs v1 |
|---|---:|---:|
| v1 朴素逐 tile eager | 53.07 ± 0.51 s | 1.00× |
| v3 H2D + pinned overlap | 13.36 ± 0.05 s | 3.97× |
| v7 融合 pointwise（Triton） | 9.18 ± 0.03 s | 5.78× |
| v7 + P0.1（IFFT 线性性） | 2.80 ± 0.08 s | 19.0× |
| **v7 + P0.1 + P0.3（async D2H）** | **2.36 ± 0.05 s** | **22.5×** |

**关键归因与 kernel 级证据：**

- 瓶颈迁移链：H2D 占 init 85% → launch 税（trace 段 ~22.9 万次 kernel launch / 1.10 s CPU API）→
  FFT 链 50.0% GPU 时间 → IFFT 循环（1.27 ms/tile-iter，= 前向 fft2 的 43×）
- P0.1 后 trace 复核：复数乘 12.3% → 0.4%，FFT 链 50.0% → 33.9%
- fused pointwise：49.2 → 23.4 µs（2.1×），DRAM 流量按形状推算 96 → 16 MiB/tile（6×↓，无 NCU 实测）
- fuse4（v8 bigbuf，4096² canvas）：eager 179.8 → Triton 46.0 µs（3.9×）；手写 CUDA 29.2 µs 为上限参考
- 扩展性：1024 tiles 8.31 s，per-tile 延迟 flat（+1%），峰值 VRAM 23337 MB（24GB 的 95%）

## 4. 架构（最终形态：v7 + P0.1 + P0.3）

```text
              ┌───────────────┐  pinned, overlapped
 OAS files ──▶│  rasterize +  │── tiles (H2D) ───┐
              │  tile partition│                  ▼
              └───────────────┘         ┌──────────────────┐
                                        │  LithoSim forward │  1× fft2 + 1× ifft2
                                        │  (P0.1 combined   │  （24 核频域预合并，
                                        │   kernel FFT)     │    init 一次常驻显存）
                                        └──────────┬───────┘
                                                   │ aerial image
                                                   ▼
                                        ┌──────────────────┐
                                        │ fused_pointwise  │  ← Triton kernel
                                        │ _grad_loss       │   （10 算子 → 1，实测）
                                        └──────────┬───────┘
                                                   │ grad + diff²
                                 ┌─────────────────┴─────────────────┐
                                 ▼                                   ▼
                      ┌────────────────────┐              ┌────────────────────┐
                      │  flat fuse/split   │              │  Level-Set update  │
                      │  (bigbuf, <1%)     │              │  (in-place, fused  │
                      │                    │              │   w/ pointwise)    │
                      └────────────────────┘              └────────────────────┘
                                 │
                                 ▼
                         ┌────────────────┐  async D2H + pinned
                         │  ThreadPool +  │──▶ PNG/numpy to disk
                         │  pinned D2H    │   （P0.3: 与 fuse 重叠）
                         └────────────────┘
```

## 5. 复现

环境：RTX 4090 24GB · torch 2.11.0+cu130 · Triton 3.6.0 · CUDA 13.0 · Python 3.13
（8 卡主机上请用 `CUDA_VISIBLE_DEVICES` 独占一张空闲卡——争抢会让数字差 5-15×）

```bash
# 0. 安装
pip install -e .
export FUILT_TILES_DIR=/path/to/tiles_from_patch

# 1. 头条全表（v1→P0.1+P0.3，3 reps，落盘 headline_e2e.json）
CUDA_VISIBLE_DEVICES=<idle> PYTHONPATH=. python -m baseline.rerun_headline --repeats 3

# 2. 主流水线（v7 + P0.1 + P0.3 + graph capture）
CUDA_VISIBLE_DEVICES=<idle> FUILT_USE_COMBINED_KERNEL=1 FUILT_USE_ASYNC_D2H=1 \
    python -m baseline.test_baseline_v7_fuselevelset --mode pytorch_fft

# 3. CUDA Graph stream bug 的 5-mode 可控变量证明
CUDA_VISIBLE_DEVICES=<idle> python -m baseline.benchmark_v7_graph_modes --iters 20

# 4. Triton/CUDA/eager 后端消融（fuse4/split4，diff=0.0 校验）
python -m baseline.test_baseline_v8_bigbuf_ablation --out-json benchmarks/results/v8_ablation.json

# 5. P0.1 等价性独立校验（24-loop vs 合并核，fp32/fp64 + 扰动 sanity）
CUDA_VISIBLE_DEVICES=<idle> python -m baseline.verify_p01_equivalence --out-json benchmarks/results/p01_equivalence.json
```

全部 raw 输出已提交：`benchmarks/results/*.json`（含 nsys 派生报告 `nsys_reports/`；二进制 `.nsys-rep` trace 未包含在本仓）；
"数字 → 脚本 → 命令 → 环境"的完整台账见 [`verifiable_results.md`](verifiable_results.md)。

## 6. 限制条件

- **不外推**：22.5× 仅在 RTX 4090 / 256-tile / 该 FFT 路径验证；未在 H100、多卡或更大 tile 上测过
- **VRAM 边界**：1024-tile 峰值占 24GB 的 95%，贴悬崖运行，fresh run 可能 OOM（如实标注，不承诺生产稳定）
- **DRAM 流量是推算**：仓库无 NCU 数据，融合收益的流量减少按张量形状算术得出；实测支撑只有同负载耗时下降
- **kernel 面收敛**：可辩护的自写 kernel 面只有"Triton elementwise + 2D 模板"；手写 CUDA 仅作性能上限参考；
  `levelset_simple.cu`（直接空间卷积）是旁支实验路径——baseline 起始即 PyTorch FFT，它从未进入主线
- **两个测量域不可互换**：kernel 微基准（µs/调用）≠ 端到端吞吐；流水线重叠会摊薄微基准差距
  （Triton 隔离 launch 慢 ~3×，in-pipeline 后与 CUDA 持平在 ~7.5 µs）
- **多卡未实现**：sharded ILT 有通信预算设计与 halo-exchange 微基准
  （`baseline/multigpu_band_exchange.py`，含 unbatched isend/irecv 在本机 VM 崩溃、GPU 间
  P2P 不可用两个环境发现），但完整多卡流水线不存在；单卡上 flat fuse 优于 hierarchical（45×），
  hierarchical 的价值只在多卡
