# FuILT GPU Acceleration — From a Bandwidth-Bound Pipeline to a 22.5× Algorithm Rewrite

> **From "why is it slow" to a strictly-equivalent algorithm rewrite + data-flow orchestration**
>
> 计算光刻（ILT）反演 · PyTorch FFT 基线 · RTX 4090 · torch 2.11 / Triton 3.6 / CUDA 13 · 256 tiles × 20 iters
>
> **核心贡献**：定位并消除"GPU 大部分时间在等数据"这一系统性瓶颈——不是靠更快的 kernel，而是靠三层叠加：
> **① 算法级重写**：利用 IFFT 线性性把每 iter 的 24 次 ifft2 合并成 1 次（数学严格等价；实测 fp32 max_abs_diff ≤ 3.7e-9，单项 3.20×）；
> **② 数据流编排**：pinned + async + stream 把 H2D/D2H 藏进计算（3.97× 起）；**③ 证据整改**：每条头条数字都落成
> "committed 脚本 + raw JSON + 一条命令可复现"。最终 **v1 53.07s → P0.1+P0.3 2.36s = 22.5×**，全程以 Triton kernel + 算法/内存优化为主线。

---

## 0. Executive Summary

这个项目最初不是为了"写一个快 kernel"，而是从一个更宽的工程问题开始：

> **计算光刻反演每步要在 GPU 上对 256 个 tile 做 FFT 仿真。它为什么这么慢？瓶颈到底在 kernel、在数据搬运、还是在算法本身的表达方式？**

整个项目经历了五次明显的收敛：

1. **先测清瓶颈而不是先改 kernel**：P0.5 的 profile 表明 256× H2D 占 H2D_Init 阶段 **85%**——问题不是"算得慢"，是"GPU 在等数据"。这个反直觉的观察把方向从"优化 kernel"拉到"优化数据流编排"。
2. **先做算法级重写而不是优化 kernel**：盯着 profile 看到 IFFT 是绝对大头，但**不是 IFFT 慢，是 24 次太多了**。利用 IFFT 线性性把 `Σ s[k]·IFFT(X_k)` 重写成 `IFFT(Σ s[k]·X_k)`——24 次塌缩成 1 次，**数学严格等价**。
3. **用数据流编排解决传输**：H2D/D2H 用 pinned + async + stream 藏进计算。DMA 和 SM 是不同引擎，传输本就不该占计算的时间线。
4. **融合 + 数据结构重构**：v7 融合减少中间张量过 HBM；v8 把独立小张量重构为统一 bigbuf，地址可算、访问合并。
5. **证据整改**：把"历史 print-only"的数字全部重跑落盘或删除；不可复现的（P1/P2 graph 数字）删除，反常的（v5 AMP 无差异）如实记录。

这个项目最终不是"调了几个 launch 参数"，而是一条完整的 systems investigation：

```text
baseline characterization（H2D 占 init 85%，问题在"等数据"）
    ↓
data-flow orchestration（pinned + async + stream overlap）
    ↓
algorithm-level rewrite（IFFT 线性性 24→1，数学严格等价）
    ↓
fusion + data-structure restructure（减中间张量、bigbuf 地址合并）
    ↓
CUDA Graph capture bug（默认流被静默跳过 → getCurrentCUDAStream 修复）
    ↓
scale / robustness（1024-tile 撞 VRAM 悬崖 → allocator slack 修复）
    ↓
evidence audit（全表重跑、删除不可复现、修正 harness bug）
    ↓
22.5× 且每条数字可复现
```

---

## 1. Project Scope

### 1.1 What this repository is

这是一个 **计算光刻 GPU 加速实验台**，目标是把 ILT 反演中"慢在哪"拆成可测、可归因、可复现的系统问题。

主要研究对象：

- FFT 仿真（convolution theorem → 频域点乘）的 GPU 加速；
- H2D/D2H 数据流的编排（pinned / async / stream / overlap）；
- 算法级等价重写（IFFT 线性性把多次 IFFT 合并）；
- kernel 融合与中间张量流量（HBM 往返）的控制；
- 数据结构重构（独立小张量 → 统一 bigbuf 地址合并）；
- CUDA Graph 捕获语义（手写 kernel 的 stream 陷阱）；
- VRAM 边界下的可扩展性（1024-tile 的 allocator 行为）；
- 证据可复现性（committed raw JSON + 一条命令重跑）。

### 1.2 What this repository is not

- 不是完整的光刻生产部署方案（tile 切分、OAS 解析已简化）；
- 不是为了证明"手写 CUDA 一定快于 Triton"（v8 实测 CUDA 在特定 kernel 更快，但 Triton 是生产推荐路径）；
- 不是对 4090 之外的 GPU / 更大模型的吞吐外推（未在 H100/多卡验证）；
- 不把调度器/自报时间当作 worker/GPU 真值（以 nsys 第三方对账为准）；
- **不是手写 CUDA kernel 的展示**：baseline 从第一天起就是 PyTorch FFT 路径；`levelset_simple.cu`（CUDA 直接空间卷积）只是旁支实验路径，从未进入主线、也未通过等价校验（`compare_*_cuda_op.json`），**不作为性能声明**；可辩护的 kernel 面只到"Triton elementwise + 2D 模板"。

---

## 2. Environment and Evidence Policy

### 2.1 Main experimental environment

| Dimension | Setting |
|---|---|
| GPU | 1× RTX 4090 24GB（8-GPU 主机，**CUDA_VISIBLE_DEVICES 隔离**；争抢会差 5-15×） |
| 计算光刻负载 | 256 tiles（16 base × 16 clone），1024×1024 tile，20 iters |
| torch / Triton / CUDA | 2.11.0+cu130 / 3.6.0 / 13.0 |
| nsys | 2025.3.2（`.nsys-rep` 已提交，~11-20MB × 5 variants） |
| benchmark | 3 reps，mean ± std（`rerun_headline.py` 子进程驱动） |
| tiles 数据 | `/data/lyj/FuILT/tiles_from_patch` |

### 2.2 Evidence levels

本项目统一使用以下证据等级：

- **FACT**：直接测量、源码可定位、committed raw JSON 可复算（`headline_e2e.json`、`graph_modes.json`、`v8_ablation.json`）；
- **INFERENCE**：由多个 FACT 支持的解释，但不是独立因果识别（如"H2D 占 init 85% 所以 overlap 有效"）；
- **HYPOTHESIS**：尚未验证；
- **REFUTED**：曾经合理，但已被后续实验推翻（如 P0.4 的 batch H2D、"Triton 27% faster"）。

被推翻的结论不会从历史中删除，而是保留"为什么当时合理、什么数据推翻了它"。

---

## 3. Phase A — Establish the Baseline: 问题不是算得慢，是 GPU 在等

### 3.1 v1 基线

最早的端到端：逐 tile 串行 `H2D 搬参数 → GPU 算 → D2H 搬回 → 下一个 tile`，纯 PyTorch FFT。直觉上"GPU 够快"。结果：**53.07 ± 0.51 s**。

### 3.2 反直觉的 profile 结果

P0.5 对 256 次 H2D 的 profile 显示：**H2D 占 H2D_Init 阶段 ~85%**。

| 观察 | 含义 |
|---|---|
| H2D 占 init 阶段 85% | 每个 tile 串行搬一次，PCIe 往返 + launch 开销叠加 |
| kernel 本身快 | 计算不是瓶颈，**数据进不来的时间才是** |
| 过载首先体现为等待 | 不是"请求失败"，是"GPU 空转等数据" |

**关键判断**：问题根本不在计算，在**数据进出的编排**。53s 里大部分是"等"而不是"算"。这直接决定了后续方向——先解决传输，而不是优化 kernel。

---

## 4. Phase B — Data-Flow Orchestration: 把 H2D/D2H 藏进计算

### 4.1 v3：H2D overlap（3.97×）

**做法**：H2D 改用 `pinned host memory + non_blocking=True`，让拷贝在 GPU 干初始化/前一轮计算时异步进行。

**为什么能重叠**：DMA copy engine 和 SM 是**不同硬件引擎**——传输不占计算资源。**为什么必须 pinned**：DMA 要求地址物理固定，async 的前提是 pinned。

**结果**：53.07 → 13.36 s，**3.97×**。

> 核心洞察：**不是搬得快了，是搬的时候 GPU 没闲着。**

### 4.2 P0.3：async D2H（对称性，~1.20×）

**做法**：D2H 从同步拷贝改成 `pinned + non_blocking`，让结果搬回 CPU 与下一个 tile 的 fuse 重叠。代码注释原话：*"non_blocking lets the D2H run concurrently with the GPU workspace.fuse()"*。

**结果**：P0.1 2.80 → P0.1+P0.3 **2.36 s**，端到端 **22.5×**（头条数字）。

### 4.3 传输的代价与边界

- async 的前提是 pinned → pinned 占物理 RAM（1024-tile 时 ~7.8GB 锁定的 host 内存）；
- overlap 的正确性要靠 stream/event 保序（kernel 不能读到没搬完的数据）。

---

## 5. Phase C — Algorithm-Level Rewrite: IFFT 线性性（P0.1，单项 3.20×）

### 5.1 观察

v7 每 tile 每 iter 做 24 次 `ifft2(mask_fft * kernel_fft[k])`。盯着 profile，IFFT 是绝对大头——但**不是 kernel 慢，是 24 次太多了**。

### 5.2 数学（三个引理）

基线：`aerial = Σ s[k]·Re{IFFT2(mask_fft ⊙ kernel_fft[k])}`（24 次 IFFT）。

| 引理 | 内容 | 前提 |
|---|---|---|
| L1 实标量可进出 IFFT | `IFFT(s·X) = s·IFFT(X)` | s 是实数（剂量权重，实 fp32 ✓） |
| L2 IFFT 线性 | `IFFT(ΣXₖ) = ΣIFFT(Xₖ)` | IFFT 定义里的和是线性的 |
| L3 实部算子可进出实加权和 | `Σ s·Re(z) = Re(Σ s·z)` | s 实数 ✓ |

推导：`Σ s[k]·Re{IFFT(X_k)} = Re{IFFT(Σ s[k]·X_k)} = Re{IFFT(mask_fft ⊙ combined_kernel_fft)}`。

### 5.3 为什么成立（关键前提）

`combined_kernel_fft = Σ s[k]·kernel_fft[k]` 只依赖**固定光学系统**（与 mask 无关）→ 可以 **init 预计算一次**（~25ms，5120 次 tile-iter 全复用）。

**`combined_kernel_fft` 是什么**：一块**常驻 GPU 显存的复数张量**——`[1024,1024]` complex64，~8MB，装着 24 个光刻核按剂量加权求和后的**频域 PSF**。这是"static workspace"的实体：init 算一次、地址不变、每 iter 复用。

### 5.4 结果

| | 原 | 优化后 |
|---|---|---|
| 每 iter ifft2 | 24 | **1** |
| LevelSet 阶段 | — | **10.5×** |
| 端到端（v7→P0.1） | 8.96s | 2.80s（**3.20×**） |
| 数值 | — | **数学严格等价**；实测 fp32 `max_abs_diff ≤ 3.7e-9`（真实光学核 ≤ 2.2e-11；fp64 ~1e-18；测量与推导见 §5.5） |
| 新 kernel / 新硬件 | — | 零 |

> 核心洞察：**profiler first, kernel later——瓶颈往往在"算法怎么表达"，不在 kernel。** 一个被遗忘的线性性事实让 GPU 少做了 23/24 的活。

### 5.5 数值等价性：怎么测、怎么推导（2026-08-25 补全）

**测量**（`baseline/verify_p01_equivalence.py`）：同一进程、同一份 seeded 输入，两条**零共享代码**的路径各跑一遍——OLD（24× ifft2 + 像域累加）vs NEW（频域合并 + 1× ifft2）——取 1024² 元素 `|OLD−NEW|` 的 max。三个防假阳性设计：① **1-ulp 扰动 sanity 臂**（对 NEW 单元素加 1 ulp，比较器必须报非零——灵敏度实证）；② **fp32/fp64 双臂**（若数学不等价，fp64 差异不会塌缩）；③ raw JSON 落盘。

**推导**（前向误差模型，完整版见 proof 文档 "Error-scale derivation"）：每条路径返回 `fl(A)+e`，`|e| ≲ γ_c·u·S`（u=单位舍入 2⁻²⁴/2⁻⁵³，c≈K+log₂N≈44 级操作，S=中间部分和量级）⟹ 差异**对 u 线性**。两个可证伪预言，全部命中：

| 预言 | 实测 |
|---|---|
| `diff_fp32/diff_fp64 = u₃₂/u₆₄ = 2²⁹ ≈ 5.369e8` | 合成 5 seed：**4 个三位有效数字精确命中 5.37e8**；真实核 3 seed：1 个精确命中、2 个 6.4–6.7e8 |
| 绝对量级在 `44u ≈ 2.6e-6` 界内 | 实测 4–8 ulp（随机符号 √c 消去解释富余） |

**真实核复测**（`--real-kernels`：`focus.pt` 经流水线自己的 `_build_kernel_fft` 构造，scales 和≈210）：fp32 ≤ **2.2e-11**、fp64 ≤ 3.4e-20——比合成臂低两个量级（真实 35×35 核稀疏、幅值小），同样 2²⁹ 签名（`p01_equivalence_real.json`）。

> 一句话：**等价是定理；残差对 u 线性（2²⁹ 精确命中 ⟹ 无精度相关放大）、4–8 ulp、有界可复现。** "数学严格等价、浮点纯舍入"从声明升级为推导 + 双预言验证。

---

## 6. Phase D — Kernel Fusion + Data Structure: 减中间张量、地址合并

### 6.1 v7 融合 + Triton pointwise

把 levelset 计算也 fuse 进流水线；pointwise 融合 kernel 用 **Triton** 实现（`triton_fused_pointwise.py`）。

**为什么融合**：每多一个 kernel 就多一次 launch + 中间结果过 HBM。融合后中间值留在寄存器，不碰 HBM（中间张量流量 G12）。

**融合了多少个算子（2026-08-25 实测定论）**：生产 eager 链（`_compute_pointwise` unfused 分支，v5/v6 真身）
**10 个 ATen kernel → 1 个**（torch.profiler 计数，`pointwise_kernel_count.json`）。
此前文档写 "8"——那是 microbench 简化表达式的数（漏了 steepness 乘法链与 diff_sq）；README 的 "~5" 是更早
粗估。"数表达式"与"profiler 实测"的差值本身就是一课。

**融合收益（三域 × 三后端实测，1024² fp32，`pointwise_kernel_count.json`）**：

| 域 | eager 10-op 链 | Triton 融合（生产） | CUDA 融合（上限参考） |
|---|---:|---:|---:|
| 纯 GPU 时间（profiler self-time） | 34.9 µs | **6.1 µs（5.8×）** | **5.4 µs（6.5×）** |
| DRAM 流量（形状推算） | 96 MiB | **16 MiB（6×）** | 16 MiB（同算法结构，与实现无关） |
| wall（含分发） | 90.9 µs | **29.3 µs（3.1×）** | **11.4 µs（8.0×）** |

三列对照给出三层结论：

1. **GPU 侧：流量比 6× ≈ GPU 时间比 5.8×/6.5×**——两个独立方法（形状算术 vs profiler）证实同一本质
   收益（少搬 80 MiB + 消掉 9 个 kernel 尾部）；CUDA kernel 本身略优（5.4 vs 6.1µs，~12%），
   手写版是真实上限——与 §11.2 的诚实结论一致。
2. **wall 侧：比值从 3.1× 拉开到 8.0×**——差距不在 kernel 而在分发路径：Triton 的 Python 分发 ~23µs
   vs pybind ~6µs，同一个"分发税"相差 4 倍。生产的完整答案是**融合 + graph capture**（分发也归零），
   这也是 v7 端到端 CUDA 8.96s ≈ Triton 9.18s（+2.5%）的原因——流水线重叠摊薄了分发差。
3. **确定性**：eager 链 wall 两轮 71→91µs 波动 ±25%（10 次 Python 分发对主机状态敏感），两个融合版
   都纹丝不动——**确定性**是融合的隐性收益。

**结果**：v7-CUDA 8.96s / v7-Triton **9.18s**——Triton 几乎追平 CUDA（elementwise 是 Triton 的甜区，且 Triton 默认 CUDA-Graph-safe）。

### 6.2 v8：bigbuf 数据结构重构

把"每 tile 独立小张量"的树形组织，重构为**统一 bigbuf**——所有 tile 按固定 stride 平铺进一块连续大缓冲。

**为什么**：独立小张量 = 每个 tile 一次分配 + 地址不可预测 + 访问分散。bigbuf 让 `tile_i 起始 = base + i×S×S` 地址可推导 → 一次分配 + 行内连续（合并访问）+ 地址稳定（graph friendly）。

**结果**（fuse4 同算子对比，H=W=4096，`v8_ablation.json`）：**PyTorch eager 179.8µs → Triton 46.0µs = 3.9×**。
（诚实记录：同配置手写 CUDA 29.2µs 是性能上限参考；Triton 是**生产推荐路径**。
旧文曾写 "PyTorch 285.5µs = 6.2×"——那是 fuse4+split4 总和对比 fuse4 单项，口径错配，已修正。）

---

## 7. Phase E — CUDA Graph 捕获 bug: 手写 kernel 的 stream 陷阱

### 7.1 现象

把流水线放进 `torch.cuda.graph()`，**replay 输出错但不报错**。

### 7.2 排查

eager 模式正确、graph 模式错 → 不是数学问题，是**捕获语义**问题。

**根因**：手写 kernel 用裸 `<<<>>>` 走**默认流**，而 graph 捕获发生在**私有 capture 流** → kernel 被静默跳过，图里没有它。

**修复**：所有手写 kernel launch 显式传 `at::cuda::getCurrentCUDAStream()`。PyTorch 算子天然安全（内部走当前流），Triton 也默认安全（launch 自带当前流），**只有手写 CUDA 踩这个坑**。

### 7.3 可控变量证明（5-mode）

`graph_modes.json`：同一份代码只切 launch 用默认流还是当前流。

| mode | 配置 | loss-rel | 状态 |
|---|---|---|---|
| 3 | graph + capture_safe=OFF（默认流） | **1.02** | **FAIL** |
| 5 | graph + capture_safe=ON（当前流） | 0.00 | **PASS** |

> 这是"可控变量实验"证明 bug 根因的标准示范——不是猜，是同一个 kernel 只改 stream 从错到对。
>
> **节点级复核（2026-08-25）**：`verify_graph_stream_capture.py` 用同一个编译产物的运行时开关捕获单 kernel 区域——OFF 时 torch 直接警告 "CUDA Graph is empty"、replay 后输出冻结（Δ=0.0）；ON 时 replay 重算且与 eager 完全一致。"没进图"从推断变成直接观测（`graph_stream_capture.json`）。

---

## 8. Phase F — Scale & Robustness: 1024-tile 撞 VRAM 悬崖

### 8.1 现象

batch 扩到 1024 tile，撞上 24GB 显存极限，**OOM 2/3**。

### 8.2 根因：allocator slack

`reserved 23352MB` vs `live 19769MB`——**~3.6GB 是被 torch 缓存分配器"预留但不还"的内存**。

### 8.3 修复 + 边界诚实标注

`torch.cuda.empty_cache()` 在峰值 D2H_Save 前调用 → 5/5 跑通，峰值 23337MB。

**诚实标注**：这是贴着 VRAM 悬崖（24GB 的 95%）的边界探测——**fresh run 可能 OOM**，不承诺生产稳定。

---

## 9. The Core Rewrite — 什么真正交付了 22.5×

如果只讲一个优化，是 **P0.1 IFFT 线性性**：不是更快的 kernel，是让 GPU 少做 23/24 的活，且**数学严格等价**。它的可迁移模式：

```text
"K 个固定核的加权 IFFT/卷积 对同一输入的求和"
    ↓
线性性 → 把加权和先算进频域（init 一次）
    ↓
每 iter 只做 1 次 IFFT
```

适用场景：multi-head attention 的 per-head QK^T 归约、多核卷积。**这是"算法级等价重写"这一类优化的代表——投入 5 行代码，换 3.20×，零风险（数学严格等价，实测 fp32 diff ≤ 3.7e-9）。**

---

## 10. Causal Validation — 头条数字为什么可信

### 10.1 全表重跑（2026-08-04，GPU 7 独占，3 reps）

| Variant | Pure (s) | Speedup | README old |
|---|---:|---:|---:|
| v1 | 53.07 ± 0.51 | 1.00× | 53.37 ± 0.35 |
| v3 | 13.36 ± 0.05 | 3.97× | 13.46 ± 0.05 |
| v5 AMP=1 | 18.05 ± 0.19 | 2.94× | 18.03 ± 0.07 |
| v5 AMP=0 | 18.01 ± 0.16 | 2.95× | (未记录) |
| v6 | 19.61 ± 0.10 | 2.71× | 19.53 ± 0.13 |
| v7-CUDA | 8.96 ± 0.08 | 5.92× | 8.71 ± 0.11 |
| v7-Triton | 9.18 ± 0.03 | 5.78× | 8.98 ± 0.27 |
| P0.1 | 2.80 ± 0.08 | 19.0× | 2.81 ± 0.02 |
| **P0.1+P0.3** | **2.36 ± 0.05** | **22.5×** | 2.34 ± 0.02 |
| P0.4 | 4.24 ± 0.11 | 12.5× | 4.32 ± 0.04 |

每一行都在 README 表的几个百分点内复现；22.8× → 22.5× 是安静主机下的权威值。

**2026-08-26 哨兵复测**（真空闲 GPU 7 自动触发，`headline_e2e_20260826_sentinel.json`）：全表复现，基线侧 ±0.5% 内，**头条 21.6×**（P0.1+P0.3 侧 +4.1%，D2H/IO 敏感 + 时钟残留）——22.5× 保持权威，跨环境散布 ~±4%。

**2026-08-26 全栈补测（graph capture 叠加，此前从未进头条数字）**：eager 每 iter 循环 40.3 ms（stage 表）
vs graph replay **29.16 ms/iter**（三连测 29.156/29.158/29.154，±0.01%）→ 循环节省 27%；但循环节只占纯
流水线 ~31%（H2D 30% + D2H 38% 是大头）→ 全栈纯 ≈ 2.23 s ≈ **23.8×**（推导值：graph 模式不打印同口径
纯耗时，非循环段按 eager 同段代入）。nsys 全栈 trace（`v7_fullstack_graph`，已入库）证实剩余格局不变：
FFT 链 34.1% / abs+pow 24.0%（eval 相）/ pointwise 0.2% / **D2H 占 mem-op 90.5%**——graph 的收益被
P0.1/P0.3 提前吃掉大半，新瓶颈是 IO 段。**"全部有效加速"盘点**：头条链（22.5×）+ graph（≈23.8× 上限）
为栈内全部；v8 bigbuf（3.9× kernel 级）、P0.2（1.48× eval 相）未整合进主流水线；P0.4/AMP/v4 为反例；
v2 从未进现代复测表（被后续吸收，小缺口）。

### 10.2 反例排查

- **v5 AMP 双态**：脚本默认 AMP=off，README 声称 fp16——两个状态都测了：**18.05 vs 18.01，无差异**（graph 开销主导，AMP 声明成立但改变不了什么）。
- **49.9 vs 50.0 对账**：README 文本与表格矛盾，根因是混用了两个 nsys trace（`v7_triton` vs `v7_triton_rerun`），统一后 **50.0%**。
- **harness 两个 bug**：kernel micro-bench 的 import 错误（模块名不存在被 try/except 吞掉）+ 缺 warmup（首次 Triton 调用把 ~1.3s JIT 计进测量）。
- **P0.1 "bit-exact" 声明撤回（2026-08-20）**：原引用的 `grad_max_abs_diff=0.0` 出自 Triton-vs-CUDA pointwise 校验，与 P0.1 无关（张冠李戴）。独立 A/B 校验（`verify_p01_equivalence.py`）：fp32 max_abs_diff ≤ 3.7e-9、fp64 ~1e-18——数学严格等价成立，逐位相等是过度声明，全部文档改为实测口径。**同期撤回的还有"levelset_simple.cu 被 FFT 路径（逐位等价）替代"的整个叙事**：baseline 起始就是 PyTorch FFT，不存在"空间卷积被替代"的历史；`compare_levelset_vs_fft_*.json` 的 0.0 测的是 FFT 原版 vs FFT 重排（参考实现前向本走 FFT），CUDA 空间卷积从未参与任何等价对比。随后补全了完整论证：前向误差模型预测残差对 u 线性，实测 fp32/fp64 差异比精确 = 2²⁹（两精度 epsilon 之比），真实核复测 ≤ 2.2e-11（见 §5.5）。

---

## 11. Measurement Audit — 数字到底意味着什么

### 11.1 传输与 kernel 是不同测量域

- nsys in-pipeline 的 pointwise kernel 耗时 ~7.5µs（CUDA 和 Triton 一样）；
- 隔离的 per-launch 延迟：CUDA 8.1µs vs Triton 24.5µs；
- 两者不矛盾——Triton 的 Python 分发开销在 launch 重叠后被藏住。**不能拿两个域的 8.1 vs 24.5 去否定 nsys 的 7.5。**
- **三域收敛的标准案例（2026-08-25，`pointwise_kernel_count.json`）**：同一对（eager 10-op 链 vs Triton 融合）
  在三个域给出三个比值——流量推算 6×、纯 GPU 时间 5.8×、wall 3.1×。前两者收敛验证本质收益；wall 的差价
  是 Triton Python 分发（29.3µs wall 中 GPU 只占 6.1µs）。结论：**报融合收益必须声明测量域**，
  wall 域会系统性低估 kernel 能力。

### 11.2 手写 CUDA vs Triton 的诚实结论

- pointwise：CUDA 8.1µs < Triton 24.5µs（elementwise 手写确实更快）；
- fuse4（v8）：CUDA 29.2µs < Triton 46.0µs；
- 但 v7 端到端：CUDA 8.96s ≈ Triton 9.18s（+2.5%）——**kernel 微基准的差距被流水线重叠摊薄**。

结论：Triton 是**生产推荐路径**（开发效率 + graph-safe + 差距可接受）；CUDA 存在但作为 perf-ceiling 参考，**不宣传"Triton 更快"**。

### 11.3 反直觉的反常都保留

- v5 AMP 无差异、P0.4 失败、P0.5 dead-end——全部如实记录，不为了"一路加速"的叙事抹掉。

---

## 12. What the Project Actually Proves

### 12.1 Strong claims

在已测试的 256-tile × 20-iter / RTX 4090 / 计算光刻 FFT 路径中：

1. 早期瓶颈是**数据流编排**（H2D 占 init 85%）而不是 kernel 计算；
2. IFFT 线性性把 24 次 ifft2 合并成 1 次，**数学严格等价**（实测 fp32 diff ≤ 3.7e-9）、零新 kernel、单项 3.20×；
3. pinned + async + stream 的 H2D/D2H overlap 把传输藏进计算（3.97× 起）；
4. 手写 kernel 在 CUDA Graph 捕获时若走默认流会被**静默跳过**（mode3 FAIL / mode5 PASS）；
5. 头条 22.5× 在安静、隔离 GPU 上 3 reps 复现，每行数字有 committed raw JSON；
6. Triton 作为生产路径，端到端与 CUDA 差 ~2.5%，但开发效率与 graph-safe 收益明显。

### 12.2 Claims intentionally not made

- 不声称"手写 CUDA 一定慢于 Triton"（fuse4 实测 CUDA 更快）；
- 不声称 22.5× 能外推到其他 GPU / 模型 / 更大 tile（未在 H100/多卡验证）；
- 不把 1024-tile 的 8.31s 当作稳定生产数字（贴着 VRAM 悬崖，fresh run 可能 OOM）；
- 不把 kernel 微基准（8.1 vs 24.5µs）当作端到端吞吐的预测（流水线重叠会摊薄）；
- 不声称算法级重写适用于所有 FFT 场景（需要固定核 + 线性 + 无非线性算子）。

---

## 13. Negative Results and Why They Matter

| Branch | Result | Decision |
|---|---|---|
| **P0.4 batch H2D**（`np.stack+unbind` 批量搬） | 引入 2GB 额外 copy，反而 12.5× | **Refuted**——"显然快"的优化不一定真快，搬移总量没减 |
| **P0.5 `.contiguous()` 移除** | 无收益 | **Dead-end 文档化** |
| **v5 AMP** | 18.05 vs 18.01 无差异 | **保留反常**——graph 开销主导 |
| **FP8 / 手写复杂 CUDA**（levelset 直接卷积） | 旁支实验路径：baseline 起始即 FFT，它从未进入主线 | **不作为性能声明** |
| **1024-tile 稳定运行** | OOM 2/3 后才 empty_cache 修复 | **边界条件，如实标注** |
| **P1/P2 graph 数字**（2.11× 等） | 来自未提交 POC，不可复现 | **删除** |

保留负结果是可信度的一部分：每次 pivot 都由数据触发，而不是为了维持预设故事。

---

## 14. Reproducibility and Evidence Layout

```text
benchmarks/results/            # committed raw JSON（头条/5-mode/v8/hierarchical/microbench）
├── headline_e2e.json          # 头条全表 3 reps（rerun_headline.py）
├── graph_modes.json           # 5-mode stream bug 证明
├── v8_ablation.json           # Triton/CUDA/PyTorch fuse4
├── hierarchical.json          # 45×
├── p01_equivalence.json       # P0.1 等价性 A/B（合成数据，fp32 ≤3.7e-9）
├── p01_equivalence_real.json  # 同上、真实光学核（fp32 ≤2.2e-11）
├── graph_stream_capture.json  # capture-stream bug 节点级证明（空图 + replay 冻结）
├── pointwise_kernel_count.json # pointwise 算子数+计时实测（10→1；wall 3.1×/GPU 5.8×）
├── nsys_reports/              # 6 traces 派生统计（含 08-26 全栈 graph 版）
├── compare_*.json             # CUDA/PyTorch/levelset 数值等价
outputs/nsys_traces/*.nsys-rep # 已提交的 nsys trace（~11-20MB × 5）
docs/
├── fuilt_readme_final.md      # 招聘材料版 README（问题/贡献/数字/架构/复现/限制）
├── p01_ifft_linearity_proof.md    # P0.1 数学证明
├── bottleneck_shift.md            # 瓶颈迁移 Before/After（nsys + microbench 数字汇总）
├── gpu4fuilt_interview.md         # 面试作战文档（实验重要度表 + 三条故事路径）
├── kernel_report_card.md          # kernel 十连问报告卡（SASS 反汇编：向量化/寄存器/occupancy）
├── verifiable_results.md          # 证据台账（数字→脚本→复现命令→环境）
└── nsys_trace_guide.md
baseline/rerun_headline.py         # 头条表可复用驱动
baseline/verify_p01_equivalence.py  # P0.1 等价性 A/B（--real-kernels 用真实核）
baseline/verify_graph_stream_capture.py  # graph 捕获 bug 节点级证明
baseline/multigpu_band_exchange.py # 多卡 halo 交换微基准（冒烟通过，等空闲采集）
baseline/count_pointwise_kernels.py # pointwise 算子计数+计时（torch.profiler）
```

复现入口：`docs/verifiable_results.md` 里每个数字都有**脚本 + 命令 + 环境 + raw 输出**。

---

## 15. Interview Narrative

如果只讲这个项目，不建议从"我加速了光刻仿真"开始。

建议主线：

> 我先做 profile，发现 H2D 占 init 阶段 85%——问题不是算得慢，是 GPU 在等数据。所以第一步不是优化 kernel，而是用 pinned + async + stream 把传输藏进计算（3.97×）。但真正的跳变是 P0.1：我发现每 tile 要做 24 次 ifft2，而 IFFT 是线性算子——把 24 个核的加权和先算进频域（init 一次），每 iter 只剩 1 次 IFFT——数学上严格等价，浮点上实测 diff 在 1e-9 量级且可推导：fp32/fp64 差异比精确等于 2²⁹，纯舍入无放大（不是逐位相等）。这个算法级重写单项 3.20×，零新 kernel。之后我做了融合减中间张量、bigbuf 地址合并，还在把流水线放进 CUDA Graph 时亲手定位了一个手写 kernel 的 stream bug：裸 `<<<>>>` 在 capture 时被静默跳过，我用 5-mode 可控变量实验证明了根因。最后我把所有头条数字重跑落盘，删掉不可复现的，修正了 harness 的两个 bug——端到端 22.5× 是 3 reps 复现的，不是纸面数字。

面试官继续追问时，应优先展开：

1. 为什么 IFFT 线性性成立？（三个引理 + 前提：实标量、核不随 mask 变、无非线性算子）
2. 为什么 H2D 能和计算重叠？（DMA 和 SM 是不同引擎 + pinned 前提）
3. 为什么手写 kernel 在 graph 里被静默跳过？（默认流 vs 私有捕获流）
4. 为什么 v5 AMP 开关无差异？（graph 开销主导）
5. 为什么 P0.4"显然快的批量 H2D"反而慢了？（引入中间拼接，搬移总量翻倍）
6. Triton vs CUDA 你最终怎么选？（Triton 生产路径，差 ~2.5%，graph-safe + 开发效率）
7. 这个 22.5× 能外推吗？（未在 H100/多卡验证；1024-tile 贴 VRAM 悬崖）

---

## 16. Final Takeaway

这个项目最有价值的不是 22.5× 这个绝对数字，而是一套可复用的 **GPU 加速调查方法**：

```text
先做 profile（瓶颈在哪，而不是"哪里该优化"）
    ↓
区分瓶颈类型（带宽受限？传输受限？算法表达受限？）
    ↓
先试算法级等价重写（数学化简，零风险，收益最大）
    ↓
再做数据流编排（overlap、pinned、stream）
    ↓
融合与数据结构（减中间张量、地址合并）
    ↓
处理框架语义坑（CUDA Graph 捕获、默认流）
    ↓
诚实记录失败与反常（负结果也是结论）
    ↓
证据落盘（committed raw JSON + 一条命令可复现）
```

**项目定位**：不是"把 FFT 跑快了"，而是能**从 profile 定位到真正的瓶颈类型，用算法重写 + 数据流编排层层收敛，并把结论做成可复现、可审查、诚实的证据闭环**——这是一条完整的 GPU/推理性能工程方法。
