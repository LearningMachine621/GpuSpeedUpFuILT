# 瓶颈迁移图 — Before / After（全部来自已提交证据）

> 两条面试用的故事线，每个数字都能指到 committed 文件。**没有新跑任何 benchmark，没有新写任何 CUDA。**
>
> 证据口径：
> - **实测**：`outputs/nsys_traces/*.nsys-rep`（已提交）派生的 `benchmarks/results/nsys_reports/*.txt`，
>   以及 `benchmarks/results/*.json`（raw 输出，2026-08-03/04，GPU 7 独占 RTX 4090，torch 2.11 / triton 3.6）。
> - **推算**：本仓库**没有 NCU 数据**（无 `.ncu-rep`），DRAM 流量一律是按张量形状做的算术
>   （fp32 1024×1024 = 4 MiB/tensor），明确标注为推算值，不冒充实测。
> - 台账：`docs/verifiable_results.md`；复现命令见 `benchmarks/results/README.md`。

---

## 1. 故事一：瓶颈是怎么一步步迁移的

```text
Before  (v1 基线, 53.07 s, headline_e2e.json)
│
├─ CPU/H2D ──────── 逐 tile 串行 H2D→算→D2H；P0.5 profile:
│                   256× H2D 占 H2D_Init 阶段 85% (README §P0.5)
│
├─ pointwise ────── eager 链 ~8 kernel/tile；v7 trace 里 launch 仍可见:
│   launch          cudaLaunchKernel 160,796 次 / 848 ms (18.1% API 时间)
│                   + cuLaunchKernel  67,988 次 / 248 ms  ( 5.3%)
│
└─ FFT ───────────  传输/launch 清干净后浮出水面 (v7 trace, pre-P0.1):
                   FFT 链 = 50.0% GPU 时间
                   = regular_fft 19.5% + vector_fft 18.2%
                     + 复数乘(只为 IFFT 循环存在) 12.3%
        │
        │  v3  pinned+async H2D          53.07 → 13.36 s  (3.97×)
        │  v7  融合 + graph capture      pointwise 8 kernel → 1 (0.3% GPU 时间)
        │                                 整条流水线 graph replay 116.5 ms/iter
        │  P0.3 async D2H                把 D2H 藏进计算
        ▼
After
└─ IFFT 循环成为最大单项:
     · 24× ifft2+累加 = 1.272 ms/tile-iter   (kernel_microbench.json)
     · 单次前向 fft2 = 0.029 ms             → IFFT 循环 ≈ 43× 前向
     · 融合 pointwise  = 0.023 ms           → IFFT 循环 ≈ 54× pointwise
        │
        │  P0.1  IFFT 线性性: Σ s·IFFT(Xₖ) = IFFT(Σ s·Xₖ)
        │        24 次 ifft2 → 1 次, 数学严格等价
        ▼
     v7 8.96 s → P0.1 2.80 s (3.20×) → P0.1+P0.3 2.36 s (22.5×)
     post-P0.1 trace: 复数乘 12.3% → 0.4%, FFT 链 50.0% → 33.9%
```

### 1.1 Before 三行各自的证据

| 行 | 数字 | 出处（均为 committed） |
|---|---|---|
| CPU/H2D | v1 逐 tile 串行，53.07 ± 0.51 s | `headline_e2e.json` |
| CPU/H2D | 256× H2D 占 **H2D_Init 阶段** 85%（README 原话口径，不是"占总时间"） | README §P0.5 |
| CPU/H2D | 即便到 v7（pre-P0.3），trace 段内 `cudaMemcpyAsync` 59,597 次 / 1.448 s（30.9% API 时间）；`cudaStreamSynchronize` 58,573 次 / 1.831 s（39.1%）——CPU 在"等" | `v7_triton_rerun_cuda_api_sum.txt` |
| CPU/H2D | mem-op 时间里 **D2H 占 91.4%**（0.966 s vs H2D 0.090 s）——这正是 P0.3 async D2H 要藏掉的东西 | `v7_triton_rerun_cuda_gpu_mem_time_sum.txt` |
| pointwise launch | eager 链每 tile ~8 个 elementwise kernel（`sub, mul, sigmoid, sub, rsub, mul, mul, mul`，按 `bench_pointwise_torch` 表达式数出，代码可查）；整个 trace 段 ~22.9 万次 kernel launch、~1.10 s CPU API 时间 | `run_benchmark_matrix.py:495` + `v7_triton_rerun_cuda_api_sum.txt` |
| FFT | v7 trace GPU 时间 3.096 s 中 FFT 链 **50.0%**：`regular_fft` 19.5% + `vector_fft` 18.2% + 复数乘 `BinaryFunctor<c10::complex>` 12.3%（30,768 次，每 tile-iter 24 次地喂 IFFT） | `v7_triton_rerun_cuda_gpu_kern_sum.txt` + `nsys_reports/README.md` 对账表 |

### 1.2 After："IFFT becomes dominant" 的证据

| 数字 | 出处 |
|---|---|
| 前向 `fft2`（1024², complex64）= **0.0293 ms** | `kernel_microbench.json` (`fft2_ms`) |
| 24× `ifft2+累加` 循环 = **1.2717 ms**/tile-iter（**43×** 前向；54× 融合 pointwise） | `kernel_microbench.json` (`ifft2_accum_ms`)，bench 代码 `run_benchmark_matrix.py:488` |
| IFFT 侧在 GPU 时间线里的下界：复数乘 12.3% **只为** IFFT 循环存在（前向 fft2 不需要它）→ IFFT 侧（cmul+ifft2+accum）≥ 30.5%，大于前向侧 19.5% | `v7_triton_rerun_cuda_gpu_kern_sum.txt` |
| P0.1 之后复数乘塌缩到 **0.4%**（折进 init 一次的 `combined_kernel_fft` 频域求和），FFT 链 50.0% → 33.9% | `v7_p013_nvtx_cuda_gpu_kern_sum.txt` |
| E2E：v7 8.96 s → P0.1 2.80 s（**3.20×**），实测 fp32 `max_abs_diff ≤ 3.7e-9`（fp64 ~1e-18，`p01_equivalence.json`）；P0.1+P0.3 2.36 s = **22.5×** | `headline_e2e.json`、`baseline/verify_p01_equivalence.py` |

**讲的时候的两个诚实边界**（面试官如果懂 cuFFT 会问）：

1. kern_sum 里 `regular_fft` / `vector_fft` 是 cuFFT 的两类内部 kernel，**无法逐个归因**前向还是逆向（两 trace 中二者实例数严格成对：33,610/33,610、4,124/4,124）。所以"IFFT 是大头"的证据是 **microbench 的 per-op 实测（1.27 ms vs 0.029 ms）+ 复数乘的归因（12.3% 只服务 IFFT）**，不是把 `vector_fft` 说成 IFFT。
2. 两个 trace 的迭代段长度不同（33,610 vs 4,124 次 FFT 实例），**不能拿实例数比值当 24→1 的证据**；24→1 的证据是代码 + 数学证明 + E2E 3.20× + 复数乘 12.3%→0.4%。

---

## 2. 故事二：fused kernel — pointwise grad+loss（Triton）

选 `_fused_pointwise_grad_loss_kernel`（`src/fuilt/ilt/func/triton_fused_pointwise.py`，
CUDA 版 `fused_pointwise_kernel.cu` 的 drop-in 替换）：每个元素算
`z = k·(aerial − ρ); printed = σ(z); diff = printed − t; grad = 2·diff·k·printed·(1−printed); diff²`，
**一次读入、寄存器里做完、写出两个结果**。

### 2.1 Before → After

| 维度 | Before（PyTorch eager） | After（Triton fused） | 证据 |
|---|---|---|---|
| kernel 数 | **8 个** elementwise kernel（`sub→mul→sigmoid→sub→rsub→mul→mul→mul`，中间量逐个落 HBM） | **1 个** kernel（2 输入 2 输出，中间量留寄存器） | 算子表达式 `run_benchmark_matrix.py:495`；融合版 `triton_fused_pointwise.py:27` |
| 耗时（1024² fp32，CUDA events，warmup 后 100 reps） | 49.2 µs | **23.4 µs（2.10×）** | `kernel_microbench.json`（`pointwise_torch_ms` / `pointwise_triton_ms`） |
| DRAM 流量（**推算**，非实测；4 MiB/tensor） | 读 44 + 写 32 = **76 MiB** | 读 8 + 写 8 = **16 MiB**（**4.75×↓**） | 按上表 8 步逐个"读入+写出"累加 vs 单趟 |
| launch 开销 | 8 次 launch ×（trace 段 `cudaLaunchKernel` avg 5.3 µs / med 4.3 µs） | **1 次** | `v7_triton_rerun_cuda_api_sum.txt` |
| 真实流水线里（nsys，in-pipeline） | —（v1 的 eager 链没有单独 trace） | **7,861 ns** avg / 7,840 med，整类只占 **0.3%** GPU 时间；CUDA 版 7,526 ns 几乎持平 | `v7_triton_rerun_cuda_gpu_kern_sum.txt`（对账表 ✅） |

**为什么实测只有 2.1× 而流量推算是 4.75×**：eager 链的工作集（~5×4 MiB）能命中 4090 的 L2，
中间量并非每次都真落到 HBM——所以流量是理论下界、时间是含 L2 命中的实测。方向一致，数量级解释得通，
这也是"没有 NCU 就不把推算说成实测"的原因。

### 2.2 第二个数据点：fuse4（v8 bigbuf，`v8_ablation.json`，canvas 4096²）

| 后端 | fuse4 µs/call | vs eager | VRAM | 数值 |
|---|---:|---:|---:|---|
| pytorch_inplace（accumulator + count buffer + 多次 `add_`，N kernel） | 179.8 | 1× | 134.5 MB | diff=0 |
| **triton_bigbuf（1 kernel）** | **46.0** | **3.9×** | 96.0 MB | diff=0 |
| cuda_bigbuf（性能上限参考） | 29.2 | 6.2× | 96.0 MB | 基准 |

→ 同一个模式出现两次：**N kernel → 1 kernel，时间 2~4×↓，launch 次数线性↓，缓冲占用同步↓**。

### 2.3 这段故事的诚实边界

- "DRAM traffic ↓" 是**形状推算**（仓库无 NCU 数据）；实测支撑是同一负载的耗时下降（49.2→23.4 µs）。
- "8 kernels" 是从 committed 代码的表达式数的，不是 trace 数出来的（v1 时代的 eager 链没有单独 trace）。
- 隔离 launch 延迟是另一个测量域：Triton 单次 24.5 µs vs CUDA 8.1 µs（Python 分发开销），
  但 in-pipeline 重叠后 GPU 侧 7,861 vs 7,526 ns 几乎持平——**不要拿两个域的数字互相否定**（journey §11）。
- Triton 额外红利：launch 自带当前流，**默认 CUDA-Graph-safe**；手写 CUDA 要显式
  `getCurrentCUDAStream()`，否则 graph capture 时被静默跳过（5-mode：mode3 loss-rel=1.02 FAIL / mode5 PASS）。

---

## 3. 证据索引

| 文件 | 用到的数字 |
|---|---|
| `benchmarks/results/nsys_reports/v7_triton_rerun_cuda_gpu_kern_sum.txt` | FFT 链 50.0%（19.5+18.2+12.3）、fused pointwise 7,861 ns / 0.3%、总 GPU 3.096 s |
| `benchmarks/results/nsys_reports/v7_triton_rerun_cuda_api_sum.txt` | launch 160,796+67,988 次 / 1.10 s；`cudaMemcpyAsync` 1.448 s；`cudaStreamSynchronize` 1.831 s |
| `benchmarks/results/nsys_reports/v7_triton_rerun_cuda_gpu_mem_time_sum.txt` | D2H 0.966 s（91.4% mem-op）vs H2D 0.090 s |
| `benchmarks/results/nsys_reports/v7_p013_nvtx_cuda_gpu_kern_sum.txt` | post-P0.1：复数乘 0.4%、FFT 链 33.9% |
| `benchmarks/results/kernel_microbench.json` | fft2 0.0293 ms、ifft2_accum 1.2717 ms、pointwise 49.2→23.4 µs |
| `benchmarks/results/v8_ablation.json` | fuse4 179.8 / 46.0 / 29.2 µs，VRAM 134.5→96 MB，diff=0 |
| `benchmarks/results/headline_e2e.json` | v1 53.07 / v3 13.36 / v7 8.96 / P0.1 2.80 / P0.1+P0.3 2.36 s |
| `benchmarks/results/v7_profile.json` | graph replay 116.5 ms/iter |
| `benchmarks/results/graph_modes.json` | mode3 FAIL / mode5 PASS（stream bug 可控变量证明） |
| README §P0.5 | 256× H2D 占 H2D_Init 85% |
