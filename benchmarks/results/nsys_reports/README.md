# nsys derived reports

Machine-readable reports regenerated **2026-08-04** from the **committed** traces in
`outputs/nsys_traces/*.nsys-rep` (which are tracked in git). No GPU needed — pure
`nsys stats` computation on the existing `.nsys-rep` files.

Command per trace `<t>`:

```bash
nsys stats --report cuda_gpu_kern_sum  outputs/nsys_traces/<t>.nsys-rep > nsys_reports/<t>_cuda_gpu_kern_sum.txt
nsys stats --report cuda_gpu_mem_time_sum outputs/nsys_traces/<t>.nsys-rep > nsys_reports/<t>_cuda_gpu_mem_time_sum.txt
nsys stats --report cuda_api_sum       outputs/nsys_traces/<t>.nsys-rep > nsys_reports/<t>_cuda_api_sum.txt
```

`gputrace` (a legacy Nsight Systems report name) does not exist in nsys 2025.3.2 —
the per-kernel detail it provided is fully covered by `cuda_gpu_kern_sum` (per-kernel
Avg/Med/Min/Max/StdDev + total time).

## Files

| File | Contents |
|---|---|
| `*_cuda_gpu_kern_sum.txt` | Per-kernel GPU time share + instances + latency (Avg/Med/Min/Max/StdDev) |
| `*_cuda_gpu_mem_time_sum.txt` | CUDA memory operation time summary |
| `*_cuda_api_sum.txt` | CUDA driver/runtime API time summary (launch, memcpy, alloc…) |

## Ledger cross-check (README §What the trace tells us)

All numbers below were **re-derived from the committed traces** and reconcile the
README claims:

| README claim | Derived value | Source trace | Status |
|---|---|---|---|
| pointwise kernel **0.3%** of GPU time | 0.30 % | `v7_triton_rerun` kern_sum | ✅ |
| Triton pointwise per-kernel **7861 ns** | **7,861** ns (avg) / 7,840 ns (med) | `v7_triton_rerun` kern_sum | ✅ |
| CUDA pointwise per-kernel **7526 ns** | **7,525.7** ns (avg) / 7,616 ns (med) | `v7_cuda_rerun` kern_sum | ✅ |
| v7-CUDA total GPU time **3.075 s** | **3.0746** s | `v7_cuda_rerun` kern_sum (Σ rows) | ✅ |
| v7-Triton total GPU time **3.096 s** | **3.0964** s | `v7_triton_rerun` kern_sum (Σ rows) | ✅ |
| FFT chain **49.9 %** | 19.5 + 18.2 + 12.3 = **50.0 %** | `v7_triton_rerun` kern_sum | ⚠ rounding — text updated to 50.0 % |
| FFT chain **after P0.1** | `regular_fft` 17.2 % + `vector_fft` 16.7 % = **33.9 %** (complex multiply folded into combined kernel: residual BinaryFunctor-complex 0.4 %) | `v7_p013_nvtx` kern_sum (P0.1+P0.3 trace) | added 2026-08-04 — post-P0.1 timeline |

> **Why 49.9 vs 50.0:** the README table rows (19.5 % / 18.2 % / 12.3 % = the FFT
> forward + IFFT + complex multiply) come from the **GPU-isolated rerun** trace
> `v7_triton_rerun`, whose sum is exactly 50.0 %. The prose "49.9 %" came from the
> non-rerun `v7_triton` trace (19.4 + 18.2 + 12.3 = 49.9). The prose is updated to
> 50.0 % to match the table's cited source trace. That 50.0 % is the **pre-P0.1**
> picture; the P0.1+P0.3 trace `v7_p013_nvtx` puts the FFT chain at ~34 % (see
> the row above).

Also verified across the non-rerun pair (`v7_triton` = 3.0947 s, `v7_cuda` = 5.5192 s):
the two traces are **not** the same config — `v7_cuda.nsys-rep` predates the
GPU-isolated rerun (the pair that README §Trace cites is the `_rerun` pair).
