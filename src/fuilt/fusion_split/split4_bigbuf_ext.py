import torch
from torch.utils.cpp_extension import load_inline

# 简单平均版本的 split：从父块(bigbuf)按几何逆映射切回 A/B/C/D 四个子块
# 约定：
# - 父块尺寸 S_out = S + P (P = S - O)
# - A/B/C/D 子块尺寸 S
# - A 位于父块局部 [0:S, 0:S]
# - B 位于父块局部 [0:S, P:P+S]
# - C 位于父块局部 [P:P+S, 0:S]
# - D 位于父块局部 [P:P+S, P:P+S]
#
# split 的目的：把父块对应区域拷贝回四个子块 bigbuf 的对应位置（不做额外缩放/权重），
# 让下一步 tile/子块更新时在重叠区共享同一个“共识值”。

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t cast_from_float(float v);

template <>
__device__ __forceinline__ at::Half cast_from_float<at::Half>(float v) {
    // 安全转换：不做指针重解释
    return static_cast<at::Half>(v);
}

template <>
__device__ __forceinline__ float cast_from_float<float>(float v) {
    return v;
}

template <typename scalar_t_in>
__device__ __forceinline__ float load_in_as_float(const scalar_t_in* p) {
    return static_cast<float>(*p);
}

template <>
__device__ __forceinline__ float load_in_as_float<at::Half>(const at::Half* p) {
    return __half2float(*reinterpret_cast<const __half*>(p));
}

template <typename scalar_t_in, typename scalar_t_out>
__global__ void split4_to_bigbuf_kernel(
    const scalar_t_in* __restrict__ in,      // parent bigbuf [H,W]
    scalar_t_out* __restrict__ out,          // child bigbuf [H,W] (write A/B/C/D into this bigbuf)
    int H, int W,
    // input parent block top-left
    int ix0, int iy0,
    // output A/B/C/D top-left
    int ax0, int ay0,
    int bx0, int by0,
    int cx0, int cy0,
    int dx0, int dy0,
    int S, int P, int O,
    int S_out
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x; // x within child
    int u = blockIdx.y * blockDim.y + threadIdx.y; // y within child
    if (u >= S || v >= S) return;

    // 父块坐标（bigbuf）
    int pa_x = ix0 + v;
    int pa_y = iy0 + u;

    int pb_x = ix0 + (P + v);
    int pb_y = iy0 + u;

    int pc_x = ix0 + v;
    int pc_y = iy0 + (P + u);

    int pd_x = ix0 + (P + v);
    int pd_y = iy0 + (P + u);

    // 边界检查（bigbuf边界）
    if ((unsigned)pa_x >= (unsigned)W || (unsigned)pa_y >= (unsigned)H) return;
    if ((unsigned)pb_x >= (unsigned)W || (unsigned)pb_y >= (unsigned)H) return;
    if ((unsigned)pc_x >= (unsigned)W || (unsigned)pc_y >= (unsigned)H) return;
    if ((unsigned)pd_x >= (unsigned)W || (unsigned)pd_y >= (unsigned)H) return;

    float va = load_in_as_float<scalar_t_in>(in + pa_y * W + pa_x);
    float vb = load_in_as_float<scalar_t_in>(in + pb_y * W + pb_x);
    float vc = load_in_as_float<scalar_t_in>(in + pc_y * W + pc_x);
    float vd = load_in_as_float<scalar_t_in>(in + pd_y * W + pd_x);

    // 子块坐标（bigbuf）
    int oa_x = ax0 + v;
    int oa_y = ay0 + u;
    int ob_x = bx0 + v;
    int ob_y = by0 + u;
    int oc_x = cx0 + v;
    int oc_y = cy0 + u;
    int od_x = dx0 + v;
    int od_y = dy0 + u;

    if ((unsigned)oa_x < (unsigned)W && (unsigned)oa_y < (unsigned)H) out[oa_y * W + oa_x] = cast_from_float<scalar_t_out>(va);
    if ((unsigned)ob_x < (unsigned)W && (unsigned)ob_y < (unsigned)H) out[ob_y * W + ob_x] = cast_from_float<scalar_t_out>(vb);
    if ((unsigned)oc_x < (unsigned)W && (unsigned)oc_y < (unsigned)H) out[oc_y * W + oc_x] = cast_from_float<scalar_t_out>(vc);
    if ((unsigned)od_x < (unsigned)W && (unsigned)od_y < (unsigned)H) out[od_y * W + od_x] = cast_from_float<scalar_t_out>(vd);
}

torch::Tensor split4_to_bigbuf(
    torch::Tensor in_big,     // [H,W], fp16/fp32 (parent fused field)
    torch::Tensor out_big,    // [H,W], fp16/fp32 (children destination bigbuf)
    torch::Tensor offsets,    // [5,2] int32 CPU: A,B,C,D,I(parent) offsets
    int S,
    int O
) {
    TORCH_CHECK(in_big.is_cuda() && out_big.is_cuda(), "in_big/out_big must be CUDA tensors");
    TORCH_CHECK(in_big.dim() == 2 && out_big.dim() == 2, "in_big/out_big must be 2D");
    TORCH_CHECK(in_big.is_contiguous() && out_big.is_contiguous(), "in_big/out_big must be contiguous");
    TORCH_CHECK(!offsets.is_cuda(), "offsets must be CPU tensor (int32)");
    TORCH_CHECK(offsets.dim() == 2 && offsets.size(0) == 5 && offsets.size(1) == 2, "offsets shape must be [5,2]");
    TORCH_CHECK(offsets.scalar_type() == torch::kInt32, "offsets must be int32 CPU tensor");
    TORCH_CHECK(in_big.size(0) == out_big.size(0) && in_big.size(1) == out_big.size(1), "in/out bigbuf shape mismatch");

    int H = (int)in_big.size(0);
    int W = (int)in_big.size(1);

    int ax0 = offsets[0][0].item<int>();
    int ay0 = offsets[0][1].item<int>();
    int bx0 = offsets[1][0].item<int>();
    int by0 = offsets[1][1].item<int>();
    int cx0 = offsets[2][0].item<int>();
    int cy0 = offsets[2][1].item<int>();
    int dx0 = offsets[3][0].item<int>();
    int dy0 = offsets[3][1].item<int>();
    int ix0 = offsets[4][0].item<int>();
    int iy0 = offsets[4][1].item<int>();

    int P = S - O;
    int S_out = S + P;

    dim3 block(16, 16);
    dim3 grid((S + block.x - 1) / block.x, (S + block.y - 1) / block.y);

    // Launch on the current (capture) stream so the kernel is recorded into
    // any enclosing torch.cuda.graph() — getDefaultCUDAStream() (stream 0) is
    // NOT the capture stream and would be silently skipped on replay.
    auto stream = at::cuda::getCurrentCUDAStream();

    if (out_big.scalar_type() == torch::kFloat16) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(in_big.scalar_type(), "split4_to_bigbuf_kernel", [&]{
            split4_to_bigbuf_kernel<scalar_t, at::Half><<<grid, block, 0, stream>>>(
                (const scalar_t*)in_big.data_ptr<scalar_t>(),
                (at::Half*)out_big.data_ptr<at::Half>(),
                H, W,
                ix0, iy0,
                ax0, ay0, bx0, by0, cx0, cy0, dx0, dy0,
                S, P, O, S_out
            );
        });
    } else if (out_big.scalar_type() == torch::kFloat32) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(in_big.scalar_type(), "split4_to_bigbuf_kernel", [&]{
            split4_to_bigbuf_kernel<scalar_t, float><<<grid, block, 0, stream>>>(
                (const scalar_t*)in_big.data_ptr<scalar_t>(),
                (float*)out_big.data_ptr<float>(),
                H, W,
                ix0, iy0,
                ax0, ay0, bx0, by0, cx0, cy0, dx0, dy0,
                S, P, O, S_out
            );
        });
    } else {
        TORCH_CHECK(false, "out_big dtype must be fp16 or fp32");
    }

    return out_big;
}
"""

CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor split4_to_bigbuf(torch::Tensor in_big, torch::Tensor out_big, torch::Tensor offsets, int S, int O);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("split4_to_bigbuf", &split4_to_bigbuf, "Split 1 parent block in bigbuf back into 4 child blocks in bigbuf (simple copy)");
}
"""

_ext = None

def load_ext(verbose: bool = False):
    global _ext
    if _ext is not None:
        return _ext
    _ext = load_inline(
        name="split4_bigbuf_ext",
        cpp_sources=CPP_SRC,
        cuda_sources=CUDA_SRC,
        functions=None,
        extra_cuda_cflags=["--use_fast_math"],
        extra_cflags=[],
        with_cuda=True,
        verbose=verbose,
    )
    return _ext

@torch.no_grad()
def split4_to_bigbuf(in_big: torch.Tensor, out_big: torch.Tensor, offsets_cpu_i32: torch.Tensor, S: int, O: int) -> torch.Tensor:
    """
    offsets_cpu_i32: [5,2] CPU int32
        0:A(x0,y0) 1:B 2:C 3:D 4:I(parent input top-left)
    将 parent(I) 的四个象限（含 overlap 的布局）拷贝回 A/B/C/D 子块位置。
    """
    ext = load_ext(verbose=False)
    return ext.split4_to_bigbuf(in_big, out_big, offsets_cpu_i32, int(S), int(O))