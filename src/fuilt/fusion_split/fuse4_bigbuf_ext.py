import torch
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
__device__ __forceinline__ float load_as_float(const scalar_t* p) {
    return static_cast<float>(*p);
}

template <>
__device__ __forceinline__ float load_as_float<at::Half>(const at::Half* p) {
    // at::Half 在CUDA端可与 __half 等价使用（经由 reinterpret）
    return __half2float(*reinterpret_cast<const __half*>(p));
}

__device__ __forceinline__ at::Half float_to_half(float v) {
    // 更“优雅/安全”的方式：用半精度转换指令
    return static_cast<at::Half>(v);
}

template <typename scalar_t>
__global__ void fuse4_from_bigbuf_kernel(
    const scalar_t* __restrict__ in,   // [H, W]
    at::Half* __restrict__ out,        // [H, W]
    int H, int W,
    int ax0, int ay0,
    int bx0, int by0,
    int cx0, int cy0,
    int dx0, int dy0,
    int ox0, int oy0,
    int S, int P, int O,
    int S_out
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // out-local x
    int y = blockIdx.y * blockDim.y + threadIdx.y; // out-local y
    if (x >= S_out || y >= S_out) return;

    int gx = ox0 + x;
    int gy = oy0 + y;
    if ((unsigned)gx >= (unsigned)W || (unsigned)gy >= (unsigned)H) return;

    float sum = 0.0f;
    int cnt = 0;

    // A: [0,S) x [0,S)
    if (x < S && y < S) {
        int ix = ax0 + x;
        int iy = ay0 + y;
        sum += load_as_float<scalar_t>(in + iy * W + ix);
        cnt++;
    }
    // B: [P,P+S) x [0,S)
    if (x >= P && x < P + S && y < S) {
        int ix = bx0 + (x - P);
        int iy = by0 + y;
        sum += load_as_float<scalar_t>(in + iy * W + ix);
        cnt++;
    }
    // C: [0,S) x [P,P+S)
    if (x < S && y >= P && y < P + S) {
        int ix = cx0 + x;
        int iy = cy0 + (y - P);
        sum += load_as_float<scalar_t>(in + iy * W + ix);
        cnt++;
    }
    // D: [P,P+S) x [P,P+S)
    if (x >= P && x < P + S && y >= P && y < P + S) {
        int ix = dx0 + (x - P);
        int iy = dy0 + (y - P);
        sum += load_as_float<scalar_t>(in + iy * W + ix);
        cnt++;
    }

    float outv = (cnt > 0) ? (sum / (float)cnt) : 0.0f;
    out[gy * W + gx] = float_to_half(outv);
}

torch::Tensor fuse4_from_bigbuf(
    torch::Tensor in_big,   // [H,W], fp16/fp32, contiguous
    torch::Tensor out_big,  // [H,W], fp16, contiguous (ping-pong)
    torch::Tensor offsets,  // [5,2] int32 CPU: A,B,C,D,O offsets
    int S,
    int O
) {
    TORCH_CHECK(in_big.is_cuda() && out_big.is_cuda(), "in_big/out_big must be CUDA tensors");
    TORCH_CHECK(in_big.dim() == 2 && out_big.dim() == 2, "in_big/out_big must be 2D");
    TORCH_CHECK(in_big.is_contiguous() && out_big.is_contiguous(), "in_big/out_big must be contiguous");
    TORCH_CHECK(!offsets.is_cuda(), "offsets must be CPU tensor (int32)");
    TORCH_CHECK(offsets.dim() == 2 && offsets.size(0) == 5 && offsets.size(1) == 2, "offsets shape must be [5,2]");
    TORCH_CHECK(offsets.scalar_type() == torch::kInt32, "offsets must be int32 CPU tensor");
    TORCH_CHECK(out_big.scalar_type() == torch::kFloat16, "out_big must be fp16 for this version");

    int H = (int)in_big.size(0);
    int W = (int)in_big.size(1);
    TORCH_CHECK(out_big.size(0) == H && out_big.size(1) == W, "in_big/out_big shape mismatch");

    int ax0 = offsets[0][0].item<int>();
    int ay0 = offsets[0][1].item<int>();
    int bx0 = offsets[1][0].item<int>();
    int by0 = offsets[1][1].item<int>();
    int cx0 = offsets[2][0].item<int>();
    int cy0 = offsets[2][1].item<int>();
    int dx0 = offsets[3][0].item<int>();
    int dy0 = offsets[3][1].item<int>();
    int ox0 = offsets[4][0].item<int>();
    int oy0 = offsets[4][1].item<int>();

    int P = S - O;
    int S_out = S + P;

    dim3 block(16, 16);
    dim3 grid((S_out + block.x - 1) / block.x, (S_out + block.y - 1) / block.y);

    // Launch on the current (capture) stream so the kernel is recorded into
    // any enclosing torch.cuda.graph() — getDefaultCUDAStream() (stream 0) is
    // NOT the capture stream and would be silently skipped on replay.
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(in_big.scalar_type(), "fuse4_from_bigbuf_kernel", [&]{
        fuse4_from_bigbuf_kernel<scalar_t><<<grid, block, 0, stream>>>(
            (const scalar_t*)in_big.data_ptr<scalar_t>(),
            (at::Half*)out_big.data_ptr<at::Half>(),
            H, W,
            ax0, ay0, bx0, by0, cx0, cy0, dx0, dy0,
            ox0, oy0,
            S, P, O, S_out
        );
    });

    return out_big;
}
"""

CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor fuse4_from_bigbuf(torch::Tensor in_big, torch::Tensor out_big, torch::Tensor offsets, int S, int O);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fuse4_from_bigbuf", &fuse4_from_bigbuf, "Fuse 4 sub-blocks from a big buffer into another big buffer (avg blend)");
}
"""

_ext = None

def load_ext(verbose: bool = False):
    global _ext
    if _ext is not None:
        return _ext

    _ext = load_inline(
        name="fuse4_bigbuf_ext",
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
def fuse4_from_bigbuf(in_big: torch.Tensor, out_big: torch.Tensor, offsets_cpu_i32: torch.Tensor, S: int, O: int) -> torch.Tensor:
    """
    offsets_cpu_i32: [5,2] CPU int32
        0:A(x0,y0) 1:B 2:C 3:D 4:O(output top-left)
    """
    ext = load_ext(verbose=False)
    return ext.fuse4_from_bigbuf(in_big, out_big, offsets_cpu_i32, int(S), int(O))