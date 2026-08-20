#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be float32")

// ============================================================================
// CUDA Graph capture-safety toggle.
//
// Hand-written kernels that launch on `at::cuda::getDefaultCUDAStream()` (or
// the default `<<<grid, block>>>` form, which is stream 0) are silently NOT
// recorded into a graph captured via `torch.cuda.graph(g)`: capture happens on
// a private side stream, so the default-stream launch is skipped on replay,
// leaving the output in its pre-replay state (numerically wrong).
//
// Launching on `at::cuda::getCurrentCUDAStream()` instead fixes this: during
// capture that returns the capture stream (so the launch is recorded), and in
// eager mode it returns the current stream (identical behavior to before — no
// regression outside capture).
//
// This toggle exists so a single benchmark binary can reproduce BOTH the bug
// (capture_safe_stream=0 → wrong output) and the fix (=1 → correct), proving
// the root cause rather than just asserting it. Default is the fixed path.
// ============================================================================
static int g_capture_safe_stream = 1;

void set_capture_safe_stream(int v) {
    g_capture_safe_stream = v ? 1 : 0;
}

namespace {

__global__ void fused_pointwise_grad_loss_kernel(
    const float* __restrict__ aerial,
    const float* __restrict__ target,
    float* __restrict__ grad_out,
    float* __restrict__ diff_sq_out,
    int total_elements,
    float target_density,
    float print_steepness
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) {
        return;
    }

    float a = aerial[idx];
    float t = target[idx];

    float z = print_steepness * (a - target_density);
    float printed = 1.0f / (1.0f + __expf(-z));

    float diff = printed - t;
    float dL_dprinted = 2.0f * diff;
    float dprinted_daerial = print_steepness * printed * (1.0f - printed);

    grad_out[idx] = dL_dprinted * dprinted_daerial;
    diff_sq_out[idx] = diff * diff;
}

}  // namespace

std::vector<torch::Tensor> fused_pointwise_forward(
    torch::Tensor aerial,
    torch::Tensor target,
    double target_density,
    double print_steepness
) {
    CHECK_CUDA(aerial);
    CHECK_CUDA(target);
    CHECK_CONTIGUOUS(aerial);
    CHECK_CONTIGUOUS(target);
    CHECK_FLOAT32(aerial);
    CHECK_FLOAT32(target);
    TORCH_CHECK(aerial.sizes() == target.sizes(), "aerial and target must have the same shape");

    auto grad_out = torch::empty_like(aerial);
    auto diff_sq_out = torch::empty_like(aerial);

    const int64_t total_elements_64 = aerial.numel();
    TORCH_CHECK(total_elements_64 >= 0 && total_elements_64 <= INT_MAX, "tensor numel out of range");
    const int total_elements = static_cast<int>(total_elements_64);

    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    // Honor the PyTorch capture stream when capture-safe mode is on (default);
    // fall back to the legacy default stream to reproduce the pre-fix bug.
    auto stream = g_capture_safe_stream
        ? at::cuda::getCurrentCUDAStream()
        : at::cuda::getDefaultCUDAStream();

    fused_pointwise_grad_loss_kernel<<<blocks, threads, 0, stream>>>(
        aerial.data_ptr<float>(),
        target.data_ptr<float>(),
        grad_out.data_ptr<float>(),
        diff_sq_out.data_ptr<float>(),
        total_elements,
        static_cast<float>(target_density),
        static_cast<float>(print_steepness)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_out, diff_sq_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "fused_pointwise_forward",
        &fused_pointwise_forward,
        "Fused pointwise gradient/loss kernel (CUDA)",
        py::arg("aerial"),
        py::arg("target"),
        py::arg("target_density") = 0.5,
        py::arg("print_steepness") = 1.0
    );
    m.def(
        "set_capture_safe_stream",
        &set_capture_safe_stream,
        "Toggle capture-safe stream launch (1=fixed/current stream, 0=buggy/default stream)",
        py::arg("value")
    );
}