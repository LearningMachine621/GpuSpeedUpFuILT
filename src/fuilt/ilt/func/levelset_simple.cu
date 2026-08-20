#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cmath>

// 为 1024x1024 的图像块和 35x35 的卷积核调优的启动参数。
#define BLOCK_SIZE 16
#define KERNEL_RADIUS 17
#define KERNEL_DIM 35
#define SHARED_DIM (BLOCK_SIZE + 2 * KERNEL_RADIUS)

// 一些用于检查输入数据格式的宏定义，确保送进来的都是合法的 GPU Tensor
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " 必须是 CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " 必须在内存中是连续的")
#define CHECK_FLOAT32(x) TORCH_CHECK((x).scalar_type() == at::kFloat, #x " 必须是 float32 类型")

namespace {

// 融合算子（把好几步计算揉在一起，为了跑得快）：
// 1) 在共享内存中对输入参数 params_in 进行切块操作，并用 K 个卷积核进行卷积（也就是计算模糊效果）。
// 2) 应用 Sigmoid 打印模型（模拟曝光/冲洗等物理化学过程，把连续值变成接近0或1的值）。
// 3) 计算相对于空中图像（aerial image）的 L2 损失梯度（也就是算一下为了让打印结果更接近目标，该怎么调整输入）。
__global__ void FusedComputeGradientKernel(
        const float* __restrict__ params_in,    // 输入的参数（比如光罩上的图案）
        const float* __restrict__ target_mask,  // 我们最终想要打印出的目标图案
        float* __restrict__ grad_out,           // 输出：梯度（告诉我们怎么修改 params_in）
        const float* __restrict__ opt_kernels,  // 卷积核（模拟光学模糊的滤镜）
        const float* __restrict__ kernel_scales,// 多个卷积核的权重比例
        int kernel_count,                       // 卷积核的数量
        int width,                              // 图像宽度
        int height,                             // 图像高度
        float target_density,                   // 目标密度（判定是否能打印出来的阈值）
        float print_steepness) {                // 打印陡峭度（Sigmoid的参数，控制黑白分明程度）
    
    // 获取当前线程和线程块的索引
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * BLOCK_SIZE;
    int by = blockIdx.y * BLOCK_SIZE;

    // 计算当前线程负责的全局像素坐标
    int gx = bx + tx;
    int gy = by + ty;

    // 分配共享内存（相当于这组线程的“公共黑板”，比直接读显存快得多）
    __shared__ float s_params[SHARED_DIM][SHARED_DIM];

    // 团队协作：把这个图像块 + 周围一圈“光环”（halo/边缘）一起加载到共享内存里
    const int tid = ty * BLOCK_SIZE + tx;
    const int threads_per_block = BLOCK_SIZE * BLOCK_SIZE;
    const int shared_size = SHARED_DIM * SHARED_DIM;
    for (int i = tid; i < shared_size; i += threads_per_block) {
        const int sy = i / SHARED_DIM;
        const int sx = i % SHARED_DIM;

        int global_y = by - KERNEL_RADIUS + sy;
        int global_x = bx - KERNEL_RADIUS + sx;

        // 处理边界情况：如果超出了图像范围，就取边缘像素（clamp 操作）
        global_y = max(0, min(global_y, height - 1));
        global_x = max(0, min(global_x, width - 1));
        s_params[sy][sx] = params_in[global_y * width + global_x];
    }

    // 等待所有线程都把数据搬完，黑板写满了再继续往下走
    __syncthreads();

    // 如果当前线程对应的坐标超出了图像实际大小，直接下班不干了
    if (gx >= width || gy >= height) {
        return;
    }

    float aerial_nom = 0.0f; // 用来存所有卷积核计算出来的光学图像总和

    // 多核累加卷积计算（计算光是怎么模糊的）
    for (int k = 0; k < kernel_count; ++k) {
        float conv_val = 0.0f;
        const int kernel_offset = k * KERNEL_DIM * KERNEL_DIM;

        // 展开循环以提升性能，遍历 35x35 的卷积核窗口
        for (int ky = -KERNEL_RADIUS; ky <= KERNEL_RADIUS; ++ky) {
            #pragma unroll
            for (int kx = -KERNEL_RADIUS; kx <= KERNEL_RADIUS; ++kx) {
                // 从刚刚搬到共享内存里的数据取像素值，乘以对应的卷积核权重
                const float pixel = s_params[ty + KERNEL_RADIUS + ky][tx + KERNEL_RADIUS + kx];
                const float weight = opt_kernels[kernel_offset + (ky + KERNEL_RADIUS) * KERNEL_DIM + (kx + KERNEL_RADIUS)];
                conv_val += pixel * weight;
            }
        }

        // 把这个卷积核的结果乘上它的比例系数，加到总强度里
        const float scale = (kernel_scales != nullptr) ? kernel_scales[k] : 1.0f;
        aerial_nom += scale * conv_val;
    }

    // 计算当前像素在内存里的一维位置
    const int idx = gy * width + gx;
    const float target_val = target_mask[idx]; // 获取我们希望打印出的标准答案

    // 模拟光刻显影物理过程：用 Sigmoid 激活函数算出实际印出来的结果
    const float z = print_steepness * (aerial_nom - target_density);
    const float printed_nom = 1.0f / (1.0f + expf(-z));

    // 计算梯度（核心目的：求导数，知道怎么调整输入能让误差变小）
    // L2 损失函数对于 printed_nom 的导数：d/dx (x-y)^2 = 2(x-y)
    const float dL_dprinted = 2.0f * (printed_nom - target_val);
    
    // Sigmoid 函数本身的导数
    const float dprinted_daerial = print_steepness * printed_nom * (1.0f - printed_nom);
    
    // 链式法则相乘，得到最终相对于输入图像的梯度，存到输出结果中
    grad_out[idx] = dL_dprinted * dprinted_daerial;
}

}  // namespace

// C++ 外壳函数：负责做安全检查，并把工作派发给 GPU (CUDA) 去做
torch::Tensor levelset_step_cuda(
        torch::Tensor params_in,
        torch::Tensor target_mask,
        torch::Tensor opt_kernels,
        c10::optional<torch::Tensor> kernel_scales,
        double target_density,
        double print_steepness) {
    // 一顿严苛的安检：确保传进来的都是存在 GPU 上、连续且是 float32 格式的数据
    CHECK_CUDA(params_in);
    CHECK_CUDA(target_mask);
    CHECK_CUDA(opt_kernels);
    CHECK_CONTIGUOUS(params_in);
    CHECK_CONTIGUOUS(target_mask);
    CHECK_CONTIGUOUS(opt_kernels);
    CHECK_FLOAT32(params_in);
    CHECK_FLOAT32(target_mask);
    CHECK_FLOAT32(opt_kernels);

    // 检查形状是不是符合要求（必须是二维图，必须一样大，卷积核必须是 35x35）
    TORCH_CHECK(params_in.dim() == 2, "params_in 必须是 [H, W] 二维的");
    TORCH_CHECK(target_mask.dim() == 2, "target_mask 必须是 [H, W] 二维的");
    TORCH_CHECK(params_in.sizes() == target_mask.sizes(), "params_in 和 target_mask 尺寸不匹配");

    TORCH_CHECK(opt_kernels.dim() == 3, "opt_kernels 必须是 [K, KH, KW] 三维的");
    TORCH_CHECK(opt_kernels.size(1) == KERNEL_DIM && opt_kernels.size(2) == KERNEL_DIM,
                            "opt_kernels 的卷积核尺寸必须是 35x35");

    // 获取有几个卷积核，并防止溢出
    const int64_t kernel_count_64 = opt_kernels.size(0);
    TORCH_CHECK(kernel_count_64 > 0, "opt_kernels 至少得包含一个卷积核");
    TORCH_CHECK(kernel_count_64 <= INT_MAX, "kernel_count 数量溢出");
    const int kernel_count = static_cast<int>(kernel_count_64);

    // 如果传了 scales（各个滤镜的权重），也给它做个安检并拿到指针
    const float* scale_ptr = nullptr;
    if (kernel_scales.has_value()) {
        const torch::Tensor scales = kernel_scales.value();
        CHECK_CUDA(scales);
        CHECK_CONTIGUOUS(scales);
        CHECK_FLOAT32(scales);
        TORCH_CHECK(scales.dim() == 1, "kernel_scales 必须是一维的 [K]");
        TORCH_CHECK(scales.size(0) == kernel_count_64, "kernel_scales 长度必须和卷积核数量一致");
        scale_ptr = scales.data_ptr<float>();
    }

    // 拿到图像的高和宽
    const int64_t height_64 = params_in.size(0);
    const int64_t width_64 = params_in.size(1);
    TORCH_CHECK(height_64 > 0 && width_64 > 0, "输入 tensor 不能是空的");
    TORCH_CHECK(height_64 <= INT_MAX && width_64 <= INT_MAX, "输入尺寸溢出");
    const int height = static_cast<int>(height_64);
    const int width = static_cast<int>(width_64);

    // 准备好一个空白的输出容器，大小和输入一模一样
    auto grad_out = torch::zeros_like(params_in);

    // 设置 GPU 上的打工人编队：每 32x32 个线程为一个小组（Block）
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    // 计算一共需要多少个这样的小组才能把整张图覆盖满
    const dim3 blocks(
            (width + threads.x - 1) / threads.x,
            (height + threads.y - 1) / threads.y);

    // 大喊一声：兄弟们，开干！(启动 CUDA kernel)
    // 显式在当前 (capture) stream 上启动，使本 kernel 能被 torch.cuda.graph()
    // 正确捕获；默认 <<<>>> 走 stream 0，capture 时会被静默跳过。
    auto stream = at::cuda::getCurrentCUDAStream();
    FusedComputeGradientKernel<<<blocks, threads, 0, stream>>>(
            params_in.data_ptr<float>(),
            target_mask.data_ptr<float>(),
            grad_out.data_ptr<float>(),
            opt_kernels.data_ptr<float>(),
            scale_ptr,
            kernel_count,
            width,
            height,
            static_cast<float>(target_density),
            static_cast<float>(print_steepness));

    // 检查一下运行过程中有没有出错
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    // 把结果返回给 Python 端
    return grad_out;
}

// Pybind11 魔法：把 C++ 的函数打包，让 Python 可以直接像调包一样调用它
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
            "levelset_step",
            &levelset_step_cuda,
            "Fused LevelSet gradient step (CUDA)",
            py::arg("params_in"),
            py::arg("target_mask"),
            py::arg("opt_kernels"),
            py::arg("kernel_scales") = c10::nullopt,
            py::arg("target_density") = 0.5,
            py::arg("print_steepness") = 1.0);
}