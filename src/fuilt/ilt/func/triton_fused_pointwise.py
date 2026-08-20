"""Triton port of fused_pointwise_grad_loss_kernel.

Drop-in replacement for the CUDA kernel at fused_pointwise_kernel.cu.
Same inputs, same outputs, same fp32 dtype. Callable inside CUDA Graph
capture (after warmup triggers JIT compilation).
"""

from pathlib import Path
from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_pointwise_grad_loss_kernel(
    aerial_ptr,        # *fp32  [N]
    target_ptr,        # *fp32  [N]
    grad_ptr,          # *fp32  [N] (output)
    diff_sq_ptr,       # *fp32  [N] (output)
    n_elements,
    target_density: tl.constexpr,
    print_steepness: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute per-element sigmoid-print loss gradient + diff^2.

    For each element i:
        z_i             = print_steepness * (aerial_i - target_density)
        printed_i       = sigmoid(z_i)
        diff_i          = printed_i - target_i
        grad_i          = dLoss/dAerial_i = 2 * diff_i * print_steepness * printed_i * (1 - printed_i)
        diff_sq_i       = diff_i ** 2
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(aerial_ptr + offsets, mask=mask, other=0.0)
    t = tl.load(target_ptr + offsets, mask=mask, other=0.0)

    # ============================================================
    # TODO(you): implement the per-element math described above.
    #
    # Constraints:
    #   - All ops are vectorized over the BLOCK_SIZE-wide tile.
    #   - Use `tl.sigmoid(x)` (or `1.0 / (1.0 + tl.exp(-x))` if you prefer).
    #   - `target_density` and `print_steepness` are compile-time
    #     constants (tl.constexpr) — feel free to use them directly.
    #   - Final stores must write into grad_ptr and diff_sq_ptr using
    #     the same `mask=mask` you used for the loads.
    #
    # Trade-offs to consider:
    #   - `tl.sigmoid` is one PTX instruction on Ada/Hopper (`sigmoid.f32`).
    #     Manual `1/(1+exp(-x))` is two ops. Pick whichever you find clearer;
    #     the compiler will likely fuse either way.
    #   - Pre-compute `printed * (1 - printed)` once and reuse — Triton's
    #     compiler is good but not magical about CSE across ops with
    #     fpUnsafeMath-style optimizations.
    #
    # Aim: 4–6 lines of code, producing two SSA values `grad` and `diff_sq`.
    # ============================================================
    z = print_steepness * (a - target_density)
    printed = tl.sigmoid(z)
    diff = printed - t
    grad = 2.0 * diff * print_steepness * printed * (1.0 - printed)
    diff_sq = diff * diff
    # When you've filled it in, uncomment the two stores below:
    tl.store(grad_ptr + offsets, grad, mask=mask)
    tl.store(diff_sq_ptr + offsets, diff_sq, mask=mask)


def fused_pointwise_forward_triton(
    aerial: torch.Tensor,
    target: torch.Tensor,
    target_density: float = 0.5,
    print_steepness: float = 1.0,
    block_size: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Drop-in Triton replacement for the CUDA `fused_pointwise_forward`.

    Same shape/dtype/device contract as the CUDA version. Accepts any shape
    (flattens internally), returns outputs reshaped to `aerial.shape`.
    """
    assert aerial.is_cuda and target.is_cuda, "inputs must be CUDA tensors"
    assert aerial.dtype == torch.float32, f"aerial must be fp32, got {aerial.dtype}"
    assert target.dtype == torch.float32, f"target must be fp32, got {target.dtype}"
    assert aerial.shape == target.shape, f"shape mismatch: {aerial.shape} vs {target.shape}"
    assert (block_size & (block_size - 1)) == 0, "block_size must be a power of 2"

    aerial_flat = aerial.contiguous().flatten()
    target_flat = target.contiguous().flatten()
    grad_out = torch.empty_like(aerial_flat)
    diff_sq_out = torch.empty_like(aerial_flat)

    n = aerial_flat.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    _fused_pointwise_grad_loss_kernel[grid](
        aerial_flat,
        target_flat,
        grad_out,
        diff_sq_out,
        n,
        target_density,
        print_steepness,
        BLOCK_SIZE=block_size,
    )

    return grad_out.view_as(aerial), diff_sq_out.view_as(aerial)


@torch.no_grad()
def verify_vs_cuda(
    n: int = 1024 * 1024,
    target_density: float = 0.5,
    print_steepness: float = 1.0,
    atol: float = 1e-5,
    block_size: int = 1024,
) -> dict:
    """Numerical equivalence test: Triton output vs reference CUDA kernel.

    Compiles the original `fused_pointwise_kernel.cu` as the oracle, then
    checks max abs diff on both outputs against the Triton port.
    """
    from torch.utils.cpp_extension import load

    cuda_src = Path(__file__).parent / "fused_pointwise_kernel.cu"
    ext = load(
        name="fused_pointwise_ref",
        sources=[str(cuda_src)],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=False,
    )

    torch.manual_seed(0)
    aerial = torch.randn(n, device="cuda", dtype=torch.float32)
    target = torch.rand(n, device="cuda", dtype=torch.float32)

    grad_cuda, diff_sq_cuda = ext.fused_pointwise_forward(
        aerial, target, target_density, print_steepness
    )
    grad_triton, diff_sq_triton = fused_pointwise_forward_triton(
        aerial, target, target_density, print_steepness, block_size=block_size
    )

    grad_diff = (grad_cuda - grad_triton).abs().max().item()
    diff_sq_diff = (diff_sq_cuda - diff_sq_triton).abs().max().item()

    return {
        "n_elements": n,
        "block_size": block_size,
        "grad_max_abs_diff": grad_diff,
        "diff_sq_max_abs_diff": diff_sq_diff,
        "grad_pass": grad_diff < atol,
        "diff_sq_pass": diff_sq_diff < atol,
        "overall_pass": (grad_diff < atol) and (diff_sq_diff < atol),
    }


if __name__ == "__main__":
    result = verify_vs_cuda()
    print("=" * 60)
    print("Triton port vs CUDA reference (numerical equivalence)")
    print("=" * 60)
    for k, v in result.items():
        print(f"  {k:<22}: {v}")
    print("=" * 60)
    print("PASS" if result["overall_pass"] else "FAIL")
