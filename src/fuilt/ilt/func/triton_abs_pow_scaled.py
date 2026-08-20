"""Triton port of the `_computeImage` final reduction.

Replaces:
    torch.sum(scale * torch.pow(torch.abs(tmp), 2), dim=<K axis>)

with a single Triton kernel that fuses abs + pow + scale-mul + sum-reduce
into one pass over the [K, H, W] complex tensor.

Math:
    out[h, w] = sum_k(scale[k] * (tmp[k,h,w].real^2 + tmp[k,h,w].imag^2))

This kernel does NOT appear in the v7 main loop hot path — it's used by
`_LithoSim._computeImage` during evaluation. The nsys trace showed
`abs_kernel_vectorized2` at 10.8% + `pow_tensor_scalar_kernel` at 5.3%
of total GPU time, but those came from eval, not per-iter compute.

Still worth porting because:
  1. It's a textbook reduction-kernel exercise (Triton tl.sum over a K-loop)
  2. It exercises the rare [K, H, W] strided gather pattern
  3. Eval time matters for development iteration speed
"""

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _abs_pow_scale_sum_kernel(
    real_ptr,         # *fp32 [K, H, W]
    imag_ptr,         # *fp32 [K, H, W]
    scale_ptr,        # *fp32 [K]
    out_ptr,          # *fp32 [H, W]
    H, W,
    K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """For each spatial (h, w), accumulate sum_k(scale[k] * (real^2 + imag^2))."""
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)

    # 2D spatial coordinates within this program's tile
    h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)   # [BLOCK_H]
    w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)   # [BLOCK_W]

    # Broadcast to 2D for indexing: h_2d[i, j] = h[i], w_2d[i, j] = w[j]
    h_2d = h[:, None]   # [BLOCK_H, 1]
    w_2d = w[None, :]   # [1, BLOCK_W]

    spatial_mask = (h_2d < H) & (w_2d < W)        # [BLOCK_H, BLOCK_W]

    # Accumulator in fp32 (matches PyTorch's reduction dtype)
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    # K is constexpr → compiler unrolls the loop, hoisting scale loads
    for k in range(K):
        s = tl.load(scale_ptr + k)   # scalar fp32

        # Strided addressing into [K, H, W] layout: offset = k*H*W + h*W + w
        offsets = k * H * W + h_2d * W + w_2d   # [BLOCK_H, BLOCK_W]

        r = tl.load(real_ptr + offsets, mask=spatial_mask, other=0.0)
        i = tl.load(imag_ptr + offsets, mask=spatial_mask, other=0.0)

        # |tmp|^2 = real^2 + imag^2, then weighted by scale
        acc += s * (r * r + i * i)

    # Store the reduced [H, W] output
    out_offsets = h_2d * W + w_2d
    tl.store(out_ptr + out_offsets, acc, mask=spatial_mask)


def abs_pow_scale_sum_triton(
    tmp: torch.Tensor,
    scale: torch.Tensor,
    block_h: int = 32,
    block_w: int = 32,
) -> torch.Tensor:
    """Drop-in Triton replacement for `torch.sum(scale * torch.pow(torch.abs(tmp), 2), dim=K_axis)`.

    Args:
        tmp:   [K, H, W] complex64 (or [B, K, H, W] — batch handled by loop).
        scale: [K] fp32, weights for the K-axis reduction.
        block_h, block_w: Triton spatial block sizes (powers of 2).

    Returns:
        [H, W] fp32 if tmp is 3D, [B, H, W] fp32 if tmp is 4D.
    """
    assert tmp.is_cuda and scale.is_cuda
    assert tmp.dtype == torch.complex64, f"tmp must be complex64, got {tmp.dtype}"
    assert scale.dtype == torch.float32, f"scale must be fp32, got {scale.dtype}"

    if tmp.dim() == 4:
        # Batched: loop over B in Python, run 3D kernel per batch
        B, K, H, W = tmp.shape
        outs = []
        for b in range(B):
            outs.append(abs_pow_scale_sum_triton(tmp[b], scale, block_h, block_w))
        return torch.stack(outs, dim=0)

    K, H, W = tmp.shape
    assert scale.shape == (K,), f"scale shape {scale.shape} != ({K},)"

    # Extract real/imag as separate fp32 tensors (Triton doesn't natively
    # support complex64; this is the standard workaround).
    real = tmp.real.contiguous()   # [K, H, W] fp32
    imag = tmp.imag.contiguous()   # [K, H, W] fp32
    out = torch.empty(H, W, device=tmp.device, dtype=torch.float32)

    grid = (triton.cdiv(H, block_h), triton.cdiv(W, block_w))

    _abs_pow_scale_sum_kernel[grid](
        real, imag, scale, out,
        H, W,
        K=K,
        BLOCK_H=block_h,
        BLOCK_W=block_w,
    )
    return out


@torch.no_grad()
def verify_vs_pytorch(
    K: int = 24,
    H: int = 512,
    W: int = 512,
    seed: int = 0,
    atol: float = 1e-4,
) -> dict:
    """Numerical equivalence test: Triton reduction vs PyTorch eager.

    The PyTorch reference is:
        torch.sum(scale * torch.pow(torch.abs(tmp), 2), dim=0)
    """
    torch.manual_seed(seed)
    tmp = torch.randn(K, H, W, device="cuda", dtype=torch.complex64)
    scale = torch.rand(K, device="cuda", dtype=torch.float32) * 2.0  # [0, 2)

    ref = torch.sum(scale.view(-1, 1, 1) * torch.pow(torch.abs(tmp), 2), dim=0)
    got = abs_pow_scale_sum_triton(tmp, scale)

    diff = (ref - got).abs()
    return {
        "shape": f"({K}, {H}, {W})",
        "ref_max": ref.abs().max().item(),
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "pass": diff.max().item() < atol,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Triton abs+pow+scale+sum reduction vs PyTorch reference")
    print("=" * 60)
    # Small case first (fast)
    r_small = verify_vs_pytorch(K=24, H=256, W=256)
    print(f"\nSmall case {r_small['shape']}:")
    for k, v in r_small.items():
        print(f"  {k:<16}: {v}")

    # Production-shape case (matches LithoSim tile)
    r_prod = verify_vs_pytorch(K=24, H=1024, W=1024)
    print(f"\nProduction case {r_prod['shape']}:")
    for k, v in r_prod.items():
        print(f"  {k:<16}: {v}")
    print("=" * 60)
    print("PASS" if r_prod["pass"] else "FAIL")
