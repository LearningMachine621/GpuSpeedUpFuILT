"""Triton port of fuse4_from_bigbuf + split4_to_bigbuf kernels.

Drop-in replacements for the CUDA kernels in fuse4_bigbuf_ext.py and
split4_bigbuf_ext.py. Same inputs (in_big, out_big, offsets_cpu_i32, S, O),
same outputs, same dtypes.

These two kernels are 2D scatter/gather — significantly more complex than the
pointwise port in triton_fused_pointwise.py. Read docs/triton_tutorial_02_bigbuf.md
before filling in the TODO blocks.
"""

from pathlib import Path
from typing import Tuple

import torch
import triton
import triton.language as tl


# =============================================================================
# fuse4_from_bigbuf
# =============================================================================
# Geometry (matches fuse4_bigbuf_ext.py):
#   - Input `in_big` is a [H, W] buffer containing 4 sub-blocks A, B, C, D.
#   - Output `out_big` is a [H, W] buffer; we write into the S_out × S_out region
#     starting at (ox0, oy0).
#   - Each output pixel (x, y) in the S_out × S_out region averages the
#     sub-blocks that cover it. Coverage map (output-local coords):
#         A: [0, S) × [0, S)
#         B: [P, P+S) × [0, S)
#         C: [0, S) × [P, P+S)
#         D: [P, P+S) × [P, P+S)
#     where P = S - O (stride), S_out = S + P.
#   - Output dtype is fp16 (matches the CUDA kernel's hardcoded `at::Half`).
# =============================================================================

@triton.jit
def _fuse4_from_bigbuf_kernel(
    in_ptr,            # *fp32 or *fp16, [H * W]
    out_ptr,           # *fp16, [H * W]
    H, W,
    ax0, ay0, bx0, by0, cx0, cy0, dx0, dy0,
    ox0, oy0,
    S,
    P,                 # P = S - O
    S_OUT,             # S_OUT = S + P
    BLOCK: tl.constexpr,
):
    """Fuse 4 sub-blocks (A, B, C, D) into the output region via averaging."""
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)            # 1D index into S_OUT*S_OUT
    n_out = S_OUT * S_OUT

    # Output-local 2D coordinates
    x = idx % S_OUT
    y = idx // S_OUT

    # Global output coordinates
    gx = ox0 + x
    gy = oy0 + y

    # Output-of-interest mask: idx in range AND global coords in [0, H) × [0, W)
    out_mask = (idx < n_out) & (gx < W) & (gy < H)

    # 4 coverage masks (output-local). Each is True where that sub-block
    # contributes to this output pixel.
    mask_A = (x < S) & (y < S)
    mask_B = (x >= P) & (x < P + S) & (y < S)
    mask_C = (x < S) & (y >= P) & (y < P + S)
    mask_D = (x >= P) & (x < P + S) & (y >= P) & (y < P + S)

    # Load each sub-block's contribution (masked positions read 0.0).
    # The arithmetic here is the CUDA kernel's `sum += load_as_float(...)` pattern.
    a = tl.load(in_ptr + (ay0 + y) * W + (ax0 + x),         mask=mask_A, other=0.0)
    b = tl.load(in_ptr + (by0 + y) * W + (bx0 + x - P),     mask=mask_B, other=0.0)
    c = tl.load(in_ptr + (cy0 + y - P) * W + (cx0 + x),     mask=mask_C, other=0.0)
    d = tl.load(in_ptr + (dy0 + y - P) * W + (dx0 + x - P), mask=mask_D, other=0.0)

    # ============================================================
    # TODO(you): compute the mean over covered sub-blocks and store.
    #
    # What you have so far:
    #   - `a`, `b`, `c`, `d`: BLOCK-wide fp32 vectors, 0.0 where masked
    #   - `mask_A`, `mask_B`, `mask_C`, `mask_D`: BLOCK-wide bool vectors
    #   - `out_mask`: BLOCK-wide bool, True where output exists
    #   - `gx`, `gy`: BLOCK-wide int32, global output coords
    #
    # What you need to compute:
    #   1. `count` = number of sub-blocks covering each pixel (0..4)
    #      Hint: bool.to(tl.int32) gives 0/1, sum the four masks.
    #   2. `total` = a + b + c + d  (masked positions contributed 0)
    #   3. `mean`  = total / count, with 0 where count == 0
    #      Hint: tl.where(count > 0, total / count, 0.0)
    #   4. Store to `out_ptr + gy * W + gx`, cast to fp16 (tl.float16),
    #      masked by `out_mask`.
    #
    # Aim: 4-6 lines.
    # ============================================================
    count = tl.where(mask_A, 1.0, 0.0) + tl.where(mask_B, 1.0, 0.0) + tl.where(mask_C, 1.0, 0.0) + tl.where(mask_D, 1.0, 0.0)
    total = a + b + c + d
    mean = tl.where(count > 0, total / count, 0.0)
    tl.store(out_ptr + gy * W + gx, mean.to(tl.float16), mask=out_mask)


def fuse4_from_bigbuf_triton(
    in_big: torch.Tensor,
    out_big: torch.Tensor,
    offsets_cpu_i32: torch.Tensor,
    S: int,
    O: int,
    block_size: int = 1024,
) -> torch.Tensor:
    """Drop-in Triton replacement for the CUDA `fuse4_from_bigbuf`.

    Args:
        in_big:        [H, W] fp16 or fp32 contiguous CUDA tensor.
        out_big:       [H, W] fp16 contiguous CUDA tensor (written in-place).
        offsets_cpu_i32: [5, 2] int32 CPU tensor — rows are
                       (A_x, A_y), (B_x, B_y), (C_x, C_y), (D_x, D_y), (O_x, O_y).
        S:             sub-block side length.
        O:             overlap (sub-blocks overlap by O pixels).
        block_size:    Triton 1D block (number of output pixels per program).

    Returns:
        out_big (same object, modified in-place).
    """
    assert in_big.is_cuda and out_big.is_cuda
    assert in_big.dim() == 2 and out_big.dim() == 2
    assert in_big.is_contiguous() and out_big.is_contiguous()
    assert offsets_cpu_i32.dim() == 2 and offsets_cpu_i32.shape == (5, 2)
    assert offsets_cpu_i32.dtype == torch.int32 and not offsets_cpu_i32.is_cuda
    assert out_big.dtype == torch.float16, "out_big must be fp16 (matches CUDA)"

    H, W = in_big.shape
    assert out_big.shape == in_big.shape

    ax0, ay0 = int(offsets_cpu_i32[0, 0]), int(offsets_cpu_i32[0, 1])
    bx0, by0 = int(offsets_cpu_i32[1, 0]), int(offsets_cpu_i32[1, 1])
    cx0, cy0 = int(offsets_cpu_i32[2, 0]), int(offsets_cpu_i32[2, 1])
    dx0, dy0 = int(offsets_cpu_i32[3, 0]), int(offsets_cpu_i32[3, 1])
    ox0, oy0 = int(offsets_cpu_i32[4, 0]), int(offsets_cpu_i32[4, 1])

    P = S - O
    S_OUT = S + P
    n_out = S_OUT * S_OUT
    grid = (triton.cdiv(n_out, block_size),)

    _fuse4_from_bigbuf_kernel[grid](
        in_big, out_big,
        H, W,
        ax0, ay0, bx0, by0, cx0, cy0, dx0, dy0,
        ox0, oy0,
        S, P, S_OUT,
        BLOCK=block_size,
    )
    return out_big


# =============================================================================
# split4_to_bigbuf
# =============================================================================
# Geometry (matches split4_bigbuf_ext.py):
#   - Input `in_big` is the *parent* fused field (one S_out × S_out block
#     starting at (ix0, iy0)).
#   - Output `out_big` is the *child* buffer; we write 4 sub-blocks into it at
#     (ax0, ay0), (bx0, by0), (cx0, cy0), (dx0, dy0).
#   - For each output-local (x, y) in [0, S) × [0, S):
#       child A[x, y] = parent[ix0 + x,     iy0 + y]
#       child B[x, y] = parent[ix0 + P + x, iy0 + y]
#       child C[x, y] = parent[ix0 + x,     iy0 + P + y]
#       child D[x, y] = parent[ix0 + P + x, iy0 + P + y]
#   - Each thread writes 4 output pixels (one per sub-block).
# =============================================================================

@triton.jit
def _split4_to_bigbuf_kernel(
    in_ptr,            # parent bigbuf
    out_ptr,           # child bigbuf
    H, W,
    ix0, iy0,          # parent top-left
    ax0, ay0, bx0, by0, cx0, cy0, dx0, dy0,
    S,
    P,
    S_OUT,
    BLOCK: tl.constexpr,
):
    """Split one parent block into 4 child sub-blocks (simple copy, no blend)."""
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)            # 1D index into S*S

    # Child-local 2D coordinates
    x = idx % S
    y = idx // S
    in_child = idx < S * S

    # Parent read coordinates (one per sub-block)
    pa_x = ix0 + x;     pa_y = iy0 + y
    pb_x = ix0 + P + x; pb_y = iy0 + y
    pc_x = ix0 + x;     pc_y = iy0 + P + y
    pd_x = ix0 + P + x; pd_y = iy0 + P + y

    # Parent in-bounds masks
    pa_mask = in_child & (pa_x < W) & (pa_y < H)
    pb_mask = in_child & (pb_x < W) & (pb_y < H)
    pc_mask = in_child & (pc_x < W) & (pc_y < H)
    pd_mask = in_child & (pd_x < W) & (pd_y < H)

    # Load the 4 parent values that will be scattered
    va = tl.load(in_ptr + pa_y * W + pa_x, mask=pa_mask, other=0.0)
    vb = tl.load(in_ptr + pb_y * W + pb_x, mask=pb_mask, other=0.0)
    vc = tl.load(in_ptr + pc_y * W + pc_x, mask=pc_mask, other=0.0)
    vd = tl.load(in_ptr + pd_y * W + pd_x, mask=pd_mask, other=0.0)

    # ============================================================
    # TODO(you): scatter the 4 parent values into the child bigbuf.
    #
    # What you have so far:
    #   - `va`, `vb`, `vc`, `vd`: BLOCK-wide fp32 vectors
    #   - Sub-block output top-lefts: ax0/ay0, bx0/by0, cx0/cy0, dx0/dy0
    #   - `x`, `y`: BLOCK-wide int32, child-local coords
    #   - `in_child`: BLOCK-wide bool, True where (x, y) is in [0, S) × [0, S)
    #   - `H`, `W`: bigbuf shape
    #
    # For each sub-block (A, B, C, D):
    #   1. Compute global output address: (ay0 + y) * W + (ax0 + x) for A, etc.
    #   2. Compute output mask: in_child & (ox < W) & (oy < H)
    #   3. tl.store(out_ptr + addr, value.to(tl.float16), mask=mask)
    #
    # The CUDA kernel writes whatever dtype `out` was declared as (fp16 or fp32).
    # For drop-in parity with the most common CUDA path, cast to tl.float16.
    #
    # Aim: 4 stores × 3 lines each = ~12 lines. Repetitive but mechanical.
    # ============================================================
    oa_x = ax0 + x; oa_y = ay0 + y
    tl.store(out_ptr + oa_y * W + oa_x, va.to(tl.float16), mask=in_child & (oa_x < W) & (oa_y < H))
    ob_x = bx0 + x; ob_y = by0 + y
    tl.store(out_ptr + ob_y * W + ob_x, vb.to(tl.float16), mask=in_child & (ob_x < W) & (ob_y < H))
    oc_x = cx0 + x; oc_y = cy0 + y
    tl.store(out_ptr + oc_y * W + oc_x, vc.to(tl.float16), mask=in_child & (oc_x < W) & (oc_y < H))
    od_x = dx0 + x; od_y = dy0 + y
    tl.store(out_ptr + od_y * W + od_x, vd.to(tl.float16), mask=in_child & (od_x < W) & (od_y < H))

def split4_to_bigbuf_triton(
    in_big: torch.Tensor,
    out_big: torch.Tensor,
    offsets_cpu_i32: torch.Tensor,
    S: int,
    O: int,
    block_size: int = 1024,
) -> torch.Tensor:
    """Drop-in Triton replacement for the CUDA `split4_to_bigbuf`."""
    assert in_big.is_cuda and out_big.is_cuda
    assert in_big.dim() == 2 and out_big.dim() == 2
    assert in_big.is_contiguous() and out_big.is_contiguous()
    assert offsets_cpu_i32.dim() == 2 and offsets_cpu_i32.shape == (5, 2)
    assert offsets_cpu_i32.dtype == torch.int32 and not offsets_cpu_i32.is_cuda

    H, W = in_big.shape
    assert out_big.shape == in_big.shape

    ax0, ay0 = int(offsets_cpu_i32[0, 0]), int(offsets_cpu_i32[0, 1])
    bx0, by0 = int(offsets_cpu_i32[1, 0]), int(offsets_cpu_i32[1, 1])
    cx0, cy0 = int(offsets_cpu_i32[2, 0]), int(offsets_cpu_i32[2, 1])
    dx0, dy0 = int(offsets_cpu_i32[3, 0]), int(offsets_cpu_i32[3, 1])
    ix0, iy0 = int(offsets_cpu_i32[4, 0]), int(offsets_cpu_i32[4, 1])

    P = S - O
    S_OUT = S + P
    grid = (triton.cdiv(S * S, block_size),)

    _split4_to_bigbuf_kernel[grid](
        in_big, out_big,
        H, W,
        ix0, iy0,
        ax0, ay0, bx0, by0, cx0, cy0, dx0, dy0,
        S, P, S_OUT,
        BLOCK=block_size,
    )
    return out_big


# =============================================================================
# Numerical equivalence vs CUDA reference
# =============================================================================

@torch.no_grad()
def verify_fuse4_vs_cuda(
    H: int = 4096,
    W: int = 4096,
    S: int = 1024,
    O: int = 64,
    seed: int = 0,
    atol: float = 1e-3,    # fp16 tolerance
) -> dict:
    """Compare Triton fuse4 output to CUDA reference."""
    from fuilt.fusion_split.fuse4_bigbuf_ext import fuse4_from_bigbuf as fuse4_cuda

    torch.manual_seed(seed)
    P = S - O
    S_OUT = S + P

    # Construct a valid scenario: A, B, C, D sub-blocks + output region, all in-bounds.
    # Layout (top-left corners):
    #   A at (0, 0)
    #   B at (S, 0)
    #   C at (0, S)
    #   D at (S, S)
    #   O at (2*S, 2*S)  — output region top-left
    ax0, ay0 = 0, 0
    bx0, by0 = S, 0
    cx0, cy0 = 0, S
    dx0, dy0 = S, S
    ox0, oy0 = 2 * S, 2 * S

    # Make sure (ox0 + S_OUT, oy0 + S_OUT) fits in H, W
    assert ox0 + S_OUT <= W and oy0 + S_OUT <= H, f"bigbuf too small: need >= {ox0 + S_OUT}"

    in_big = torch.randn(H, W, device="cuda", dtype=torch.float32)
    offsets = torch.tensor(
        [[ax0, ay0], [bx0, by0], [cx0, cy0], [dx0, dy0], [ox0, oy0]],
        dtype=torch.int32,
    )

    out_cuda = torch.zeros_like(in_big, dtype=torch.float16)
    out_triton = torch.zeros_like(in_big, dtype=torch.float16)

    fuse4_cuda(in_big, out_cuda, offsets, S, O)
    fuse4_from_bigbuf_triton(in_big, out_triton, offsets, S, O)

    # Compare only the written S_OUT × S_OUT region
    region_cuda = out_cuda[oy0:oy0 + S_OUT, ox0:ox0 + S_OUT]
    region_triton = out_triton[oy0:oy0 + S_OUT, ox0:ox0 + S_OUT]
    diff = (region_cuda.float() - region_triton.float()).abs()

    return {
        "kernel": "fuse4_from_bigbuf",
        "region": f"{S_OUT}x{S_OUT}",
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "pass": diff.max().item() < atol,
    }


@torch.no_grad()
def verify_split4_vs_cuda(
    H: int = 8192,
    W: int = 8192,
    S: int = 1024,
    O: int = 64,
    seed: int = 0,
    atol: float = 1e-3,
) -> dict:
    """Compare Triton split4 output to CUDA reference."""
    from fuilt.fusion_split.split4_bigbuf_ext import split4_to_bigbuf as split4_cuda

    torch.manual_seed(seed)
    P = S - O
    S_OUT = S + P

    # Layout: parent at (0, 0). Children in a 2x2 grid offset to not overlap
    # with the parent.
    ix0, iy0 = 0, 0
    ax0, ay0 = 0,         S_OUT
    bx0, by0 = S,         S_OUT
    cx0, cy0 = 0,         S_OUT + S
    dx0, dy0 = S,         S_OUT + S

    assert dx0 + S <= W and dy0 + S <= H, "bigbuf too small"

    in_big = torch.randn(H, W, device="cuda", dtype=torch.float32)
    offsets = torch.tensor(
        [[ax0, ay0], [bx0, by0], [cx0, cy0], [dx0, dy0], [ix0, iy0]],
        dtype=torch.int32,
    )

    out_cuda = torch.zeros_like(in_big, dtype=torch.float16)
    out_triton = torch.zeros_like(in_big, dtype=torch.float16)

    split4_cuda(in_big, out_cuda, offsets, S, O)
    split4_to_bigbuf_triton(in_big, out_triton, offsets, S, O)

    diff = (out_cuda.float() - out_triton.float()).abs()
    return {
        "kernel": "split4_to_bigbuf",
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "pass": diff.max().item() < atol,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("bigbuf Triton port vs CUDA reference (numerical equivalence)")
    print("=" * 60)
    for verify in (verify_fuse4_vs_cuda, verify_split4_vs_cuda):
        try:
            result = verify()
            for k, v in result.items():
                print(f"  {k:<18}: {v}")
            print("-" * 60)
        except Exception as e:
            print(f"  {verify.__name__}: FAILED with exception: {e}")
            print("-" * 60)
