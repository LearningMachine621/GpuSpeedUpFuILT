"""CUDA-backed fuse4/split4 backend (hand-written kernel baseline).

Wraps the existing `fuse4_bigbuf_ext.fuse4_from_bigbuf` and
`split4_bigbuf_ext.split4_to_bigbuf` to satisfy the BigbufBackend interface.

This is the "perf ceiling" reference in the v8 ablation: the hand-tuned CUDA
implementation that the Triton port is compared against.
"""

import torch

from .base import BigbufBackend
from ..fuse4_bigbuf_ext import fuse4_from_bigbuf as _fuse4_cuda
from ..split4_bigbuf_ext import split4_to_bigbuf as _split4_cuda


class CudaBigbufBackend(BigbufBackend):
    name = "cuda_bigbuf"

    def fuse4(self, in_big, out_big, offsets, S, O):
        # The CUDA kernel requires fp16 output. We assert this in the wrapper
        # to surface dtype mismatches early instead of letting the kernel
        # silently miscompile.
        assert out_big.dtype == torch.float16, (
            f"cuda_bigbuf requires out_big fp16, got {out_big.dtype}"
        )
        return _fuse4_cuda(in_big, out_big, offsets, S, O)

    def split4(self, in_big, out_big, offsets, S, O):
        # The CUDA kernel supports fp16 or fp32 output; we just pass through.
        return _split4_cuda(in_big, out_big, offsets, S, O)
