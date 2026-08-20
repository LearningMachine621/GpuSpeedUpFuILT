"""Triton-backed fuse4/split4 backend.

Wraps the Triton port in `triton_bigbuf_ext` to satisfy the BigbufBackend
interface. This is the recommended production path post-migration: same
numerical output as cuda_bigbuf, but in pure Python + Triton.
"""

import torch

from .base import BigbufBackend
from ..triton_bigbuf_ext import (
    fuse4_from_bigbuf_triton as _fuse4_triton,
    split4_to_bigbuf_triton as _split4_triton,
)


class TritonBigbufBackend(BigbufBackend):
    name = "triton_bigbuf"

    def fuse4(self, in_big, out_big, offsets, S, O):
        # The Triton kernel matches the CUDA kernel's fp16 output contract.
        assert out_big.dtype == torch.float16, (
            f"triton_bigbuf requires out_big fp16, got {out_big.dtype}"
        )
        return _fuse4_triton(in_big, out_big, offsets, S, O)

    def split4(self, in_big, out_big, offsets, S, O):
        return _split4_triton(in_big, out_big, offsets, S, O)
