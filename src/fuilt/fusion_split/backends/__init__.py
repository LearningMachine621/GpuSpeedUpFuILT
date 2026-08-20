"""v8 ablation backends for fuse4/split4 single-layer operation.

Three interchangeable implementations of the same operation:
- pytorch_inplace: pure PyTorch slice + add_/mul_/copy_
- cuda_bigbuf:     hand-written CUDA via torch.utils.cpp_extension
- triton_bigbuf:   Triton port (triton_bigbuf_ext)

All three implement the same `BigbufBackend` interface.
"""

from .base import BigbufBackend
from .pytorch_inplace import PytorchInplaceBackend
from .cuda_bigbuf import CudaBigbufBackend
from .triton_bigbuf import TritonBigbufBackend


def get_backend(name: str) -> BigbufBackend:
    """Factory: return the backend instance by name.

    Valid names: pytorch_inplace | cuda_bigbuf | triton_bigbuf
    """
    mapping = {
        "pytorch_inplace": PytorchInplaceBackend,
        "cuda_bigbuf":     CudaBigbufBackend,
        "triton_bigbuf":   TritonBigbufBackend,
    }
    if name not in mapping:
        raise ValueError(f"unknown backend: {name!r}. valid: {list(mapping)}")
    return mapping[name]()


__all__ = ["BigbufBackend", "get_backend"]
