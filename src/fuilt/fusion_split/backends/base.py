"""Abstract base class for v8 ablation backends.

The contract is a single-layer fuse4 + split4 cycle:

  fuse4:  4 sub-blocks (A, B, C, D) scattered in `in_big` → averaged into
          the S_out × S_out region starting at (ox0, oy0) of `out_big`.

  split4: the S_out × S_out parent block at (ix0, iy0) of `in_big` →
          4 child sub-blocks scattered into `out_big` at A/B/C/D positions.

All backends must produce bit-equivalent output (modulo fp16 cast tolerance)
so that ablation measures implementation cost, not algorithmic differences.
"""

from abc import ABC, abstractmethod

import torch


class BigbufBackend(ABC):
    """Single-layer fuse4 + split4 backend interface."""

    name: str = "base"

    @abstractmethod
    def fuse4(
        self,
        in_big: torch.Tensor,
        out_big: torch.Tensor,
        offsets: torch.Tensor,
        S: int,
        O: int,
    ) -> torch.Tensor:
        """Fuse 4 sub-blocks (A, B, C, D) from `in_big` into `out_big`.

        Args:
            in_big:  [H, W] fp16 or fp32 contiguous CUDA tensor.
            out_big: [H, W] fp16 contiguous CUDA tensor (written in-place).
            offsets: [5, 2] int32 CPU tensor — rows are
                     (A_x, A_y), (B_x, B_y), (C_x, C_y), (D_x, D_y), (O_x, O_y).
            S:       sub-block side length.
            O:       overlap (sub-blocks overlap by O pixels; P = S - O).

        Returns:
            out_big (same object, modified in-place).
        """
        ...

    @abstractmethod
    def split4(
        self,
        in_big: torch.Tensor,
        out_big: torch.Tensor,
        offsets: torch.Tensor,
        S: int,
        O: int,
    ) -> torch.Tensor:
        """Split parent block from `in_big` into 4 child sub-blocks in `out_big`.

        Args:
            in_big:  [H, W] fp16 or fp32 contiguous CUDA tensor (parent source).
            out_big: [H, W] fp16 contiguous CUDA tensor (children destination).
            offsets: [5, 2] int32 CPU tensor — rows are
                     (A_x, A_y), (B_x, B_y), (C_x, C_y), (D_x, D_y), (I_x, I_y).
            S, O:    as above.

        Returns:
            out_big (same object, modified in-place).
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
