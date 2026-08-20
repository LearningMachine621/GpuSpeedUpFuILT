"""PyTorch-native fuse4/split4 implementation.

Pure-PyTorch reference path using slice + add_/mul_/copy_. Acts as the
zero-kernel baseline in the v8 ablation: "is a hand-written kernel still
worth it once PyTorch in-place ops are competitive?"

Geometry matches fuse4_bigbuf_ext.py / split4_bigbuf_ext.py exactly so the
three backends can be diffed element-wise.
"""

import torch

from .base import BigbufBackend


class PytorchInplaceBackend(BigbufBackend):
    name = "pytorch_inplace"

    def fuse4(self, in_big, out_big, offsets, S, O):
        assert in_big.is_cuda and out_big.is_cuda
        assert in_big.dim() == 2 and out_big.dim() == 2
        assert offsets.dim() == 2 and offsets.shape == (5, 2)

        # fp32 working buffer (fuse must average — fp16 would lose precision)
        H, W = in_big.shape
        P = S - O
        S_out = S + P

        ax0, ay0 = int(offsets[0, 0]), int(offsets[0, 1])
        bx0, by0 = int(offsets[1, 0]), int(offsets[1, 1])
        cx0, cy0 = int(offsets[2, 0]), int(offsets[2, 1])
        dx0, dy0 = int(offsets[3, 0]), int(offsets[3, 1])
        ox0, oy0 = int(offsets[4, 0]), int(offsets[4, 1])

        # Accumulator + count, fp32 to match the CUDA kernel's intermediate math
        acc = torch.zeros((S_out, S_out), dtype=torch.float32, device=in_big.device)
        cnt = torch.zeros((S_out, S_out), dtype=torch.float32, device=in_big.device)

        # Quadrant coords within S_out × S_out output region (matches the
        # CUDA kernel's coverage map exactly):
        #   A: [0, S) × [0, S)
        #   B: [P, P+S) × [0, S)
        #   C: [0, S) × [P, P+S)
        #   D: [P, P+S) × [P, P+S)
        for (sx, sy), (x0, y0) in [
            ((0, 0), (ax0, ay0)),
            ((P, 0), (bx0, by0)),
            ((0, P), (cx0, cy0)),
            ((P, P), (dx0, dy0)),
        ]:
            acc[sy:sy + S, sx:sx + S].add_(in_big[y0:y0 + S, x0:x0 + S].float())
            cnt[sy:sy + S, sx:sx + S] += 1.0

        # Mean with divide-by-zero guard (matches tl.where(count > 0, ...))
        cnt[cnt == 0] = 1.0
        acc.div_(cnt)

        # Write to out_big (cast to fp16 to match CUDA kernel's hardcoded output)
        out_big[oy0:oy0 + S_out, ox0:ox0 + S_out] = acc.to(torch.float16)
        return out_big

    def split4(self, in_big, out_big, offsets, S, O):
        assert in_big.is_cuda and out_big.is_cuda
        assert in_big.dim() == 2 and out_big.dim() == 2
        assert offsets.dim() == 2 and offsets.shape == (5, 2)

        P = S - O

        ax0, ay0 = int(offsets[0, 0]), int(offsets[0, 1])
        bx0, by0 = int(offsets[1, 0]), int(offsets[1, 1])
        cx0, cy0 = int(offsets[2, 0]), int(offsets[2, 1])
        dx0, dy0 = int(offsets[3, 0]), int(offsets[3, 1])
        ix0, iy0 = int(offsets[4, 0]), int(offsets[4, 1])

        # For each child position (x, y) in [0, S) × [0, S):
        #   child A[x, y] = parent[ix0 + x,     iy0 + y]
        #   child B[x, y] = parent[ix0 + P + x, iy0 + y]
        #   child C[x, y] = parent[ix0 + x,     iy0 + P + y]
        #   child D[x, y] = parent[ix0 + P + x, iy0 + P + y]
        out_big[ay0:ay0 + S, ax0:ax0 + S] = in_big[iy0:iy0 + S,         ix0:ix0 + S].to(torch.float16)
        out_big[by0:by0 + S, bx0:bx0 + S] = in_big[iy0:iy0 + S,         ix0 + P:ix0 + P + S].to(torch.float16)
        out_big[cy0:cy0 + S, cx0:cx0 + S] = in_big[iy0 + P:iy0 + P + S, ix0:ix0 + S].to(torch.float16)
        out_big[dy0:dy0 + S, dx0:dx0 + S] = in_big[iy0 + P:iy0 + P + S, ix0 + P:ix0 + P + S].to(torch.float16)
        return out_big
