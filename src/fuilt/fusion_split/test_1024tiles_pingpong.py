import time
import torch

from fuse4_bigbuf_ext import fuse4_from_bigbuf
from split4_bigbuf_ext import split4_to_bigbuf


def derive_levels(S0: int, levels: int, ratio: float = 1 / 16):
    """
    返回每层的 dict: {S, O, P, S_next}
    Level l: 输入子块边长 S
    父块边长 S_next = S + P (P=S-O)
    """
    out = []
    S = S0
    for _ in range(levels):
        O = int(round(S * ratio))
        O = max(0, min(O, S - 1))
        P = S - O
        S_next = S + P
        out.append({"S": S, "O": O, "P": P, "S_next": S_next})
        S = S_next
    # 最后一层（顶层块）只需要 S
    out.append({"S": S, "O": int(round(S * ratio)), "P": S - int(round(S * ratio)), "S_next": None})
    return out


def canvas_size_for_level0(grid0: int, S0: int, P0: int) -> int:
    return S0 + (grid0 - 1) * P0


def get_offsets_tensor(ax0, ay0, bx0, by0, cx0, cy0, dx0, dy0, ox0, oy0):
    return torch.tensor(
        [[ax0, ay0], [bx0, by0], [cx0, cy0], [dx0, dy0], [ox0, oy0]],
        dtype=torch.int32,
        device="cpu",
    )


def precompute_offsets_up(levels, base_grid: int):
    """
    为 Up-pass 预生成 offsets：
    level l 有 grid = base_grid / 2^(l+1) 个父块（例如 32->16->8->4->2->1）
    每个父块对应一组 offsets（A,B,C,D, OUT）
    """
    all_levels = []
    grid_in = base_grid
    for l in range(len(levels) - 1):  # fuse 到顶层，所以到倒数第二个index
        P_in = levels[l]["P"]
        # 父块在下一层 canvas 上的步长：P_out = S_out - O_out
        S_out = levels[l]["S_next"]
        O_out = levels[l + 1]["O"]
        P_out = S_out - O_out

        grid_out = grid_in // 2  # 父块网格
        offsets_level = []
        for r in range(grid_out):
            for c in range(grid_out):
                ax0, ay0 = (2 * c) * P_in, (2 * r) * P_in
                bx0, by0 = (2 * c + 1) * P_in, (2 * r) * P_in
                cx0, cy0 = (2 * c) * P_in, (2 * r + 1) * P_in
                dx0, dy0 = (2 * c + 1) * P_in, (2 * r + 1) * P_in
                ox0, oy0 = c * P_out, r * P_out
                offsets_level.append(get_offsets_tensor(ax0, ay0, bx0, by0, cx0, cy0, dx0, dy0, ox0, oy0))
        all_levels.append({"grid_out": grid_out, "offsets": offsets_level})
        grid_in = grid_out
    return all_levels


def precompute_offsets_down(levels, base_grid: int):
    """
    Down-pass：每层 split 从父块（位于 level l+1 canvas）切回 level l 的四个子块位置。
    offsets 仍用 [A,B,C,D, I(parent)] 的布局，但我们复用同一个 offsets tensor：
      - A/B/C/D 是子块写入位置（level l canvas）
      - I 是父块读取位置（level l+1 canvas）
    这里我们仍然用 get_offsets_tensor 生成 [A,B,C,D,O]，然后在 split 内当成 [A,B,C,D,I]
    """
    # Up-pass 有 L=0..4 五次 fuse；Down-pass L=4..0 五次 split
    all_levels = []
    # 从顶向下，父块网格依次：1->2->4->8->16
    grid_parent = 1
    for l in range(len(levels) - 2, -1, -1):  # l = 4..0
        # 当前 l 的子块步长：
        P_child = levels[l]["P"]

        # 父块在 level l+1 canvas 的步长：
        S_parent = levels[l]["S_next"]
        O_parent = levels[l + 1]["O"]
        P_parent = S_parent - O_parent

        grid_child = grid_parent * 2
        offsets_level = []
        for r in range(grid_parent):
            for c in range(grid_parent):
                # 子块写入位置（A/B/C/D）
                ax0, ay0 = (2 * c) * P_child, (2 * r) * P_child
                bx0, by0 = (2 * c + 1) * P_child, (2 * r) * P_child
                cx0, cy0 = (2 * c) * P_child, (2 * r + 1) * P_child
                dx0, dy0 = (2 * c + 1) * P_child, (2 * r + 1) * P_child

                # 父块读取位置（I）
                ix0, iy0 = c * P_parent, r * P_parent

                offsets_level.append(get_offsets_tensor(ax0, ay0, bx0, by0, cx0, cy0, dx0, dy0, ix0, iy0))
        all_levels.append({"grid_parent": grid_parent, "offsets": offsets_level})
        grid_parent = grid_child
    return all_levels


@torch.no_grad()
def write_level0_tiles(buf: torch.Tensor, base_grid: int, S0: int, P0: int):
    # 给每个 tile 写入一个可辨识的常数（编码 r/c），便于 debug
    for r in range(base_grid):
        for c in range(base_grid):
            y0 = r * P0
            x0 = c * P0
            val = float(r * base_grid + c) / (base_grid * base_grid)  # 0~1
            buf[y0 : y0 + S0, x0 : x0 + S0].fill_(val)


def main():
    assert torch.cuda.is_available(), "需要CUDA"

    S0 = 1024
    ratio = 1 / 16
    levels_fuse = 5          # fuse 5 次：32x32 -> 1
    base_grid = 32           # 1024 tiles = 32x32

    levels = derive_levels(S0, levels_fuse, ratio=ratio)
    P0 = levels[0]["P"]

    canvas0 = canvas_size_for_level0(base_grid, S0, P0)
    S_top = levels[-1]["S"]

    MAX_SIZE = int(max(canvas0, S_top))
    # 建议对齐到 256，避免奇怪的对齐/越界风险（不是必须）
    MAX_SIZE = ((MAX_SIZE + 255) // 256) * 256

    print(f"[init] base_grid={base_grid} canvas0={canvas0} S_top={S_top} => MAX_SIZE={MAX_SIZE}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    buf_A = torch.zeros((MAX_SIZE, MAX_SIZE), dtype=torch.float16, device="cuda").contiguous()
    buf_B = torch.zeros((MAX_SIZE, MAX_SIZE), dtype=torch.float16, device="cuda").contiguous()

    print("[write] writing 32x32 tiles to level0 canvas...")
    write_level0_tiles(buf_A, base_grid, S0, P0)

    print("[prep] precompute offsets...")
    up = precompute_offsets_up(levels, base_grid)
    down = precompute_offsets_down(levels, base_grid)

    torch.cuda.synchronize()
    t0 = time.time()

    print("[up-pass] ...")
    for l in range(levels_fuse):
        S_in = levels[l]["S"]
        O_in = levels[l]["O"]

        src = buf_A if (l % 2 == 0) else buf_B
        dst = buf_B if (l % 2 == 0) else buf_A

        # debug阶段建议清零 dst，避免步长/offset出错时被残留污染
        dst.zero_()

        offsets_list = up[l]["offsets"]
        for offsets in offsets_list:
            fuse4_from_bigbuf(src, dst, offsets, S=S_in, O=O_in)

    print("[down-pass] ...")
    for i, l in enumerate(range(levels_fuse - 1, -1, -1)):  # l=4..0
        # split 使用的 S/O 应该是“子块尺寸”(level l 的 S/O)
        S_child = levels[l]["S"]
        O_child = levels[l]["O"]

        # 上行结束时：
        # l=0 写B, l=1 写A, l=2 写B, l=3 写A, l=4 写B (因为 l=4 是偶数)
        # 所以顶层在 buf_B。down-pass 第一步从 buf_B 读。
        src = buf_B if (l % 2 == 0) else buf_A
        dst = buf_A if (l % 2 == 0) else buf_B

        dst.zero_()
        offsets_list = down[i]["offsets"]
        for offsets in offsets_list:
            split4_to_bigbuf(src, dst, offsets, S=S_child, O=O_child)

    torch.cuda.synchronize()
    t1 = time.time()

    elapsed_ms = (t1 - t0) * 1000
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    print("\n" + "=" * 60)
    print("1024 tiles (32x32) 全局画布 fuse(5) + split(5) 结果")
    print("=" * 60)
    print(f"MAX_SIZE={MAX_SIZE}, peak_mem={peak_mem_mb:.2f} MB, time={elapsed_ms:.2f} ms")
    print("=" * 60)

    # 简单 spot-check：检查左上 tile 的左上角值是否仍接近其编码（不一定完全相等，因为经过多层一致化）
    # 这里只做“程序能跑通”的 sanity check
    v00 = float(buf_A[0, 0].item())
    print(f"[sanity] buf_A[0,0]={v00:.6f}")


if __name__ == "__main__":
    main()