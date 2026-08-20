import torch

from fuse4_bigbuf_ext import fuse4_from_bigbuf
from split4_bigbuf_ext import split4_to_bigbuf


def main():
    assert torch.cuda.is_available(), "需要CUDA"
    dev = torch.device("cuda:0")

    # big buffer：大一点，便于放下输入四块和输出父块（1984x1984）
    H = W = 4096
    in_big = torch.zeros((H, W), device=dev, dtype=torch.float16).contiguous()
    fused_big = torch.zeros((H, W), device=dev, dtype=torch.float16).contiguous()
    out_big = torch.zeros((H, W), device=dev, dtype=torch.float16).contiguous()

    # level0: S=1024, overlap=1/16 => O=64, stride P=960, S_out=1984
    S = 1024
    O = S // 16
    P = S - O
    S_out = S + P

    # 在 in_big 里放 4 个 tile（彼此不重叠放置），用常数便于验证
    ax0, ay0 = 0, 0
    bx0, by0 = 1100, 0
    cx0, cy0 = 0, 1100
    dx0, dy0 = 1100, 1100

    in_big[ay0:ay0 + S, ax0:ax0 + S] = 0.0
    in_big[by0:by0 + S, bx0:bx0 + S] = 1.0
    in_big[cy0:cy0 + S, cx0:cx0 + S] = 2.0
    in_big[dy0:dy0 + S, dx0:dx0 + S] = 3.0

    # 父块输出位置（写到 fused_big 的这里）
    ox0, oy0 = 2000, 2000

    fuse_offsets = torch.tensor(
        [
            [ax0, ay0],  # A
            [bx0, by0],  # B
            [cx0, cy0],  # C
            [dx0, dy0],  # D
            [ox0, oy0],  # OUT
        ],
        dtype=torch.int32,
        device="cpu",
    )

    # 1) 融合：in_big(A/B/C/D) -> fused_big(OUT)
    fuse4_from_bigbuf(in_big, fused_big, fuse_offsets, S=S, O=O)

    fused_view = fused_big[oy0:oy0 + S_out, ox0:ox0 + S_out].float()

    # 验证融合结果：左上=0；顶部重叠带=0.5；左侧重叠带=1.0；中心四重叠=1.5
    p00 = float(fused_view[0, 0].item())
    p_top = float(fused_view[0, P + 10].item())      # x in [P, S) => A+B
    p_left = float(fused_view[P + 10, 0].item())     # y in [P, S) => A+C
    p_center = float(fused_view[P + 10, P + 10].item())  # 四重叠区

    print("FUSE check:")
    print("  fused(0,0) =", p00, "expect 0.0")
    print("  fused(top overlap) =", p_top, "expect 0.5")
    print("  fused(left overlap) =", p_left, "expect 1.0")
    print("  fused(center) =", p_center, "expect 1.5")

    assert abs(p00 - 0.0) < 1e-2
    assert abs(p_top - 0.5) < 1e-2
    assert abs(p_left - 1.0) < 1e-2
    assert abs(p_center - 1.5) < 1e-2

    # 2) 分割：把 fused_big(父块)切回四个子块写入 out_big 的 A/B/C/D 位置
    split_offsets = torch.tensor(
        [
            [ax0, ay0],  # A dst
            [bx0, by0],  # B dst
            [cx0, cy0],  # C dst
            [dx0, dy0],  # D dst
            [ox0, oy0],  # parent src
        ],
        dtype=torch.int32,
        device="cpu",
    )

    split4_to_bigbuf(fused_big, out_big, split_offsets, S=S, O=O)

    # 验证 split 后的四块：它们应等于从父块取出的对应象限（包含重叠一致化后的值）
    A = out_big[ay0:ay0 + S, ax0:ax0 + S].float()
    B = out_big[by0:by0 + S, bx0:bx0 + S].float()
    C = out_big[cy0:cy0 + S, cx0:cx0 + S].float()
    D = out_big[dy0:dy0 + S, dx0:dx0 + S].float()

    # 取子块中的几个点验证（注意：这些点对应父块不同位置）
    # A 中靠近右下角会落在重叠区，因此不再是全0
    print("\nSPLIT check samples:")
    print("  A(0,0) =", float(A[0, 0].item()), "expect 0.0 (来自父块左上)")
    print("  A(0, P+10) =", float(A[0, P + 10].item()), "expect 0.5 (父块顶部重叠带)")
    print("  A(P+10, 0) =", float(A[P + 10, 0].item()), "expect 1.0 (父块左侧重叠带)")
    print("  A(P+10, P+10) =", float(A[P + 10, P + 10].item()), "expect 1.5 (父块中心四重叠)")

    assert abs(float(A[0, 0].item()) - 0.0) < 1e-2
    assert abs(float(A[0, P + 10].item()) - 0.5) < 1e-2
    assert abs(float(A[P + 10, 0].item()) - 1.0) < 1e-2
    assert abs(float(A[P + 10, P + 10].item()) - 1.5) < 1e-2

    # B 左上角对应父块 (0,P) 附近，也在顶部重叠带
    print("  B(0,0) =", float(B[0, 0].item()), "expect 0.5 (父块顶部重叠带)")
    assert abs(float(B[0, 0].item()) - 0.5) < 1e-2

    # D 左上角对应父块中心区域
    print("  D(0,0) =", float(D[0, 0].item()), "expect 1.5 (父块中心四重叠)")
    assert abs(float(D[0, 0].item()) - 1.5) < 1e-2

    print("\n✅ fuse -> split 测试通过")


if __name__ == "__main__":
    main()