import os
try:
    import klayout.db as kdb
except Exception:
    import pya as kdb

# ─────────────────────────────────────────
# 参数（与 fuse4_bigbuf_ext 保持一致）
# S0=1024, O0=64, P0=960
# 4x4 grid，步长 P0，tile大小 S0
# ─────────────────────────────────────────
S0 = 1024   # tile大小
O0 = 64     # 重叠
P0 = S0 - O0  # 步长 = 960
GRID = 4    # 4x4


def split_patch2tiles(
    patch_oas: str,
    out_dir: str,
    layer: tuple = (23, 100),
    s: int = S0,
    overlap: int = O0,
    grid: int = GRID,
):
    """
    将一个 level2 patch（3844x3844）切成 grid x grid 个 tile（带重叠）
    tile大小 = s，重叠 = overlap，步长 = s - overlap
    """
    stride = s - overlap  # P0 = 960

    layout = kdb.Layout()
    layout.read(patch_oas)
    top_cell = layout.top_cell()
    bbox = top_cell.bbox()

    print(f"Patch bbox: left={bbox.left}, bottom={bbox.bottom}, right={bbox.right}, top={bbox.top}")
    print(f"Patch 实际大小: {bbox.width()} x {bbox.height()}")
    print(f"理论需要大小: {s + (grid-1)*stride} x {s + (grid-1)*stride}")

    # 强制从 bbox.left/bottom 出发，保证切出 grid x grid 个完整 tile
    x0_origin = bbox.left
    y0_origin = bbox.bottom

    layer_info = kdb.LayerInfo(layer[0], layer[1])
    li = layout.layer(layer_info)
    full_region = kdb.Region(top_cell.shapes(li))

    os.makedirs(out_dir, exist_ok=True)

    tile_count = 0
    for row in range(grid):
        for col in range(grid):
            tx0 = x0_origin + col * stride
            ty0 = y0_origin + row * stride
            tx1 = tx0 + s  # 不再裁剪，保证每个tile都是 s x s
            ty1 = ty0 + s

            tile_box = kdb.Box(tx0, ty0, tx1, ty1)

            clipped = full_region & kdb.Region(tile_box)

            sub_layout = kdb.Layout()
            sub_layout.dbu = layout.dbu
            sub_cell = sub_layout.create_cell("TOP")
            sub_li = sub_layout.insert_layer(layer_info)

            if not clipped.is_empty():
                clipped.move(-tx0, -ty0)
                clipped.insert_into(sub_layout, sub_cell.cell_index(), sub_li)

            out_name = os.path.join(
                out_dir,
                f"tile_r{row}_c{col}_s{s}_o{overlap}_orig{tx0}_{ty0}_{s}x{s}.oas"
            )
            sub_layout.write(out_name)
            tile_count += 1
            print(f"  [r{row},c{col}] orig=({tx0},{ty0}) size={s}x{s}"
                  f" {'有图形' if not clipped.is_empty() else '空'}")

    print(f"\n[done] 共导出 {tile_count} 个 tile → {out_dir}")


if __name__ == "__main__":
    split_patch2tiles(
        patch_oas="/data/lyj/FuILT/first_patch_level2/patch_level2_L23_100_3904x3904_orig_100000_100000.oas",
        out_dir="/data/lyj/FuILT/tiles_from_patch/",
        layer=(23, 100),
        s=S0,
        overlap=O0,
        grid=GRID,
    )