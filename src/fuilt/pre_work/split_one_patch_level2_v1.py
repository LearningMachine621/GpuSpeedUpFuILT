import os
import re

try:
    import klayout.db as kdb
except Exception:
    import pya as kdb


S0 = 1024
O0 = 64
GRID = 4
PATCH_SIZE = 3904

def _parse_patch_origin_from_name(patch_oas: str):
    name = os.path.basename(patch_oas)
    m = re.search(r"orig(-?\d+)_(-?\d+).oas$", name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))

def split_patch2tiles(
    patch_oas: str,
    out_dir: str,
    layer: tuple = (23, 100),
    patch_size: int = PATCH_SIZE,
    patch_origin_abs: tuple = None,
    s: int = S0,
    overlap: int = O0,
    grid: int = GRID,
    skip_empty: bool = False,
    ):



    if s <= 0:
        raise ValueError("s 必须 > 0")
    if overlap < 0:
        raise ValueError("overlap 必须 >= 0")
    if grid <= 0:
        raise ValueError("grid 必须 > 0")
    if patch_size <= 0:
        raise ValueError("patch_size 必须 > 0")

    stride = s - overlap
    if stride <= 0:
        raise ValueError(f"无效参数: stride={stride}, 需满足 s > overlap")

    need_w = s + (grid - 1) * stride
    need_h = s + (grid - 1) * stride
    if need_w > patch_size or need_h > patch_size:
        raise ValueError(
            f"grid/s/overlap 超出 patch 尺寸: 需要 {need_w}x{need_h}, patch_size={patch_size}"
        )

    # 确定 patch 全局原点
    if patch_origin_abs is None:
        parsed = _parse_patch_origin_from_name(patch_oas)
        if parsed is None:
            raise ValueError(
                "无法从 patch 文件名解析全局原点。请传入 patch_origin_abs=(abs_x, abs_y)，"
                "或使用带 orig_x_y 的文件名。"
            )
        patch_abs_x, patch_abs_y = parsed
    else:
        patch_abs_x, patch_abs_y = int(patch_origin_abs[0]), int(patch_origin_abs[1])

    layout = kdb.Layout()
    layout.read(patch_oas)
    top_cell = layout.top_cell()
    bbox = top_cell.bbox()

    layer_info = kdb.LayerInfo(layer[0], layer[1])
    li = layout.layer(layer_info)
    full_region = kdb.Region(top_cell.shapes(li))

    # 强制以 patch 画布范围为准，而不是图形 bbox
    canvas_box = kdb.Box(0, 0, patch_size, patch_size)
    full_region = full_region & kdb.Region(canvas_box)

    os.makedirs(out_dir, exist_ok=True)

    print("Patch 坐标系信息:")
    print(f"  patch 文件: {patch_oas}")
    print(f"  图形 bbox : left={bbox.left}, bottom={bbox.bottom}, right={bbox.right}, top={bbox.top}")
    print(f"  patch 画布: (0,0) -> ({patch_size},{patch_size})")
    print(f"  patch 全局原点: ({patch_abs_x},{patch_abs_y})")
    print(f"  tile 参数: s={s}, overlap={overlap}, stride={stride}, grid={grid}x{grid}")
    print(f"  覆盖尺寸: {need_w}x{need_h}")

    tile_count = 0
    non_empty_count = 0

    for row in range(grid):
        for col in range(grid):
            tx0 = col * stride
            ty0 = row * stride
            tx1 = tx0 + s
            ty1 = ty0 + s

            # 二次保护，确保 tile 完全在 patch 内
            if tx1 > patch_size or ty1 > patch_size:
                print(f"  [r{row},c{col}] 超出 patch 边界，跳过")
                continue

            tile_box = kdb.Box(tx0, ty0, tx1, ty1)
            clipped = full_region & kdb.Region(tile_box)
            is_empty = clipped.is_empty()

            if skip_empty and is_empty:
                print(f"  [r{row},c{col}] rel=({tx0},{ty0}) abs=({patch_abs_x + tx0},{patch_abs_y + ty0}) 空，跳过")
                continue

            sub_layout = kdb.Layout()
            sub_layout.dbu = layout.dbu
            sub_cell = sub_layout.create_cell("TOP")
            sub_li = sub_layout.insert_layer(layer_info)

            if not is_empty:
                clipped.move(-tx0, -ty0)
                clipped.insert_into(sub_layout, sub_cell.cell_index(), sub_li)
                non_empty_count += 1

            gx0 = patch_abs_x + tx0
            gy0 = patch_abs_y + ty0

            out_name = os.path.join(
                out_dir,
                f"tile_r{row}_c{col}_s{s}_o{overlap}_rel{tx0}_{ty0}_abs{gx0}_{gy0}_{s}x{s}.oas"
            )
            sub_layout.write(out_name)
            tile_count += 1

            print(
                f"  [r{row},c{col}] rel=({tx0},{ty0}) abs=({gx0},{gy0}) "
                f"size={s}x{s} {'有图形' if not is_empty else '空'}"
            )

    print(f"\n[done] 导出 tile 数: {tile_count}, 非空 tile 数: {non_empty_count}, 输出目录: {out_dir}")
if __name__ == "__main__":
    split_patch2tiles(
        patch_oas="/data/lyj/FuILT/first_patch_level2/patch_level2_L23_100_3904x3904_orig_100000_100000.oas",
        out_dir="/data/lyj/FuILT/tiles_from_patch/",
        layer=(23, 100),
        patch_size=3904,
        patch_origin_abs=(100000,100000), # None 表示从文件名 orig_x_y 自动解析
        s=1024,
        overlap=64,
        grid=4,
        skip_empty=False,
        )