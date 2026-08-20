import os

try:
    import klayout.db as kdb
except Exception:
    import pya as kdb

def split_one_patch_level2(
    input_oas: str,
    output_folder: str,
    patch_size: int = 3844,    # level2 大小
    layer: tuple = (23, 100),  # 指定导出的层 (layer, datatype)
):
    layout = kdb.Layout()
    layout.read(input_oas)
    top_cell = layout.top_cell()
    bbox = top_cell.bbox()

    print(f"设计边界框: left={bbox.left}, bottom={bbox.bottom}, right={bbox.right}, top={bbox.top}")
    print(f"设计尺寸: {bbox.width()} x {bbox.height()}")
    print(f"patch_size: {patch_size}, 使用层: {layer}")

    # 取第一个 patch（从设计原点开始）
    x0 = bbox.left
    y0 = bbox.bottom
    actual_width  = min(x0 + patch_size, bbox.right)  - x0
    actual_height = min(y0 + patch_size, bbox.top)    - y0

    if actual_width <= 0 or actual_height <= 0:
        print("该区域超出设计边界，无有效内容！")
        return

    tile_box = kdb.Box(x0, y0, x0 + actual_width, y0 + actual_height)
    print(f"tile_box: {tile_box}")

    # 只取指定 layer 的 Region
    layer_info = kdb.LayerInfo(layer[0], layer[1])
    li = layout.layer(layer_info)
    region = kdb.Region(top_cell.shapes(li))

    # 裁剪
    clipped = region & kdb.Region(tile_box)
    if clipped.is_empty():
        print(f"Layer {layer} 在该区域没有任何图形，跳过！")
        return

    # 创建子 layout，只写入指定层
    sub_layout = kdb.Layout()
    sub_layout.dbu = layout.dbu
    sub_cell = sub_layout.create_cell("TOP")
    sub_li = sub_layout.insert_layer(layer_info)

    # 移动到相对坐标（原点为 patch 左下角）
    clipped.move(-x0, -y0)
    clipped.insert_into(sub_layout, sub_cell.cell_index(), sub_li)

    os.makedirs(output_folder, exist_ok=True)
    out_name = os.path.join(
        output_folder,
        f"patch_level2_L{layer[0]}_{layer[1]}_{actual_width}x{actual_height}_orig_{x0}_{y0}.oas"
    )
    sub_layout.write(out_name)
    print(f"[done] 已保存: {out_name}")


if __name__ == "__main__":
    split_one_patch_level2(
        input_oas="/data/lyj/FuILT/FullChip/1_test_deliver_clip_1x.oas",
        output_folder="/data/lyj/FuILT/first_patch_level2/",
        patch_size=3904,
        layer=(23, 100),
    )