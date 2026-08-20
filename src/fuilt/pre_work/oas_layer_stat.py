from pathlib import Path

# KLayout Python API 兼容导入
try:
    import klayout.db as kdb
except Exception:
    import pya
    kdb = pya.db

def stat_oas_layers(oas_path):
    layout = kdb.Layout()
    layout.read(str(oas_path))
    top = layout.top_cell()
    if top is None:
        print("No top cell found!")
        return

    print(f"OAS file: {oas_path}")
    print(f"Top cell: {top.name}")
    print("Layer/Datatype  |  Shape count")
    print("-" * 32)
    for li in layout.layer_indexes():
        layer_info = layout.get_info(li)
        shapes = top.shapes(li)
        count = shapes.size()
        if count > 0:
            print(f"Layer {layer_info.layer}/{layer_info.datatype} : {count}")

if __name__ == "__main__":
    # 改成你的 OAS 路径
    oas_path = "/data/lyj/FuILT/FullChip/1_test_deliver_clip_1x.oas"
    stat_oas_layers(oas_path)