import os
import json

# 项目根目录（向上四级）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# 设置路径（请根据实际数据路径修改）
seg_img_dir = os.path.join(_PROJECT_ROOT, "dataset", "raw", "unet_segment", "images")
seg_json_dir = os.path.join(_PROJECT_ROOT, "dataset", "raw", "unet_segment", "json")

# 统计被删除的文件数量
deleted_count = 0

print("🔍 开始检查并清理空标注文件...")

# 遍历 json 文件夹中的所有文件
for json_name in os.listdir(seg_json_dir):
    if not json_name.endswith('.json'):
        continue

    json_path = os.path.join(seg_json_dir, json_name)

    # 读取 JSON 文件内容
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 无法读取 JSON 文件 {json_name}: {e}")
        continue

    # 检查 'shapes' 是否为空列表
    if 'shapes' in data and len(data['shapes']) == 0:
        # 获取对应的图片文件名
        img_name = os.path.splitext(json_name)[0] + ".png" # 假设图片是 png，如果是 jpg 需要修改
        img_path = os.path.join(seg_img_dir, img_name)

        # 1. 删除空的 JSON 文件
        os.remove(json_path)
        print(f"🗑️ 已删除空 JSON: {json_name}")

        # 2. 尝试删除对应的图片文件
        if os.path.exists(img_path):
            os.remove(img_path)
            print(f"🗑️ 已删除对应图片: {img_name}")
        else:
            # 如果图片扩展名不是 png，或者图片本来就不存在
            print(f"⚠️ 找不到对应的图片文件或图片已被删除: {img_name}")

        deleted_count += 1

print("-" * 40)
print(f"🎉 清理完成！共删除了 {deleted_count} 组没有标注多边形的数据。")