import os

# 项目根目录（向上四级）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# 设置路径（请根据实际数据路径修改）
image_dir = os.path.join(_PROJECT_ROOT, "dataset", "raw", "unet_segment", "images")
mask_dir = os.path.join(_PROJECT_ROOT, "dataset", "raw", "unet_segment", "json")

# 获取文件名（不带扩展名）
image_files = {os.path.splitext(f)[0] for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))}
mask_files = {os.path.splitext(f)[0] for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f))}

# 找出不对应的文件
images_without_masks = image_files - mask_files
masks_without_images = mask_files - image_files

print(f"Images总数: {len(image_files)}")
print(f"Masks总数: {len(mask_files)}")
print(f"\n有Image但无Mask的文件: {len(images_without_masks)}个")
print(f"有Mask但无Image的文件: {len(masks_without_images)}个")

# 删除不对应的Image文件
deleted_images = []
for filename in os.listdir(image_dir):
    name_without_ext = os.path.splitext(filename)[0]
    if name_without_ext in images_without_masks:
        filepath = os.path.join(image_dir, filename)
        os.remove(filepath)
        deleted_images.append(filename)

# 删除不对应的Mask文件  
deleted_masks = []
for filename in os.listdir(mask_dir):
    name_without_ext = os.path.splitext(filename)[0]
    if name_without_ext in masks_without_images:
        filepath = os.path.join(mask_dir, filename)
        os.remove(filepath)
        deleted_masks.append(filename)

print(f"\n已删除 {len(deleted_images)} 个不对应的Image文件:")
for f in deleted_images[:10]:  # 只显示前10个
    print(f"  - {f}")
if len(deleted_images) > 10:
    print(f"  ... 还有 {len(deleted_images)-10} 个")

print(f"\n已删除 {len(deleted_masks)} 个不对应的Mask文件:")
for f in deleted_masks[:10]:
    print(f"  - {f}")
if len(deleted_masks) > 10:
    print(f"  ... 还有 {len(deleted_masks)-10} 个")

# 最终统计
final_images = len([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))])
final_masks = len([f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f))])
print(f"\n✅ 处理完成！最终对齐数量: Images={final_images}, Masks={final_masks}")