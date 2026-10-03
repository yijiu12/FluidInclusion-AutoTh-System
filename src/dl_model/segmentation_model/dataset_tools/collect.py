import os
import cv2

# 项目根目录（向上四级）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# 设置路径（请根据实际数据路径修改）
origin_img_dir = os.path.join(_PROJECT_ROOT, "dataset", "raw", "yolo_detect", "3-3", "images")
origin_label_dir = os.path.join(_PROJECT_ROOT, "dataset", "raw", "yolo_detect", "3-3", "labels")
output_img_dir = os.path.join(_PROJECT_ROOT, "dataset", "raw", "unet_segment", "3-3", "images")

# 创建输出目录
os.makedirs(output_img_dir, exist_ok=True)

# 获取所有图像文件
image_files = [f for f in os.listdir(origin_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

for img_name in image_files:
    img_path = os.path.join(origin_img_dir, img_name)
    label_name = os.path.splitext(img_name)[0] + ".txt"
    label_path = os.path.join(origin_label_dir, label_name)

    # 如果没有对应标签文件，跳过
    if not os.path.exists(label_path):
        print(f"⚠️ 无标注文件: {label_name}")
        continue

    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ 无法读取图像: {img_path}")
        continue

    h, w = img.shape[:2]  # 图像高宽

    # 读取 YOLO 标注
    with open(label_path, 'r') as f:
        lines = f.readlines()

    # 遍历每个 bounding box
    for idx, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        class_id = int(parts[0])
        x_center_norm = float(parts[1])
        y_center_norm = float(parts[2])
        width_norm = float(parts[3])
        height_norm = float(parts[4])

        # 转换为像素坐标
        x_center = int(x_center_norm * w)
        y_center = int(y_center_norm * h)
        box_width = int(width_norm * w)
        box_height = int(height_norm * h)

        # 计算左上角和右下角坐标
        x1 = max(0, int(x_center - box_width / 2))
        y1 = max(0, int(y_center - box_height / 2))
        x2 = min(w, int(x_center + box_width / 2))
        y2 = min(h, int(y_center + box_height / 2))

        # 裁剪区域
        cropped = img[y1:y2, x1:x2]

        # 保存裁剪后的图像
        output_name = f"{os.path.splitext(img_name)[0]}_{idx}.png"
        output_path = os.path.join(output_img_dir, output_name)
        cv2.imwrite(output_path, cropped)

        print(f"✅ 已保存: {output_path} (尺寸: {cropped.shape})")

print("🎉 所有包裹体裁剪完成！")