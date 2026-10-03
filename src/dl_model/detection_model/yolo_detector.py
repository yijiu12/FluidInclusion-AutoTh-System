import os
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 项目根目录（向上三级）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# === 配置 ===
model_path = os.path.join(_PROJECT_ROOT, "weights", "yolo26n_best.pt")
input_folder = os.path.join(_PROJECT_ROOT, "dataset", "yolo_detect", "sample", "images")  # 输入图片文件夹
output_base = os.path.join(_PROJECT_ROOT, "test_output", "yolo_detection")

# 创建输出目录
os.makedirs(os.path.join(output_base, 'images'), exist_ok=True)
os.makedirs(os.path.join(output_base, 'images_with_boxes'), exist_ok=True)
os.makedirs(os.path.join(output_base, 'labels'), exist_ok=True)

# === 加载模型 ===
print("🚀 加载YOLO模型...")
model = YOLO(model_path)
print("✅ 模型加载完成")

# === 获取输入文件夹中的所有图片文件 ===
supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
image_files = [f for f in os.listdir(input_folder) 
               if f.lower().endswith(supported_formats)]

print(f"📁 找到 {len(image_files)} 张图片需要处理")

if len(image_files) == 0:
    print("❌ 输入文件夹中没有找到支持的图片文件")
    exit()

# === 批量处理函数 ===
def process_single_image(image_path, output_base, model):
    """处理单张图片"""
    try:
        # 读取图片
        orig_img = cv2.imread(image_path)
        if orig_img is None:
            print(f"❌ 无法读取图片: {image_path}")
            return False
        
        img_filename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(img_filename)[0]
        
        # 使用模型预测
        results = model.predict(
            source=image_path,
            conf=0.5,
            verbose=False
        )
        
        result = results[0]  # 取第一个结果
        
        # --- 1. 保存原始无框图像到 images/ ---
        output_img_path = os.path.join(output_base, 'images', img_filename)
        cv2.imwrite(output_img_path, orig_img)
        
        # --- 2. 使用 PIL 自定义绘制带框图像（小字体）---
        img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        
        # 加载小字体
        try:
            font = ImageFont.truetype("arial.ttf", 10)  # 字体大小设为10
        except:
            try:
                font = ImageFont.truetype("Arial.ttf", 10)
            except:
                # 如果找不到字体，使用默认字体
                font = ImageFont.load_default()
        
        boxes = result.boxes
        detection_count = 0
        
        if boxes is not None and len(boxes) > 0:
            detection_count = len(boxes)
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0].cpu().numpy())
                conf = box.conf[0].cpu().numpy()
                
                # 绘制矩形框
                draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
                
                # 绘制标签（小字体）
                label = f"{model.names[cls_id]} {conf:.2f}"
                # 获取文本大小
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # 绘制文本背景
                draw.rectangle([x1, y1-text_height-4, x1+text_width+4, y1], fill='red')
                # 绘制文本
                draw.text((x1+2, y1-text_height-2), label, fill='white', font=font)
        
        # 转换回 OpenCV 格式并保存
        plotted_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        output_boxed_path = os.path.join(output_base, 'images_with_boxes', img_filename)
        cv2.imwrite(output_boxed_path, plotted_img)
        
        # --- 3. 保存YOLO格式标签 ---
        if boxes is not None and len(boxes) > 0:
            H, W = result.orig_shape
            label_filename = name_without_ext + '.txt'
            output_label_path = os.path.join(output_base, 'labels', label_filename)
            
            with open(output_label_path, 'w') as f:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls_id = int(box.cls[0].cpu().numpy())
                    x_center = (x1 + x2) / (2 * W)
                    y_center = (y1 + y2) / (2 * H)
                    width = (x2 - x1) / W
                    height = (y2 - y1) / H
                    f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        print(f"✅ 处理完成: {img_filename} | 检测到 {detection_count} 个目标")
        return True
        
    except Exception as e:
        print(f"❌ 处理图片 {image_path} 时出错: {str(e)}")
        return False

# === 批量处理所有图片 ===
print("🔄 开始批量处理图片...")
success_count = 0

for i, image_file in enumerate(image_files, 1):
    image_path = os.path.join(input_folder, image_file)
    print(f"\n📄 处理第 {i}/{len(image_files)} 张图片: {image_file}")
    
    if process_single_image(image_path, output_base, model):
        success_count += 1

# === 输出统计信息 ===
print("\n" + "="*50)
print("🎉 批量处理完成！")
print(f"📊 统计信息:")
print(f"   总图片数: {len(image_files)}")
print(f"   成功处理: {success_count}")
print(f"   失败: {len(image_files) - success_count}")
print(f"\n📁 输出文件保存在: {output_base}")
print(f"   - 原始图片: {output_base}/images/")
print(f"   - 带框图片: {output_base}/images_with_boxes/")
print(f"   - 标签文件: {output_base}/labels/")
print("="*50)