import os
import json
import numpy as np
import cv2
import glob
import math

# 项目根目录（向上四级）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# 设置路径（请根据实际数据路径修改）
image_dir = os.path.join(_PROJECT_ROOT, "dataset", "raw", "unet_segment", "3-3", "images")
json_dir = os.path.join(_PROJECT_ROOT, "dataset", "raw", "unet_segment", "3-3", "json")
output_dir = os.path.join(_PROJECT_ROOT, "dataset", "raw", "unet_segment", "3-3", "masks")
vis_output_dir = os.path.join(_PROJECT_ROOT, "dataset", "raw", "unet_segment", "3-3", "mask_vis")

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)
os.makedirs(vis_output_dir, exist_ok=True)

# 类别映射（你定义的：1=液相，2=气相，0=背景）
CLASS_MAPPING = {
    'liquid': 1,
    'gas': 2
}

def create_fluid_mask(json_path, image_shape):
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for shape in data['shapes']:
        label = shape['label']
        points = shape['points']
        shape_type = shape.get('shape_type', 'polygon')
        
        if label not in CLASS_MAPPING:
            print(f"  ⚠️ 未知标签: {label}")
            continue
        
        class_value = CLASS_MAPPING[label]
        
        # ---------------------- 修复：圆形标注 ----------------------
        if shape_type == 'circle':
            if len(points) < 2:
                print(f"  ⚠️ circle 标注点不足")
                continue
            cx, cy = int(points[0][0]), int(points[0][1])
            px, py = points[1]
            radius = int(math.hypot(px - cx, py - cy))
            # 直接画实心圆（最稳定）
            cv2.circle(mask, (cx, cy), radius, class_value, thickness=-1)
        
        # ---------------------- 修复：多边形标注 ----------------------
        elif shape_type == 'polygon':
            if len(points) < 3:
                print(f"  ⚠️ polygon 点数不足")
                continue
            pts = np.array(points, dtype=np.int32)
            # 直接填充，不要手动闭合
            cv2.fillPoly(mask, [pts], class_value)
            
        else:
            print(f"  ⚠️ 不支持的 shape_type: {shape_type}")
    
    return mask

def main():
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    print(f"✅ 找到 {len(json_files)} 个 JSON 标注文件")
    print(f"类别映射: 背景=0, {', '.join([f'{k}={v}' for k, v in CLASS_MAPPING.items()])}")
    
    for json_path in json_files:
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        print(f"🔄 处理: {base_name}")
        
        image_found = False
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
            img_path = os.path.join(image_dir, base_name + ext)
            if os.path.exists(img_path):
                image = cv2.imread(img_path)
                if image is None:
                    print(f"  ❌ 无法读取图像: {base_name + ext}")
                    continue
                
                # 生成 mask
                mask = create_fluid_mask(json_path, image.shape)
                
                # 保存单通道训练用 mask
                mask_path = os.path.join(output_dir, base_name + '.png')
                cv2.imwrite(mask_path, mask)
                
                # 可视化彩色 mask
                color_map = {
                    0: [0, 0, 0],       # 背景 
                    1: [255, 0, 0],     # 液相 蓝色
                    2: [0, 0, 255]      # 气相 红色
                }
                vis_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                for val, color in color_map.items():
                    vis_mask[mask == val] = color
                
                vis_path = os.path.join(vis_output_dir, base_name + '_vis.png')
                cv2.imwrite(vis_path, vis_mask)
                
                unique_vals = np.unique(mask)
                print(f"  ✅ 保存成功 | mask 数值: {unique_vals}")
                image_found = True
                break
        
        if not image_found:
            print(f"  ❌ 找不到对应图像: {base_name}")
    
    print("\n🎉 所有 mask 生成完毕！")

if __name__ == "__main__":
    main()