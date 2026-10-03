import os
import torch
import cv2
import numpy as np
import csv
import json
import base64
import shutil
import segmentation_models_pytorch as smp

# 项目根目录（向上三级）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ========================
# 🎯 配置区域
# ========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.path.join(_PROJECT_ROOT, "weights", "unetpp_res34_best.pth")
IMAGE_DIR  = os.path.join(_PROJECT_ROOT, "dataset", "unet_segment", "sample", "images")
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "test_output", "unet_segmentation")

ENCODER = "resnet34"
NUM_CLASSES = 3
CLASS_MAP = {1: "liquid", 2: "gas"}

GEN_VIS = True
GEN_JSON = True


COLOR_MAP_VIS = {
    0: [0, 0, 0],   
    1: [255, 0, 0],  
    2: [0, 0, 255],   
}

# 创建目录（json 而不是 jsons）
for sub_dir in ['images', 'json', 'masks']:
    os.makedirs(os.path.join(OUTPUT_DIR, sub_dir), exist_ok=True)
if GEN_VIS:
    vis_dir = os.path.join(OUTPUT_DIR, "mask_vis")
    os.makedirs(vis_dir, exist_ok=True)

# ========================
# 辅助函数
# ========================
def create_labelme_json(mask_np, img_bgr, img_name, class_map, out_dir):
    h, w = img_bgr.shape[:2]
    shapes = []
    for cls_id, label in class_map.items():
        cls_mask = (mask_np == cls_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            epsilon = 0.005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            points = approx.reshape(-1, 2).tolist()
            if len(points) >= 3:
                shapes.append({
                    "label": label, "points": points, "group_id": None,
                    "shape_type": "polygon", "flags": {}
                })
    _, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_b64 = base64.b64encode(buf).decode('utf-8')
    labelme_data = {
        "version": "5.4.1", "flags": {}, "shapes": shapes,
        "imagePath": f"../images/{img_name}",
        "imageData": img_b64,
        "imageHeight": h, "imageWidth": w
    }
    json_path = os.path.join(out_dir, os.path.splitext(img_name)[0] + ".json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(labelme_data, f, indent=2, ensure_ascii=False)
    return json_path

def generate_vis_mask(mask, color_map):
    h, w = mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in color_map.items():
        vis[mask == cls_id] = color
    return vis

# ========================
# 加载模型
# ========================
print("🔄 加载 SMP 模型中...")
model = smp.UnetPlusPlus(
    encoder_name=ENCODER,
    encoder_weights=None,
    in_channels=3,
    classes=NUM_CLASSES,
    activation=None
)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    model.load_state_dict(ckpt['model_state_dict'])
else:
    model.load_state_dict(ckpt)
model.to(DEVICE).eval()
print(f"✅ 模型加载成功 (Encoder: {ENCODER})")

# ========================
# 推理主循环
# ========================
csv_path = os.path.join(OUTPUT_DIR, "results.csv")
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Image', 'Gas Pixels', 'Liquid Pixels', 'Gas/Liquid Ratio'])

    for img_name in sorted(os.listdir(IMAGE_DIR)):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img_path = os.path.join(IMAGE_DIR, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️ 读取失败: {img_name}")
            continue
        orig_h, orig_w = image.shape[:2]

        # 预处理
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = cv2.resize(img_rgb, (256, 256))
        input_tensor = torch.from_numpy(input_tensor).permute(2,0,1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        input_tensor = (input_tensor - mean) / std
        input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

        # 推理
        with torch.no_grad():
            output = model(input_tensor)
            pred_256 = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        pred_orig = cv2.resize(pred_256.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        liquid = int(np.sum(pred_orig == 1))
        gas = int(np.sum(pred_orig == 2))
        ratio = gas / liquid if liquid > 0 else float('inf')
        ratio_str = f"{ratio:.6f}" if np.isfinite(ratio) else "inf"

        # 保存必须文件
        shutil.copy2(img_path, os.path.join(OUTPUT_DIR, 'images', img_name))
        cv2.imwrite(os.path.join(OUTPUT_DIR, 'masks', os.path.splitext(img_name)[0] + ".png"), pred_orig)

        if GEN_JSON:
            create_labelme_json(pred_orig, image, img_name, CLASS_MAP, os.path.join(OUTPUT_DIR, 'json'))
        if GEN_VIS:
            vis_mask = generate_vis_mask(pred_orig, COLOR_MAP_VIS)
            vis_filename = os.path.splitext(img_name)[0] + "_vis.png"
            cv2.imwrite(os.path.join(vis_dir, vis_filename), vis_mask)

        writer.writerow([img_name, gas, liquid, ratio_str])
        print(f"✅ {img_name} | 气={gas}, 液={liquid}, 比={ratio_str}")

print(f"\n🎉 完成！输出目录: {OUTPUT_DIR}")