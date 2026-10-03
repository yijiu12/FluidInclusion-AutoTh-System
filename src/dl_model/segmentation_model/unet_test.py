import os
import torch
import cv2
import numpy as np
import json
import time
from datetime import datetime
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 强制 CPU 将下溢出的浮点数直接置为 0，防止 FPU 速度暴降
torch.set_flush_denormal(True)

# 项目根目录（向上三级）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

if __name__ == '__main__':
    # ==================== 配置区域 ====================
    BEST_WEIGHTS = os.path.join(_PROJECT_ROOT, "weights", "unetpp_res34_best.pth")
    TEST_IMG_DIR = os.path.join(_PROJECT_ROOT, "dataset", "unet_segment", "sample", "images")
    TEST_MASK_DIR = os.path.join(_PROJECT_ROOT, "dataset", "unet_segment", "sample", "masks")
    SAVE_DIR = os.path.join(_PROJECT_ROOT, "training_output", "unetpp_res34", "unet_test_results")
    
    NUM_CLASSES = 3     # 0:背景, 1:液相, 2:气相
    IMG_SIZE = 256
    ENCODER = "resnet34"
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 评估用 GPU（若可用），测速强制 CPU
    EVAL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SPEED_DEVICE = torch.device("cpu")

    # ==================== 预处理配置 ====================
    test_transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})

    def find_mask_path(img_name, mask_dir):
        base = os.path.splitext(img_name)[0]
        for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            p = os.path.join(mask_dir, base + ext)
            if os.path.exists(p):
                return p
        return None

    # ==================== 加载模型（评估用） ====================
    print("🔄 加载模型中...")
    model = smp.UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None
    )
    ckpt = torch.load(BEST_WEIGHTS, map_location=EVAL_DEVICE, weights_only=False)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model = model.to(EVAL_DEVICE).eval()
    print(f"✅ 模型加载成功：{BEST_WEIGHTS}")
    print(f"   评估设备：{EVAL_DEVICE}")

    # ==================== 1. 测试集标准评估（GPU，内存友好） ====================
    print("\n🚀 开始测试集评估...")
    test_images = [f for f in os.listdir(TEST_IMG_DIR) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    
    for img_name in test_images:
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        mask_path = find_mask_path(img_name, TEST_MASK_DIR)
        if mask_path is None:
            print(f"⚠️ 跳过无对应Mask的图像：{img_name}")
            continue
            
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        aug = test_transform(image=img, mask=mask)
        img_tensor = aug['image'].unsqueeze(0).to(EVAL_DEVICE)
        mask_np = aug['mask']

        if torch.is_tensor(mask_np):
            mask_np = mask_np.cpu().numpy()

        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        
        pred_flat = pred.flatten().astype(np.int64)
        target_flat = mask_np.flatten().astype(np.int64)
        cm += np.bincount(
            NUM_CLASSES * target_flat + pred_flat,
            minlength=NUM_CLASSES**2
        ).reshape(NUM_CLASSES, NUM_CLASSES)

    # 计算指标
    ious, dices = [], []
    for cls in range(NUM_CLASSES):
        tp = cm[cls, cls]
        fp = cm[:, cls].sum() - tp
        fn = cm[cls, :].sum() - tp
        union = tp + fp + fn
        iou = tp / union if union > 0 else 1.0
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
        ious.append(iou)
        dices.append(dice)

    metrics = {
        'miou': np.mean(ious),
        'dice': np.mean(dices),
        'pixel_acc': np.diag(cm).sum() / cm.sum(),
        'class_ious': ious
    }


    # ==================== 2. CPU 推理速度评估 ====================
    print("\n⏱️  计算纯粹的 CPU 推理速度...")
    
    # 强制限制 PyTorch CPU 线程数（通常设为 4 或 8 性能最稳定）
    torch.set_num_threads(4) 
    
    model_cpu = smp.UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None
    )
    model_cpu.load_state_dict(state_dict)
    model_cpu = model_cpu.to(SPEED_DEVICE).eval()
    
    test_images_speed = [f for f in os.listdir(TEST_IMG_DIR) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:50]

    # --- 提前把所有图片读入内存并预处理成 Tensor，避免 I/O 干扰 ---
    tensor_list = []
    for img_name in test_images_speed:
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        aug = test_transform(image=img, mask=np.zeros(img.shape[:2], dtype=np.uint8))
        img_tensor = aug['image'].unsqueeze(0).to(SPEED_DEVICE)
        tensor_list.append(img_tensor)

    # --- 模型预热 (Warm-up) ---
    # 跑前几张图时不计时，让 CPU 缓存和 PyTorch 内部完成初始化
    with torch.no_grad():
        for i in range(5):
            _ = model_cpu(tensor_list[0])

    # --- 正式测速 ---
    total_time = 0.0
    for img_tensor in tensor_list:
        start = time.time() # 仅对推理过程计时
        with torch.no_grad():
            output = model_cpu(img_tensor)
            _ = torch.argmax(output, dim=1)
        total_time += time.time() - start

    avg_time_ms = (total_time / len(tensor_list)) * 1000
    fps = len(tensor_list) / total_time

    # ==================== 输出结果 ====================
    print("\n" + "="*65)
    print("📊 测试集评估结果")
    print("="*65)
    print(f"像素准确率 Pixel Acc: {metrics['pixel_acc']:.4f}")
    print(f"平均 IoU (mIoU):      {metrics['miou']:.4f}")
    print(f"平均 Dice (mDice):    {metrics['dice']:.4f}")
    print("-"*65)
    for i, iou in enumerate(metrics['class_ious']):
        print(f"类别 {i} IoU:          {iou:.4f}")
    print("-"*65)
    print(f"CPU 平均单图耗时:    {avg_time_ms:.2f} ms")
    print(f"CPU 推理速度:        {fps:.2f} 帧/秒")
    print("="*65)

    # ==================== 保存报告 ====================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(SAVE_DIR, f'evaluation_report_end_{timestamp}.json')

    report = {
        "model_path": BEST_WEIGHTS,
        "eval_time": timestamp,
        "metrics": {
            "pixel_accuracy": round(float(metrics['pixel_acc']), 4),
            "mIoU": round(float(metrics['miou']), 4),
            "mDice": round(float(metrics['dice']), 4),
            "class_IoU": {f"class_{i}": round(float(v), 4) for i, v in enumerate(metrics['class_ious'])}
        },
        "cpu_speed": {
            "avg_time_ms": round(avg_time_ms, 2),
            "fps": round(fps, 2)
        }
    }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    print(f"\n📄 评估报告已保存：{report_path}")