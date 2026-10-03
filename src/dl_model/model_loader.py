"""AI模型层：全局形态匹配、多候选检测、单框形态计算、ROI气液比分析。"""

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import segmentation_models_pytorch as smp

import config

_yolo_model = None
_unet_model = None

torch.set_flush_denormal(True)


def load_models():
    global _yolo_model, _unet_model
    print("\n🔄 加载YOLO目标检测模型...")
    _yolo_model = YOLO(config.YOLO_MODEL_PATH)
    print("✅ YOLO模型加载完成")

    print("🔄 加载UNet++语义分割模型...")
    _unet_model = smp.UnetPlusPlus(
        encoder_name=config.UNET_ENCODER,
        encoder_weights=None,
        in_channels=3,
        classes=config.NUM_CLASSES,
        activation=None
    )
    ckpt = torch.load(config.UNET_MODEL_PATH, map_location=config.DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        _unet_model.load_state_dict(ckpt['model_state_dict'])
    else:
        _unet_model.load_state_dict(ckpt)
    _unet_model.to(config.DEVICE).eval()
    print(f"✅ UNet++模型加载成功，设备：{config.DEVICE}")


def detect_all_inclusions(img_array: np.ndarray, conf_thresh: float = config.CONF_THRESH):
    """
    获取所有YOLO检测候选，按置信度降序排列。
    返回: list of {'box': (x1,y1,x2,y2), 'conf': float}
    """
    if _yolo_model is None:
        raise RuntimeError("YOLO模型未加载")
    
    orig = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    orig_h, orig_w = orig.shape[:2]
    results = _yolo_model.predict(orig, conf=conf_thresh, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    candidates = []
    for box in boxes:
        conf = float(box.conf[0].cpu().numpy())
        x1, y1, x2, y2 = map(float, box.xyxy[0].cpu().numpy())
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(orig_w, x2); y2 = min(orig_h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        candidates.append({'box': (x1, y1, x2, y2), 'conf': conf})

    return sorted(candidates, key=lambda x: x['conf'], reverse=True)


def choose_best_candidate(img_array: np.ndarray, candidates: list, inclusion_data: dict):
    """
    从多个候选中用形态匹配选最佳。
    若形态匹配分低于 MORPH_MATCH_THRESHOLD，直接返回失败（None），不再回退到置信度。
    返回: (best_box, best_conf, source_str) 或 (None, 0.0, "morph_fail")
    """
    target_ar = float(inclusion_data.get('长宽比', 0))
    target_circ = float(inclusion_data.get('圆度', 0))
    target_diam = float(inclusion_data.get('等效直径_um', 0))

    if target_ar <= 0 or target_diam <= 0:
        # 无形态参数，无法做形态匹配，直接失败
        return None, 0.0, "morph_fail_no_params"

    best_match = -1.0
    best_candidate = None

    for c in candidates:
        ok, cand_ar, cand_circ, cand_diam = calculate_morphology_for_box(
            img_array, c['box']
        )
        if not ok:
            continue

        ar_sim = min(cand_ar, target_ar) / max(cand_ar, target_ar)
        circ_sim = 1 - abs(cand_circ - target_circ)
        diam_sim = min(cand_diam, target_diam) / max(cand_diam, target_diam)
        match_score = 0.4 * ar_sim + 0.3 * circ_sim + 0.3 * diam_sim

        if match_score > best_match:
            best_match = match_score
            best_candidate = c

    if best_candidate and best_match >= getattr(config, 'MORPH_MATCH_THRESHOLD', 0.0):
        return best_candidate['box'], best_candidate['conf'], f"morph_{best_match:.3f}"
    else:
        # 形态匹配失败或低于阈值，不再回退到置信度，直接失败
        return None, 0.0, "morph_fail"


def detect_target_inclusion(img_array: np.ndarray, inclusion_data: dict, conf_thresh: float = config.CONF_THRESH):
    """
    全局形态匹配：在整张图像中找到与数据库目标形态最相似的包裹体。
    用于初始定位和回中。无距离阈值限制，纯形态匹配。
    
    返回: (ok, box, conf, match_score)
    """
    candidates = detect_all_inclusions(img_array, conf_thresh)
    if not candidates:
        return False, None, 0.0, 0.0
    
    best_box, best_conf, source = choose_best_candidate(img_array, candidates, inclusion_data)
    
    if best_box is None:
        # 形态匹配失败（低于阈值或无参数）
        return False, None, 0.0, 0.0
    
    # 重新计算最佳候选的match_score用于返回
    ok, cand_ar, cand_circ, cand_diam = calculate_morphology_for_box(img_array, best_box)
    if ok:
        target_ar = float(inclusion_data.get('长宽比', 0))
        target_circ = float(inclusion_data.get('圆度', 0))
        target_diam = float(inclusion_data.get('等效直径_um', 0))
        ar_sim = min(cand_ar, target_ar) / max(cand_ar, target_ar)
        circ_sim = 1 - abs(cand_circ - target_circ)
        diam_sim = min(cand_diam, target_diam) / max(cand_diam, target_diam)
        match_score = 0.4 * ar_sim + 0.3 * circ_sim + 0.3 * diam_sim
    else:
        match_score = 0.0
    
    return True, best_box, best_conf, match_score


def calculate_morphology_for_box(img_array: np.ndarray, bbox_xyxy: tuple):
    """对单个候选框实时分割，计算形态参数。
    返回: (success, aspect_ratio, circularity, equiv_diameter)"""
    if _unet_model is None:
        raise RuntimeError("UNet模型未加载")
    
    x1, y1, x2, y2 = map(int, bbox_xyxy)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_array.shape[1], x2), min(img_array.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return False, None, None, None
    
    roi_img = img_array[y1:y2, x1:x2]
    if roi_img.size == 0:
        return False, None, None, None
    
    roi_rgb = cv2.cvtColor(roi_img, cv2.COLOR_RGB2BGR)
    roi_rgb = cv2.cvtColor(roi_rgb, cv2.COLOR_BGR2RGB)
    roi_resized = cv2.resize(roi_rgb, (config.UNET_INPUT_SIZE, config.UNET_INPUT_SIZE))
    
    input_tensor = torch.from_numpy(roi_resized).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    input_tensor = (input_tensor - mean) / std
    input_tensor = input_tensor.unsqueeze(0).to(config.DEVICE)
    
    with torch.no_grad():
        output = _unet_model(input_tensor)
        pred_256 = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    
    roi_h, roi_w = roi_img.shape[:2]
    pred_orig = cv2.resize(pred_256.astype(np.uint8), (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
    
    inclusion_mask = ((pred_orig == 1) | (pred_orig == 2)).astype(np.uint8)
    contours, _ = cv2.findContours(inclusion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, None, None, None
    
    contour = max(contours, key=cv2.contourArea)
    area_px = cv2.contourArea(contour)
    area_um2 = area_px * (config.UM_PER_PIXEL ** 2)
    equiv_diameter = 2 * np.sqrt(area_um2 / np.pi) if area_um2 > 0 else 0
    
    rect = cv2.minAreaRect(contour)
    w, h = rect[1]
    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1.0
    
    perimeter = cv2.arcLength(contour, True)
    if perimeter > 0:
        circularity = 4 * np.pi * area_px / (perimeter ** 2)
        circularity = max(0, min(1, circularity))
    else:
        circularity = 0
    
    return True, aspect_ratio, circularity, equiv_diameter


def analyze_locked_roi(img_array: np.ndarray, roi: tuple | None,
                       vis_save_path: str | None = None,
                       roi_raw_path: str | None = None):
    """对锁定ROI进行语义分割，计算气液比。
    返回: (ratio, gas_pixels, liquid_pixels, total_pixels, success)"""
    if roi is None or _unet_model is None:
        return None, 0, 0, 0, False
    
    orig = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    orig_h, orig_w = orig.shape[:2]
    x1, y1, x2, y2 = map(int, roi)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(orig_w, x2); y2 = min(orig_h, y2)
    roi_img = orig[y1:y2, x1:x2]
    if roi_img.size == 0:
        return None, 0, 0, 0, False
    roi_h, roi_w = roi_img.shape[:2]

    roi_rgb = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
    roi_resized = cv2.resize(roi_rgb, (config.UNET_INPUT_SIZE, config.UNET_INPUT_SIZE))
    input_tensor = torch.from_numpy(roi_resized).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    input_tensor = (input_tensor - mean) / std
    input_tensor = input_tensor.unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        output = _unet_model(input_tensor)
        pred_256 = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    pred_orig_size = cv2.resize(pred_256.astype(np.uint8), (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
    liquid_pixels = int(np.sum(pred_orig_size == 1))
    gas_pixels = int(np.sum(pred_orig_size == 2))
    total_pixels = liquid_pixels + gas_pixels

    if liquid_pixels > 0:
        ratio = gas_pixels / liquid_pixels
    else:
        ratio = float('inf') if gas_pixels > 0 else 0.0

    if roi_raw_path is not None:
        roi_rgb_raw = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
        pil_roi_raw = Image.fromarray(roi_rgb_raw)
        pil_roi_raw.save(roi_raw_path, quality=95)

    if vis_save_path is not None:
        color_mask = np.zeros((roi_h, roi_w, 3), dtype=np.uint8)
        color_mask[pred_orig_size == 1] = config.COLOR_LIQUID
        color_mask[pred_orig_size == 2] = config.COLOR_GAS
        overlay = cv2.addWeighted(roi_img, 0.6, color_mask, 0.4, 0)
        pil_roi = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_roi)
        try:
            font = ImageFont.truetype("simhei.ttf", 12)
        except:
            font = ImageFont.load_default()
        ratio_str = f"{ratio:.4f}" if np.isfinite(ratio) else "inf"
        text = f"G/L={ratio_str} | G={gas_pixels} L={liquid_pixels}"
        draw.text((5, 5), text, fill=(255, 255, 255), font=font)
        pil_roi.save(vis_save_path, quality=95)

    return ratio, gas_pixels, liquid_pixels, total_pixels, True