# ============================================================================
# 流体包裹体均一温度自动测量系统 - 扫描分析一体化
# ============================================================================
# 功能：扫描薄片全区域 → 截图 → 目标检测 → 语义分割 → 气液比统计 → 输出表格
# 整合自：temp_XYZ.py（扫描采集）+ AutoGUI.py（自动化控制）+ mbjc_yyfg.py（模型推理）
# ============================================================================

import os
import time
import json
import csv
import base64
import shutil
import ctypes
from ctypes import c_long, c_uint, c_ubyte
from datetime import datetime
import pyautogui
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import segmentation_models_pytorch as smp

torch.set_flush_denormal(True)

# ========================
# 🔧 配置区域 - 请根据实际环境修改
# ========================

# 项目根目录（基于本文件位置向上两级）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- 文件路径（相对项目根目录）---
Z_AXIS_DLL = os.path.join(_PROJECT_ROOT, "src", "hardware", "lib", "DTStageDriver.dll")
LINKAM_DLL = os.path.join(_PROJECT_ROOT, "src", "hardware", "lib", "LinkamIpc.dll")
YOLO_MODEL_PATH = os.path.join(_PROJECT_ROOT, "weights", "yolo26n_best.pt")
UNET_MODEL_PATH = os.path.join(_PROJECT_ROOT, "weights", "unetpp_res34_best.pth")

# --- 输出目录 ---
OUTPUT_BASE = os.path.join(_PROJECT_ROOT, "output", "scan_results")

# --- 串口与速度配置 ---
COM_PORT = 3
Z_SPEED = 300
XY_MOVE_SPEED = 200  # 微米/秒，用于计算移动等待时间

# --- 扫描范围配置（微米）---
X_START = 0
X_END = -3000
X_STEP = -300

Y_START = 0
Y_END = 3000
Y_STEP = 250

Z_START = 0
Z_END = -1400
Z_STEP = -200

# --- 界面坐标（1920x1080分辨率）---
NEXUS_POS = (1206, 1057)
SCREEN_REGION = (348, 151, 786, 781)
X_INPUT_POS = (1584, 488)
Y_INPUT_POS = (1585, 521)
ZERO_BTN_POS = (1594, 403)

# --- Z轴安全范围 ---
Z_LIMIT_MIN = -10000
Z_LIMIT_MAX = 0
XY_LIMIT_MIN = -7500
XY_LIMIT_MAX = 7500

# --- 模型推理配置 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRESH = 0.5
UNET_ENCODER = "resnet34"
NUM_CLASSES = 3  # 0:背景, 1:液相(liquid), 2:气相(gas)
CLASS_MAP = {1: "liquid", 2: "gas"}
UNET_INPUT_SIZE = 256

# --- 颜色配置 (BGR格式) ---
COLOR_GAS = (0, 0, 255)      # 红色 - 气相
COLOR_LIQUID = (255, 0, 0)   # 蓝色 - 液相
COLOR_BOX = (0, 165, 255)    # 橙色 - 检测框

# ========================
# 🎛️ 输出控制开关（True/False控制是否保存该类文件）
# ========================
SAVE_ORIGINAL_IMAGE = True        # 保存原始截图
SAVE_BOXED_IMAGE = True           # 保存带检测框的图像
SAVE_ROI_VISUALIZATION = True     # 保存ROI分割彩色叠加图
SAVE_ROI_MASK = True              # 保存ROI分割掩码PNG
SAVE_LABELME_JSON = True          # 保存LabelMe预标注JSON
SAVE_YOLO_LABELS = True           # 保存YOLO格式标签
SAVE_PER_IMAGE_CSV = True         # 保存每张图的ROI统计CSV

# ========================
# 📊 全局变量
# ========================
X_POS = 0
Y_POS = 0
Z_POS = 0
z_connected = False
temp_connected = False
dt_driver = None
temp_client = None
nexus_activated = False

# 模型全局变量
yolo_model = None
unet_model = None

# 汇总数据
all_results = []  # 存储所有包裹体的信息

# ========================
# 一、界面控制模块
# ========================

def activate_nexus_once():
    """激活NEXUS软件窗口（只执行一次）"""
    global nexus_activated
    if not nexus_activated:
        pyautogui.click(NEXUS_POS[0], NEXUS_POS[1])
        time.sleep(0.3)
        nexus_activated = True

def simple_click(pos):
    """简单点击并等待"""
    pyautogui.click(pos[0], pos[1])
    time.sleep(0.2)

def wait_ui(delay=0.5):
    """等待UI响应"""
    time.sleep(delay)

# ========================
# 二、温度数据读取模块
# ========================

def init_temp():
    """初始化温度读取（IPC通信）"""
    global temp_client, temp_connected
    try:
        import clr
        clr.AddReference(LINKAM_DLL)
        from Linkam.IpcExtension import IPCClient
        temp_client = IPCClient()
        temp_client.Start()
        time.sleep(1.5)
        if temp_client.GrabHeader():
            temp_connected = True
            print("✅ 温度读取已连接")
            return True
    except Exception as e:
        print(f"❌ 温度连接失败：{e}")
    return False

def close_temp():
    """关闭温度读取"""
    global temp_client, temp_connected
    if temp_client:
        try:
            temp_client.Stop()
            print("🔌 温度读取已断开")
        except:
            pass
    temp_connected = False

def get_temp():
    """获取当前温度"""
    global temp_client
    if not temp_connected:
        return None
    try:
        header = temp_client.GrabHeader()
        if not header:
            return None
        data = temp_client.GrabData()
        titles = [t.strip() for t in header.split('\n')[1].split(',') if t.strip()]
        values = [float(v.strip()) for v in data.split(',') if v.strip()]
        if "Temp" in titles:
            temp = values[titles.index("Temp")]
            if temp == temp:  # 排除NaN
                return round(temp, 2)
    except Exception as e:
        print(f"⚠️ 温度读取异常：{e}")
    return None

# ========================
# 三、Z轴控制模块
# ========================

def init_z_axis():
    """初始化Z轴控制器"""
    global dt_driver, z_connected
    try:
        dt_driver = ctypes.CDLL(Z_AXIS_DLL)
        result = dt_driver.DTSetComm(c_long(COM_PORT))
        if result == 1:
            dt_driver.SetSpeed(c_ubyte(ord('Z')), c_uint(Z_SPEED))
            z_connected = True
            print(f"✅ Z轴已连接 (COM{COM_PORT})")
            return True
        else:
            print("❌ Z轴连接失败")
            return False
    except Exception as e:
        print(f"❌ Z轴加载失败：{e}")
        return False

def close_z_axis():
    """关闭Z轴控制器"""
    global z_connected
    z_connected = False
    print("🔌 Z轴已断开")

def move_z_absolute(target_z):
    """Z轴移动到绝对位置"""
    global Z_POS
    if not z_connected:
        return False
    delta = target_z - Z_POS
    if delta == 0:
        return True

    # 安全检查
    if target_z > Z_LIMIT_MAX:
        print(f"❌ Z轴目标 {target_z} 超过上限 {Z_LIMIT_MAX}，禁止移动")
        return False
    if target_z < Z_LIMIT_MIN:
        print(f"❌ Z轴目标 {target_z} 低于下限 {Z_LIMIT_MIN}，禁止移动")
        return False

    dir_val = 1 if delta > 0 else -1
    dt_driver.MoveStageNoWait(c_ubyte(ord('Z')), c_long(dir_val), c_long(abs(delta)))
    time.sleep(1.5)
    Z_POS = target_z
    print(f"Z = {Z_POS}")
    return True

def get_z_position():
    """获取Z轴当前位置"""
    return Z_POS

# ========================
# 四、XY轴控制模块
# ========================

def move_time(target, current):
    """计算XY轴移动所需时间"""
    d = abs(target - current)
    if d == 0:
        return 0.5
    return (d / XY_MOVE_SPEED) + 0.5

def set_axis(axis, pos, target, current):
    """设置XY轴位置"""
    global X_POS, Y_POS
    t = move_time(target, current)
    for _ in range(2):
        try:
            simple_click(pos)
            pyautogui.hotkey('ctrl', 'a')
            pyautogui.write(str(target))
            pyautogui.press('enter')
            time.sleep(t)
            if axis == "X":
                X_POS = target
            else:
                Y_POS = target
            print(f"{axis} = {target}")
            return True
        except:
            time.sleep(0.5)
    print(f"❌ {axis} 轴移动失败")
    return False

def set_x(x):
    """设置X轴位置（带安全校验）"""
    if not (XY_LIMIT_MIN <= x <= XY_LIMIT_MAX):
        print(f"❌ X轴 {x} 超出范围 [{XY_LIMIT_MIN}, {XY_LIMIT_MAX}]")
        return False
    return set_axis("X", X_INPUT_POS, x, X_POS)

def set_y(y):
    """设置Y轴位置（带安全校验）"""
    if not (XY_LIMIT_MIN <= y <= XY_LIMIT_MAX):
        print(f"❌ Y轴 {y} 超出范围 [{XY_LIMIT_MIN}, {XY_LIMIT_MAX}]")
        return False
    return set_axis("Y", Y_INPUT_POS, y, Y_POS)

def zero_xy():
    """XY轴归零"""
    activate_nexus_once()
    simple_click(ZERO_BTN_POS)
    time.sleep(6)
    global X_POS, Y_POS
    X_POS = Y_POS = 0
    print("✅ XY轴已归零")

def get_xy_position():
    """获取XY轴当前位置"""
    return X_POS, Y_POS

# ========================
# 五、图像采集模块
# ========================

def take_screenshot(save_path):
    """截取显微镜视野图像"""
    try:
        img = pyautogui.screenshot(region=SCREEN_REGION)
        img.convert("RGB").save(save_path, quality=100)
        print(f"📸 截图已保存：{save_path}")
        return True
    except Exception as e:
        print(f"❌ 截图失败：{e}")
        return False

# ========================
# 六、模型加载模块
# ========================

def load_models():
    """加载YOLO目标检测模型和UNet++语义分割模型"""
    global yolo_model, unet_model

    print("\n🔄 加载YOLO目标检测模型...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("✅ YOLO模型加载完成")

    print("🔄 加载SMP UNet++语义分割模型...")
    unet_model = smp.UnetPlusPlus(
        encoder_name=UNET_ENCODER,
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None
    )

    ckpt = torch.load(UNET_MODEL_PATH, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        unet_model.load_state_dict(ckpt['model_state_dict'])
    else:
        unet_model.load_state_dict(ckpt)

    unet_model.to(DEVICE).eval()
    print(f"✅ UNet++模型加载成功 (Encoder: {UNET_ENCODER}, Classes: {NUM_CLASSES})")
    print(f"✅ 使用设备：{DEVICE}")

# ========================
# 七、辅助函数
# ========================

def load_font(size=10):
    """加载字体（支持中文）"""
    font_candidates = [
        ("simhei.ttf", size),
        ("simsun.ttc", size),
        ("msyh.ttc", size),
        ("arial.ttf", size),
    ]
    for font_name, font_size in font_candidates:
        try:
            return ImageFont.truetype(font_name, font_size)
        except:
            continue
    return ImageFont.load_default()

def create_labelme_json(mask_np, img_bgr, img_name, class_map, out_dir):
    """生成LabelMe格式的JSON标注文件"""
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

# ========================
# 八、单张图像分析模块（核心）
# ========================

def analyze_image(img_path, img_name, output_base, x_pos, y_pos, z_pos, temperature):
    """
    对单张图像进行目标检测和语义分割分析

    参数:
        img_path: 图像路径
        img_name: 图像名称（不含扩展名）
        output_base: 输出根目录
        x_pos, y_pos, z_pos: 当前XYZ位置
        temperature: 当前温度

    返回:
        list: 该图像中所有检测到的包裹体信息列表
    """
    global all_results

    # 读取图像
    orig = cv2.imread(img_path)
    if orig is None:
        print(f"⚠️ 无法读取图片：{img_path}")
        return []

    orig_h, orig_w = orig.shape[:2]

    # 创建输出子目录
    subdirs = {
        'images': os.path.join(output_base, 'images'),
        'images_with_boxes': os.path.join(output_base, 'images_with_boxes'),
        'roi_visualizations': os.path.join(output_base, 'roi_visualizations'),
        'roi_masks': os.path.join(output_base, 'roi_masks'),
        'jsons': os.path.join(output_base, 'jsons'),
        'labels': os.path.join(output_base, 'labels'),
        'csvs': os.path.join(output_base, 'csvs'),
    }
    for d in subdirs.values():
        os.makedirs(d, exist_ok=True)

    # 保存原始图像
    if SAVE_ORIGINAL_IMAGE:
        shutil.copy2(img_path, os.path.join(subdirs['images'], os.path.basename(img_path)))

    # === 1. YOLO目标检测 ===
    results = yolo_model.predict(img_path, conf=CONF_THRESH, verbose=False)
    boxes = results[0].boxes

    image_results = []

    if boxes is None or len(boxes) == 0:
        print(f"⚠️ {img_name}: 未检测到包裹体")
        if SAVE_BOXED_IMAGE:
            cv2.imwrite(os.path.join(subdirs['images_with_boxes'], f"{img_name}.jpg"), orig)
        return []

    # === 2. 使用PIL绘制带框图像 ===
    img_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font_small = load_font(10)

    # === 3. 处理每个ROI ===
    per_image_results = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())

        # 边界裁剪
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(orig_w, x2)
        y2 = min(orig_h, y2)
        roi = orig[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        roi_h, roi_w = roi.shape[:2]

        # === 3.1 语义分割预处理 ===
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi_resized = cv2.resize(roi_rgb, (UNET_INPUT_SIZE, UNET_INPUT_SIZE))

        input_tensor = torch.from_numpy(roi_resized).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        input_tensor = (input_tensor - mean) / std
        input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

        # === 3.2 推理 ===
        with torch.no_grad():
            output = unet_model(input_tensor)
            pred_256 = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        # === 3.3 恢复原始尺寸 ===
        pred_orig_size = cv2.resize(
            pred_256.astype(np.uint8),
            (roi_w, roi_h),
            interpolation=cv2.INTER_NEAREST
        )

        # === 3.4 统计像素 ===
        liquid_pixels = int(np.sum(pred_orig_size == 1))
        gas_pixels = int(np.sum(pred_orig_size == 2))
        total_pixels = liquid_pixels + gas_pixels

        # 气液比计算
        if liquid_pixels > 0:
            gas_liquid_ratio = gas_pixels / liquid_pixels
        else:
            gas_liquid_ratio = float('inf')

        if total_pixels > 0:
            gas_area_ratio = gas_pixels / total_pixels
        else:
            gas_area_ratio = 0.0

        # 包裹体名称
        inclusion_name = f"{img_name}_ROI{i}"

        # 图片大小（像素）
        image_size = f"{orig_w}x{orig_h}"

        # ROI大小
        roi_size = f"{roi_w}x{roi_h}"

        # === 3.5 保存结果到字典 ===
        result_dict = {
            "包裹体名称": inclusion_name,
            "所属图像": img_name,
            "置信度": round(conf, 6),
            "图像大小": image_size,
            "ROI大小": roi_size,
            "气相像素": gas_pixels,
            "液相像素": liquid_pixels,
            "总像素": total_pixels,
            "气液比(G/L)": round(gas_liquid_ratio, 6) if np.isfinite(gas_liquid_ratio) else "inf",
            "气相面积占比": round(gas_area_ratio, 6),
            "X位置": x_pos,
            "Y位置": y_pos,
            "Z位置": z_pos,
            "温度": temperature,
            "检测框坐标": f"({x1},{y1},{x2},{y2})",
        }

        per_image_results.append(result_dict)
        all_results.append(result_dict)

        # === 3.6 保存ROI掩码 ===
        if SAVE_ROI_MASK:
            mask_path = os.path.join(subdirs['roi_masks'], f"{inclusion_name}_mask.png")
            cv2.imwrite(mask_path, pred_orig_size)

        # === 3.7 保存ROI彩色可视化 ===
        if SAVE_ROI_VISUALIZATION:
            color_mask = np.zeros((roi_h, roi_w, 3), dtype=np.uint8)
            color_mask[pred_orig_size == 1] = COLOR_LIQUID
            color_mask[pred_orig_size == 2] = COLOR_GAS
            overlay = cv2.addWeighted(roi, 0.6, color_mask, 0.4, 0)
            vis_path = os.path.join(subdirs['roi_visualizations'], f"{inclusion_name}_vis.png")
            cv2.imwrite(vis_path, overlay)

        # === 3.8 保存LabelMe JSON ===
        if SAVE_LABELME_JSON:
            create_labelme_json(pred_orig_size, roi, f"{inclusion_name}.jpg", CLASS_MAP, subdirs['jsons'])

        # === 3.9 PIL绘制检测框和标签 ===
        if SAVE_BOXED_IMAGE:
            ratio_txt = f"r:{gas_liquid_ratio:.2f}" if np.isfinite(gas_liquid_ratio) else "inf"
            conf_txt = f"c:{conf:.2f}"
            label = f"{ratio_txt} | {conf_txt}"

            bbox = draw.textbbox((0, 0), label, font=font_small)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            draw.rectangle([x1, y1 - text_height - 6, x1 + text_width + 6, y1], fill='red')
            draw.text((x1 + 3, y1 - text_height - 3), label, fill='white', font=font_small)
            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)

    # === 4. 保存带框图像 ===
    if SAVE_BOXED_IMAGE and len(boxes) > 0:
        plotted_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        boxed_path = os.path.join(subdirs['images_with_boxes'], f"{img_name}.jpg")
        cv2.imwrite(boxed_path, plotted_img)

    # === 5. 保存YOLO格式标签 ===
    if SAVE_YOLO_LABELS:
        label_path = os.path.join(subdirs['labels'], f"{img_name}.txt")
        with open(label_path, 'w') as f:
            for box in boxes:
                x1b, y1b, x2b, y2b = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0].cpu().numpy())
                x_center = (x1b + x2b) / (2 * orig_w)
                y_center = (y1b + y2b) / (2 * orig_h)
                width = (x2b - x1b) / orig_w
                height = (y2b - y1b) / orig_h
                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    # === 6. 保存单张图像的CSV ===
    if SAVE_PER_IMAGE_CSV and len(per_image_results) > 0:
        roi_csv_path = os.path.join(subdirs['csvs'], f"{img_name}_results.csv")
        with open(roi_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['ROI', 'X1', 'Y1', 'X2', 'Y2', '气相像素', '液相像素', '气液比', '置信度'])
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                if i < len(per_image_results):
                    r = per_image_results[i]
                    writer.writerow([
                        i, x1, y1, x2, y2,
                        r['气相像素'], r['液相像素'], r['气液比(G/L)'],
                        f"{conf:.6f}"
                    ])

    print(f"✅ {img_name} | 检测到 {len(boxes)} 个包裹体，已分析完成")
    return per_image_results

# ========================
# 九、扫描主模块
# ========================

def scan_and_analyze():
    """
    扫描薄片全区域并分析
    流程：Z轴分层 → Y轴扫描 → X轴扫描 → 截图 → 检测分割 → 统计
    """
    global all_results
    all_results = []

    # 创建输出目录
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # 生成扫描批次标识
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = os.path.join(OUTPUT_BASE, f"scan_{batch_id}")
    os.makedirs(batch_dir, exist_ok=True)

    # 临时截图保存目录
    temp_img_dir = os.path.join(batch_dir, "temp_screenshots")
    os.makedirs(temp_img_dir, exist_ok=True)

    # 生成扫描坐标列表
    xs = list(range(X_START, X_END + X_STEP, X_STEP))
    ys = list(range(Y_START, Y_END + Y_STEP, Y_STEP))
    zs = list(range(Z_START, Z_END + Z_STEP, Z_STEP))

    total = len(xs) * len(ys) * len(zs)
    current = 0

    print("\n" + "="*60)
    print(f"🔬 开始扫描分析")
    print(f"   X范围: {X_START} ~ {X_END} (步长{X_STEP})")
    print(f"   Y范围: {Y_START} ~ {Y_END} (步长{Y_STEP})")
    print(f"   Z范围: {Z_START} ~ {Z_END} (步长{Z_STEP})")
    print(f"   总点数: {total}")
    print(f"   输出目录: {batch_dir}")
    print("="*60 + "\n")

    try:
        for z in zs:
            # 移动Z轴
            if not move_z_absolute(z):
                print(f"⚠️ Z轴移动到 {z} 失败，跳过该层")
                continue

            for y in ys:
                # 移动Y轴
                if not set_y(y):
                    print(f"⚠️ Y轴移动到 {y} 失败，跳过该行")
                    continue

                for x in xs:
                    current += 1
                    print(f"\n--- [{current}/{total}] 位置: X={x}, Y={y}, Z={z} ---")

                    # 移动X轴
                    if not set_x(x):
                        print(f"⚠️ X轴移动到 {x} 失败，跳过该点")
                        continue

                    # 等待稳定
                    time.sleep(0.8)

                    # 读取温度
                    temp = get_temp()
                    temp_str = f"{temp}°C" if temp is not None else "N/A"
                    print(f"🌡️ 温度: {temp_str}")

                    # 截图
                    img_name = f"X{x}_Y{y}_Z{z}"
                    img_path = os.path.join(temp_img_dir, f"{img_name}.png")
                    if not take_screenshot(img_path):
                        continue

                    # 分析图像
                    analyze_image(
                        img_path=img_path,
                        img_name=img_name,
                        output_base=batch_dir,
                        x_pos=x,
                        y_pos=y,
                        z_pos=z,
                        temperature=temp
                    )

        # 扫描完成，保存汇总结果
        save_summary(batch_dir, batch_id)

        # 归位
        print("\n🔄 扫描完成，开始归位...")
        move_z_absolute(Z_START)
        zero_xy()

        print("\n" + "="*60)
        print("✅ 扫描分析全部完成！")
        print(f"📂 结果保存在: {batch_dir}")
        print("="*60)

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断扫描")
        # 保存已收集的结果
        save_summary(batch_dir, batch_id)
        # 尝试归位
        try:
            move_z_absolute(Z_START)
            zero_xy()
        except:
            pass
    except Exception as e:
        print(f"\n❌ 扫描过程中发生错误: {e}")
        save_summary(batch_dir, batch_id)
        raise

# ========================
# 十、汇总保存模块
# ========================

def save_summary(batch_dir, batch_id):
    """保存汇总表格和统计信息"""
    global all_results

    if len(all_results) == 0:
        print("⚠️ 未检测到任何包裹体，无汇总数据")
        return

    # 1. 保存主汇总CSV
    summary_csv = os.path.join(batch_dir, f"all_inclusions_summary_{batch_id}.csv")
    with open(summary_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        # 表头
        headers = [
            '包裹体名称', '所属图像', '置信度', '图像大小', 'ROI大小',
            '气相像素', '液相像素', '总像素', '气液比(G/L)', '气相面积占比',
            'X位置(μm)', 'Y位置(μm)', 'Z位置(μm)', '温度(°C)', '检测框坐标'
        ]
        writer.writerow(headers)

        # 数据行
        for r in all_results:
            writer.writerow([
                r['包裹体名称'], r['所属图像'], r['置信度'], r['图像大小'], r['ROI大小'],
                r['气相像素'], r['液相像素'], r['总像素'], r['气液比(G/L)'], r['气相面积占比'],
                r['X位置'], r['Y位置'], r['Z位置'], r['温度'], r['检测框坐标']
            ])

    print(f"\n📊 汇总表格已保存: {summary_csv}")

    # 2. 保存JSON格式汇总
    summary_json = os.path.join(batch_dir, f"all_inclusions_summary_{batch_id}.json")
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump({
            "batch_id": batch_id,
            "scan_time": datetime.now().isoformat(),
            "total_inclusions": len(all_results),
            "scan_config": {
                "X": {"start": X_START, "end": X_END, "step": X_STEP},
                "Y": {"start": Y_START, "end": Y_END, "step": Y_STEP},
                "Z": {"start": Z_START, "end": Z_END, "step": Z_STEP},
            },
            "results": all_results
        }, f, indent=2, ensure_ascii=False)

    print(f"📊 JSON汇总已保存: {summary_json}")

    # 3. 打印统计信息
    print("\n" + "="*60)
    print("📈 扫描统计汇总")
    print("="*60)
    print(f"   检测到的包裹体总数: {len(all_results)}")

    confidences = [r['置信度'] for r in all_results]
    print(f"   置信度范围: {min(confidences):.4f} ~ {max(confidences):.4f}")
    print(f"   平均置信度: {sum(confidences)/len(confidences):.4f}")

    ratios = []
    for r in all_results:
        val = r['气液比(G/L)']
        if val != "inf" and isinstance(val, (int, float)):
            ratios.append(val)
    if ratios:
        print(f"   气液比范围: {min(ratios):.4f} ~ {max(ratios):.4f}")
        print(f"   平均气液比: {sum(ratios)/len(ratios):.4f}")

    area_ratios = [r['气相面积占比'] for r in all_results]
    print(f"   气相面积占比范围: {min(area_ratios):.4f} ~ {max(area_ratios):.4f}")
    print(f"   平均气相面积占比: {sum(area_ratios)/len(area_ratios):.4f}")
    print("="*60)

# ========================
# 十一、主程序
# ========================

if __name__ == "__main__":
    pyautogui.FAILSAFE = True

    print("="*60)
    print("🔬 流体包裹体自动扫描分析系统")
    print("="*60)

    # 初始化
    print("\n🔧 初始化系统...")

    if not init_z_axis():
        print("❌ Z轴初始化失败，程序退出")
        exit(1)

    if not init_temp():
        print("⚠️ 温度读取初始化失败，将继续运行（温度字段将为空）")

    # 加载模型
    print("\n🔄 加载深度学习模型...")
    try:
        load_models()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        close_z_axis()
        close_temp()
        exit(1)

    # 显示当前状态
    print(f"\n📊 当前状态:")
    print(f"   X轴: {X_POS} μm")
    print(f"   Y轴: {Y_POS} μm")
    print(f"   Z轴: {Z_POS} μm")
    print(f"\n⚠️ 安全限制:")
    print(f"   Z轴范围: [{Z_LIMIT_MIN}, {Z_LIMIT_MAX}] μm")
    print(f"   XY轴范围: [{XY_LIMIT_MIN}, {XY_LIMIT_MAX}] μm")

    # 确认开始
    print(f"\n📋 扫描配置:")
    print(f"   X: {X_START} → {X_END} (步长 {X_STEP})")
    print(f"   Y: {Y_START} → {Y_END} (步长 {Y_STEP})")
    print(f"   Z: {Z_START} → {Z_END} (步长 {Z_STEP})")
    print(f"   总扫描点数: {len(list(range(X_START, X_END+X_STEP, X_STEP))) * len(list(range(Y_START, Y_END+Y_STEP, Y_STEP))) * len(list(range(Z_START, Z_END+Z_STEP, Z_STEP)))}")
    print(f"   输出目录: {OUTPUT_BASE}")

    user_input = input("\n确认开始扫描? (y/n): ").strip().lower()
    if user_input != 'y':
        print("❌ 已取消扫描")
        close_z_axis()
        close_temp()
        exit(0)

    activate_nexus_once()
    
    # 开始扫描分析
    try:
        scan_and_analyze()
    finally:
        # 清理
        close_z_axis()
        close_temp()
        print("\n✅ 程序结束")