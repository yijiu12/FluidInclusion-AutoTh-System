"""配置中心：所有运行期常量、路径、阈值、坐标、扫描网格参数。"""
import os
from typing import List, Tuple

# 项目根目录（基于 config.py 所在位置）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- 文件路径（相对项目根目录）---
Z_AXIS_DLL = os.path.join(PROJECT_ROOT, "src", "hardware", "lib", "DTStageDriver.dll")
LINKAM_DLL = os.path.join(PROJECT_ROOT, "src", "hardware", "lib", "LinkamIpc.dll")
YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, "weights", "yolo26n_best.pt")
UNET_MODEL_PATH = os.path.join(PROJECT_ROOT, "weights", "unetpp_res34_best.pth")
DEFAULT_CSV_PATH = os.path.join(PROJECT_ROOT, "experiments", "global_scan", "inclusions_with_morphology.csv")
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "output")

# --- 串口配置 ---
COM_PORT = 3
Z_SPEED = 300
XY_MOVE_SPEED = 500

# --- 界面坐标（1920x1080，有冷台时）---
NEXUS_POS = (1206, 1057)
SCREEN_REGION = (348, 151, 786, 781)
TEMP_RATE_POS = (1543, 190)
TEMP_LIMIT_POS = (1672, 190)
TEMP_START_POS = (1843, 292)
TEMP_STOP_POS = (1886, 292)
X_INPUT_POS = (1584, 520)
Y_INPUT_POS = (1585, 550)
ZERO_BTN_POS = (1594, 434)

# --- 轴安全范围 ---
Z_LIMIT_MIN = -10000
Z_LIMIT_MAX = 0
XY_LIMIT_MIN = -7500
XY_LIMIT_MAX = 7500

# --- 像素-微米转换标定 ---
STEP_X_UM_PER_SCREEN = 312.4
STEP_Y_UM_PER_SCREEN = 304.4
DIR_X = +1
DIR_Y = +1
UM_PER_PIXEL = 0.397

IMG_CENTER_X = SCREEN_REGION[2] / 2.0
IMG_CENTER_Y = SCREEN_REGION[3] / 2.0
SCALE_X = STEP_X_UM_PER_SCREEN / SCREEN_REGION[3]
SCALE_Y = STEP_Y_UM_PER_SCREEN / SCREEN_REGION[2]

# --- 批量测量核心参数 ---
BATCH_START_RANK = 9
BATCH_END_RANK = 9

# ⭐ 3D搜索网格配置（新增：中心 + 四角）
XY_SCAN_OFFSETS_UM: List[Tuple[int, int]] = [
    (0, 0),           # 中心
    (-150, -150),     # 左上
    (+150, -150),     # 右上
    (-150, +150),     # 左下
    (+150, +150),     # 右下
]
Z_SCAN_RANGE = 160
Z_SCAN_STEP = 40

# --- 降温参数 ---
COOLING_TARGET_TEMP = 30.0
COOLING_RESUME_THRESHOLD = 32.0
COOLING_CHECK_INTERVAL = 5.0
COOLING_TIMEOUT = 1000

# --- 直方图 ---
HISTOGRAM_BIN_WIDTH = 10.0

# --- 升温参数（两阶段）---
STAGE1_RATE = 40.0
STAGE1_TARGET = 180.0
STAGE2_RATE = 10.0
TARGET_TEMP_MAX = 250
SAMPLE_INTERVAL = 4.0

# --- 均一判定参数 ---
STABILITY_WINDOW = 3
RATIO_THRESHOLD = 0.02
TREND_THRESHOLD = 0.03
BURST_THRESHOLD = 0.3

# --- 目标定位与回中参数 ---
TARGET_MATCH_RADIUS_PX = 30    # 目标重定位时选取候选的邻域半径（像素）
BOUNDARY_MARGIN_PX = 80        # 触发回中的图像边界裕量（像素）
RECOOL_TIME = 60.0
RECENTER_THRESHOLD_PX = 25
RECENTER_MAX_RETRIES = 1
RECENTER_MIN_MOVE_UM = 5

# --- 自动调焦参数 ---
FOCUS_INTERVAL_TEMP = 30.0
FOCUS_INTERVAL_TIME = 60.0
FOCUS_RANGE = 200
FOCUS_STEP = 40
FOCUS_SHARPNESS_DROP = 0.4
FOCUS_BETA = 1

# --- 模型配置 ---
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRESH = 0.5
UNET_ENCODER = "resnet34"
NUM_CLASSES = 3
CLASS_MAP = {1: "liquid", 2: "gas"}
UNET_INPUT_SIZE = 256

# --- 颜色配置 (BGR) ---
COLOR_GAS = (0, 0, 255)
COLOR_LIQUID = (255, 0, 0)

MORPH_MATCH_THRESHOLD = 0.0  # 形态匹配分阈值，0表示不限制