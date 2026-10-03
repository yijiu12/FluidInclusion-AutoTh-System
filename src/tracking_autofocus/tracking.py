"""目标定位与自动调焦：基于检测的重新定位 + 清晰度评估。

说明
----
本模块提供加热过程中的目标保持与自动调焦功能，全部建立在公开的
YOLO 检测与形态匹配（见论文 Section 2.3.1 / 3.2）之上：

* 目标定位：对每一采样帧执行全图 YOLO 检测，用与上一有效框的距离约束
  选出候选，再以形态匹配分数锁定正确的包裹体；
* 空间搜索：3D（XY + Z）扫描，全局取形态匹配分数最高者；
* 主动回中：根据目标中心与图像中心的像素偏移移动 XY 平台；
* 自动调焦：Z 轴扫描，按“清晰度 × 置信度”联合评分选择最佳焦平面。
"""

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

import config
from src.hardware import hardware
from src.utils import imaging
from src.dl_model import model_loader as models


@dataclass
class SpatialMatch:
    x_um: int
    y_um: int
    z_um: int
    box: Tuple[float, float, float, float]
    match_score: float
    conf: float
    img: np.ndarray


def center_distance(box1: tuple, box2: tuple) -> float:
    if box1 is None or box2 is None:
        return float('inf')
    c1x = (box1[0] + box1[2]) / 2.0
    c1y = (box1[1] + box1[3]) / 2.0
    c2x = (box2[0] + box2[2]) / 2.0
    c2y = (box2[1] + box2[3]) / 2.0
    return np.hypot(c1x - c2x, c1y - c2y)


def evaluate_sharpness(img_array: np.ndarray, roi: tuple) -> float:
    x1, y1, x2, y2 = map(int, roi)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_array.shape[1], x2), min(img_array.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi_img = img_array[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi_img, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def _filter_candidates_by_distance(candidates: list, reference_box: tuple, threshold_px: int):
    """按距离过滤候选，返回范围内的候选列表。"""
    if reference_box is None or not candidates:
        return []
    return [c for c in candidates if center_distance(c['box'], reference_box) < threshold_px]


def search_inclusion_3d(
    inclusion_data: dict,
    center_xyz: Tuple[int, int, int],
    xy_offsets: Optional[list[Tuple[int, int]]] = None,
    z_range: int = config.Z_SCAN_RANGE,
    z_step: int = config.Z_SCAN_STEP,
) -> Optional[SpatialMatch]:
    """3D空间搜索：在每个XY位置扫描Z层，全局找形态匹配分数最高的包裹体。
    无距离阈值限制，纯形态匹配。"""
    if xy_offsets is None:
        xy_offsets = config.XY_SCAN_OFFSETS_UM

    cx, cy, cz = center_xyz
    print(f"\n🔍 启动3D空间搜索: 中心=({cx},{cy},{cz}) | XY偏移={len(xy_offsets)}个 | Z±{z_range}μm")

    global_best: Optional[SpatialMatch] = None

    for dx, dy in xy_offsets:
        test_x = cx + dx
        test_y = cy + dy

        if not (config.XY_LIMIT_MIN <= test_x <= config.XY_LIMIT_MAX and
                config.XY_LIMIT_MIN <= test_y <= config.XY_LIMIT_MAX):
            continue

        print(f"\n   [XY偏移 ({dx:+d}, {dy:+d})] 移动到 ({test_x}, {test_y})...")
        if not hardware.set_y(test_y):
            continue
        if not hardware.set_x(test_x):
            continue
        time.sleep(0.5)

        half_steps = int(z_range / z_step)
        z_layers = []
        for i in range(-half_steps, half_steps + 1):
            z = cz + i * z_step
            if config.Z_LIMIT_MIN <= z <= config.Z_LIMIT_MAX:
                z_layers.append(z)
        if cz not in z_layers:
            z_layers.append(cz)
        z_layers = sorted(set(z_layers))

        for z in z_layers:
            if not hardware.move_z_absolute(z):
                continue
            time.sleep(1.0)

            _ = hardware.get_temp()

            img = imaging.take_screenshot()
            if img is None:
                continue

            ok, box, conf, match_score = models.detect_target_inclusion(
                img, inclusion_data, conf_thresh=config.CONF_THRESH
            )

            if ok:
                _, cand_ar, cand_circ, cand_diam = models.calculate_morphology_for_box(img, box)
                print(f"      Z={z}μm | 匹配分={match_score:.4f}, 长宽比={cand_ar:.3f}, "
                      f"圆度={cand_circ:.3f}, 直径={cand_diam:.2f}μm")

                if global_best is None or match_score > global_best.match_score:
                    global_best = SpatialMatch(
                        x_um=test_x, y_um=test_y, z_um=z,
                        box=box, match_score=match_score, conf=conf, img=img
                    )
                    print(f"   🆕 当前全局最佳: 匹配分={match_score:.4f}")

    if global_best is None:
        print("❌ 3D搜索未找到任何有效目标")
        return None

    print(f"\n🏆 3D搜索完成: 最佳位置=({global_best.x_um},{global_best.y_um},{global_best.z_um}), "
          f"匹配分={global_best.match_score:.4f}")
    return global_best


def recenter_once(
    current_box: tuple,
    inclusion_data: dict
) -> Tuple[bool, Optional[Tuple[float, float, float, float]]]:
    """
    单次回中：基于当前已知框直接计算到图像中心的偏移，移动XY轴。
    移动后截图，全局形态匹配验证目标仍在视野内且位于中央附近。
    返回: (success, new_box)
    """
    x1, y1, x2, y2 = current_box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    dx_px = config.IMG_CENTER_X - cx
    dy_px = config.IMG_CENTER_Y - cy

    # 像素 → 微米
    dx_um = dy_px * config.SCALE_Y * config.DIR_Y
    dy_um = dx_px * config.SCALE_X * config.DIR_X

    x, y, z = hardware.get_position()
    new_x = x + dx_um
    new_y = y + dy_um

    # 边界安全检查
    if not (config.XY_LIMIT_MIN <= new_x <= config.XY_LIMIT_MAX and
            config.XY_LIMIT_MIN <= new_y <= config.XY_LIMIT_MAX):
        print(f"⚠️ 回中位移超范围: new_x={new_x:.1f}, new_y={new_y:.1f}")
        return False, None

    # 执行移动
    moved = False
    if abs(dx_um) > config.RECENTER_MIN_MOVE_UM:
        if hardware.set_x(int(new_x)):
            moved = True
    time.sleep(0.5)
    if abs(dy_um) > config.RECENTER_MIN_MOVE_UM:
        if hardware.set_y(int(new_y)):
            moved = True

    if not moved:
        print("⚠️ 回中位移过小，无需移动")
        return True, current_box

    print(f"   回中位移: dx={dx_um:.1f}μm, dy={dy_um:.1f}μm")
    time.sleep(1.5)

    # 验证：截图 + 全局形态匹配
    img = imaging.take_screenshot()
    if img is None:
        print("❌ 回中后截图失败")
        return False, None

    ok, box, conf, match_score = models.detect_target_inclusion(img, inclusion_data)
    if not ok:
        print("❌ 回中后全局形态匹配失败，目标丢失")
        return False, None

    # 验证：回中后的框应在图像中央附近
    bx1, by1, bx2, by2 = box
    bcx = (bx1 + bx2) / 2.0
    bcy = (by1 + by2) / 2.0
    center_dist = np.hypot(bcx - config.IMG_CENTER_X, bcy - config.IMG_CENTER_Y)

    if center_dist > config.RECENTER_THRESHOLD_PX * 2:
        print(f"⚠️ 回中后框偏离中心 {center_dist:.1f}px，可能锁定错误目标")
        return False, None

    print(f"✅ 回中成功: match_score={match_score:.3f}, 中心偏差={center_dist:.1f}px")
    return True, box


def check_boundary(
    box: Optional[Tuple[float, ...]],
    margin_px: int = config.BOUNDARY_MARGIN_PX,
) -> bool:
    """检查包裹体是否靠近图像边界。True=需要回中。"""
    if box is None:
        return False

    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    img_w = config.SCREEN_REGION[2]
    img_h = config.SCREEN_REGION[3]

    return (cx < margin_px or cx > img_w - margin_px or
            cy < margin_px or cy > img_h - margin_px)


def relocate_target(
    img: np.ndarray,
    inclusion_data: dict,
    last_valid_box: Optional[Tuple[float, float, float, float]],
) -> Tuple[Optional[Tuple[float, ...]], str, bool]:
    """
    单帧目标重定位。

    对当前帧执行全图 YOLO 检测，选取与上一有效框距离在阈值内的候选，
    再以形态匹配分数锁定最可能的目标。

    Returns:
        (当前框, 来源标签, 是否成功)
    """
    # 1. 获取所有YOLO候选
    candidates = models.detect_all_inclusions(img, conf_thresh=config.CONF_THRESH)

    # 2. 距离过滤
    nearby = _filter_candidates_by_distance(candidates, last_valid_box, config.TARGET_MATCH_RADIUS_PX)

    if len(nearby) == 0:
        return last_valid_box, "last_valid", False

    if len(nearby) == 1:
        return nearby[0]['box'], "YOLO", True

    # 多个候选：形态匹配选最佳
    chosen_box, conf, source = models.choose_best_candidate(img, nearby, inclusion_data)
    if chosen_box is None:
        print("   ⚠️ 多候选形态匹配失败")
        return last_valid_box, "YOLO_MORPH_FAIL", False
    return chosen_box, "YOLO", True


def auto_focus(
    inclusion_data: dict,
    last_valid_box: Tuple[float, ...],
    current_temp: float,
    stage_target: float,
    z_range: int = config.FOCUS_RANGE,
    z_step: int = config.FOCUS_STEP,
) -> Tuple[int, Optional[Tuple[float, float, float, float]], bool]:
    """
    自动调焦：Z轴往复扫描，每层用位置过滤+形态匹配锁定包裹体，
    综合评分(清晰度 × (1+β×yolo_conf))选最佳Z。

    基准帧或扫描层形态匹配失败时，直接返回失败。

    返回: (best_z, new_box_if_detected, success)
    """
    x, y, current_z = hardware.get_position()
    print(f"\n🔍 触发自动调焦（当前T={current_temp:.1f}°C, Z={current_z}）")

    hardware.set_target_temp(current_temp)
    time.sleep(2.0)

    # ========== 基准帧 ==========
    img = imaging.take_screenshot()
    if img is None:
        hardware.set_target_temp(stage_target)
        return current_z, None, False

    # 基准帧：尝试YOLO检测，检测不到则沿用上一有效框
    candidates = models.detect_all_inclusions(img, conf_thresh=config.CONF_THRESH)
    nearby = _filter_candidates_by_distance(candidates, last_valid_box, config.TARGET_MATCH_RADIUS_PX * 2)

    if nearby:
        base_box, base_conf, base_source = models.choose_best_candidate(img, nearby, inclusion_data)
        if base_box is None:
            print(f"   ⚠️ 基准帧形态匹配失败")
            hardware.set_target_temp(stage_target)
            return current_z, None, False
        base_yolo_conf = base_conf
    else:
        # YOLO检测不到，用上一有效框作为基准
        base_box = last_valid_box
        base_yolo_conf = 0.0
        print(f"   基准帧YOLO未检测到，使用上一有效框作为基准")

    best_sharp = evaluate_sharpness(img, base_box)
    best_combined = best_sharp * (1 + config.FOCUS_BETA * base_yolo_conf)
    best_z = current_z
    best_box = base_box
    best_yolo_conf = base_yolo_conf

    print(f"   基准: Z={current_z}, 清晰度={best_sharp:.2f}, "
          f"YOLO_conf={base_yolo_conf:.3f}, 联合评分={best_combined:.2f}")

    # 构建Z扫描候选层
    candidates_z = []
    for delta in range(-z_range, z_range + 1, z_step):
        z_test = current_z + delta
        if config.Z_LIMIT_MIN <= z_test <= config.Z_LIMIT_MAX and z_test != current_z:
            candidates_z.append(z_test)

    # ========== 扫描各Z层 ==========
    for z_test in candidates_z:
        if not hardware.move_z_absolute(z_test):
            continue
        time.sleep(1.2)

        img = imaging.take_screenshot()
        if img is None:
            continue

        # 每层：获取所有候选，用last_valid_box过滤（放宽半径）
        candidates = models.detect_all_inclusions(img, conf_thresh=config.CONF_THRESH)
        nearby = _filter_candidates_by_distance(candidates, last_valid_box, config.TARGET_MATCH_RADIUS_PX * 2)

        if not nearby:
            print(f"   Z={z_test}: 范围内无候选，跳过")
            continue

        # 形态匹配选最佳
        box, conf, source = models.choose_best_candidate(img, nearby, inclusion_data)
        if box is None:
            print(f"   Z={z_test}: 形态匹配失败，跳过")
            continue

        # 固定ROI评估清晰度（用该层锁定的框）
        sharp = evaluate_sharpness(img, box)
        combined = sharp * (1 + config.FOCUS_BETA * conf)

        print(f"   Z={z_test}, 清晰度={sharp:.2f}, YOLO_conf={conf:.3f}, "
              f"来源={source}, 联合评分={combined:.2f}")

        if combined > best_combined:
            best_combined = combined
            best_sharp = sharp
            best_z = z_test
            best_box = box
            best_yolo_conf = conf

    # 移动到最佳Z位置
    if best_z != current_z:
        hardware.move_z_absolute(best_z)
        print(f"✅ 调焦完成: Z={best_z}, 清晰度={best_sharp:.2f}, YOLO_conf={best_yolo_conf:.3f}")
    else:
        print(f"✅ 当前位置已最清晰")

    # 扫描结束，强制回到 best_z
    hardware.move_z_absolute(best_z)

    hardware.set_target_temp(stage_target)
    time.sleep(1.0)
    return best_z, best_box, True
