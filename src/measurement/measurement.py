"""测温业务核心：均一判定器 + 单包裹体完整测量流程编排。"""

import os
import time
from collections import deque
from typing import Optional, Tuple

import numpy as np

import config
from src.hardware import hardware
from src.utils import imaging
from src.dl_model import model_loader as models
from src.tracking_autofocus import tracking
from src.utils import io_utils


class HomogenizationDetector:
    """均一状态检测器：基于气液比连续低于阈值判定均一。"""

    def __init__(self, stability_window: int = config.STABILITY_WINDOW,
                 ratio_threshold: float = config.RATIO_THRESHOLD,
                 trend_threshold: float = config.TREND_THRESHOLD,
                 burst_threshold: float = config.BURST_THRESHOLD):
        self.stability_window = stability_window
        self.ratio_threshold = ratio_threshold
        self.trend_threshold = trend_threshold
        self.burst_threshold = burst_threshold
        self.history = deque(maxlen=100)
        self.stable_count = 0

    def add_data(self, temperature, ratio, gas_pixels, liquid_pixels, timestamp):
        self.history.append({
            'temperature': temperature,
            'ratio': ratio,
            'gas_pixels': gas_pixels,
            'liquid_pixels': liquid_pixels,
            'timestamp': timestamp
        })

    def check_homogenization(self):
        if len(self.history) < 1:
            return False, None, None, 0.0, "数据不足"

        curr = self.history[-1]
        curr_ratio = curr['ratio']
        curr_temp = curr['temperature']

        if not np.isfinite(curr_ratio) or curr_ratio >= self.ratio_threshold:
            self.stable_count = 0
            return False, None, None, 0.0, f"气液比{curr_ratio:.4f}高于阈值"

        self.stable_count += 1

        if len(self.history) >= 2:
            prev = self.history[-2]
            if abs(curr['ratio'] - prev['ratio']) > self.burst_threshold:
                self.stable_count = 0
                return False, "burst", curr_temp, 0.0, "检测到爆裂"

        if self.stable_count >= self.stability_window:
            confidence = 1.0 - (curr_ratio / self.ratio_threshold)
            return True, "to_liquid", curr_temp, max(0, min(1, confidence)), \
                   f"连续{self.stable_count}帧气液比低于阈值({curr_ratio:.4f})"

        return False, None, None, 0.0, f"稳定计数: {self.stable_count}/{self.stability_window}"


def measure_single_inclusion(
    inclusion_data: dict,
    output_dir: str,
    inc_idx: int,
    total: int,
) -> Tuple[bool, Optional[dict]]:
    """
    单包裹体完整测量流程。
    """
    name = inclusion_data.get('包裹体名称', 'unknown')
    target_x = int(float(inclusion_data.get('X位置(μm)', 0)))
    target_y = int(float(inclusion_data.get('Y位置(μm)', 0)))
    target_z = int(float(inclusion_data.get('Z位置(μm)', 0)))

    print("\n" + "="*70)
    print(f"🔬 [{inc_idx+1}/{total}] 开始测量: {name}")
    print(f"   数据库坐标: X={target_x}, Y={target_y}, Z={target_z}")
    print("="*70)

    # 创建输出子目录
    inc_dir = os.path.join(output_dir, f"{inc_idx:02d}_{name}")
    for sub in ["segmentation_visualizations", "roi_raw_images",
                "full_screenshots", "full_detection_images"]:
        os.makedirs(os.path.join(inc_dir, sub), exist_ok=True)

    vis_dir = os.path.join(inc_dir, "segmentation_visualizations")
    roi_raw_dir = os.path.join(inc_dir, "roi_raw_images")
    full_ss_dir = os.path.join(inc_dir, "full_screenshots")
    full_det_dir = os.path.join(inc_dir, "full_detection_images")

    # ========== 定位阶段 ==========

    # 1. 移动XY到数据库坐标
    print("\n📍 移动XY到数据库坐标...")
    if not hardware.set_y(target_y):
        print("❌ Y轴移动失败"); return False, None
    if not hardware.set_x(target_x):
        print("❌ X轴移动失败"); return False, None

    # 2. 3D空间搜索：全局形态匹配最佳
    print("\n🔍 启动3D空间搜索...")
    spatial_match = tracking.search_inclusion_3d(
        inclusion_data, (target_x, target_y, target_z)
    )
    if spatial_match is None:
        print("❌ 3D搜索未找到目标，跳过该包裹体")
        return False, None

    # 3. 移动到最佳XYZ
    print(f"\n📍 移动到最佳空间位置...")
    hardware.set_x(spatial_match.x_um)
    hardware.set_y(spatial_match.y_um)
    hardware.move_z_absolute(spatial_match.z_um)

    # 4. 主动回中：基于已知框直接位移到中央
    print("\n🎯 主动回中...")
    recenter_ok, box = tracking.recenter_once(spatial_match.box, inclusion_data)
    if not recenter_ok:
        print("❌ 主动回中失败，跳过该包裹体")
        return False, None

    # 5. 确认目标锁定
    print("\n🔒 确认目标锁定...")
    img = imaging.take_screenshot()
    if img is None:
        print("❌ 截图失败"); return False, None

    # 用形态匹配确认锁定
    yolo_ok, init_box, init_conf, match_score = models.detect_target_inclusion(
        img, inclusion_data, conf_thresh=config.CONF_THRESH
    )
    if not yolo_ok:
        print("❌ 回中后形态匹配失败，无法锁定目标"); return False, None

    print(f"✅ 形态匹配锁定目标: match_score={match_score:.3f}, conf={init_conf:.3f}")

    # ========== 加热测温阶段 ==========

    measurement_history = []

    # 记录初始状态
    time.sleep(1.0)
    initial_temp = hardware.get_temp()

    initial_vis = os.path.join(vis_dir, "frame_initial.png")
    initial_roi_raw = os.path.join(roi_raw_dir, "frame_initial_roi_raw.png")
    initial_full_ss = os.path.join(full_ss_dir, "frame_initial_full.png")
    initial_full_det = os.path.join(full_det_dir, "frame_initial_detection.png")

    img = imaging.take_screenshot()
    io_utils.save_full_screenshot(img, initial_full_ss)
    io_utils.save_detection_image(img, init_box, initial_full_det, "initial_target")
    initial_ratio, gas_p, liquid_p, total_p, _ = models.analyze_locked_roi(
        img, init_box, initial_vis, initial_roi_raw
    )
    print(f"   初始温度: {initial_temp}°C | 初始气液比: {initial_ratio}")

    # 设置加热参数
    print("\n🌡️ 设置加热参数...")
    hardware.set_heating_rate(config.STAGE1_RATE)
    hardware.set_target_temp(config.STAGE1_TARGET)

    stages = [
        {"rate": config.STAGE1_RATE, "target": config.STAGE1_TARGET, "name": "快速升温阶段"},
        {"rate": config.STAGE2_RATE, "target": config.TARGET_TEMP_MAX, "name": "慢速升温阶段"}
    ]
    current_stage_idx = 0

    detector = HomogenizationDetector(
        stability_window=config.STABILITY_WINDOW,
        ratio_threshold=config.RATIO_THRESHOLD,
        trend_threshold=config.TREND_THRESHOLD,
        burst_threshold=config.BURST_THRESHOLD,
    )

    # 加热循环状态
    start_time = time.time()
    last_focus_temp = initial_temp if isinstance(initial_temp, (int, float)) else 0
    last_focus_time = start_time
    last_sharpness = tracking.evaluate_sharpness(img, init_box)
    best_sharpness = last_sharpness
    frame_count = 0
    homogenized = False
    homo_result = None

    last_valid_box = init_box

    try:
        hardware.start_heating()

        while True:
            loop_start = time.time()
            frame_count += 1

            current_temp = hardware.get_temp()
            if current_temp is None:
                current_temp = "N/A"

            # 阶段切换
            if isinstance(current_temp, (int, float)) and current_stage_idx < len(stages) - 1:
                if current_temp >= stages[current_stage_idx]["target"] - 3.0:
                    current_stage_idx += 1
                    next_stage = stages[current_stage_idx]
                    print(f"\n🔄 切换至【{next_stage['name']}】")
                    hardware.set_heating_rate(next_stage["rate"])
                    hardware.set_target_temp(next_stage["target"])

            # 超温检查
            if isinstance(current_temp, (int, float)) and current_temp >= config.TARGET_TEMP_MAX:
                print(f"\n⚠️ 达到最高温度 {config.TARGET_TEMP_MAX}°C，停止加热")
                break

            # 截图
            img = imaging.take_screenshot()
            if img is None:
                time.sleep(1)
                continue

            # ========== 越界检查 ==========
            if tracking.check_boundary(last_valid_box, margin_px=config.BOUNDARY_MARGIN_PX):
                print("\n🔄 检测到包裹体靠近边界，触发回中...")
                hardware.stop_heating()

                ok, new_box = tracking.recenter_once(last_valid_box, inclusion_data)
                if ok:
                    # 重置参考系
                    last_valid_box = new_box
                    _ = imaging.take_screenshot()
                    print("✅ 回中完成，恢复加热...")
                    hardware.start_heating()
                    continue
                else:
                    print("❌ 回中失败，终止测量")
                    break

            # ========== 目标重定位 ==========
            box, source, track_ok = tracking.relocate_target(
                img, inclusion_data, last_valid_box
            )

            # 目标丢失直接终止
            if not track_ok:
                print(f"\n⚠️ 目标丢失，终止测量")
                break
            
            if box is not None:
                last_valid_box = box
            chosen_box = box

            # ========== 自动调焦触发判断 ==========
            current_sharpness = tracking.evaluate_sharpness(img, last_valid_box)
            need_focus = False

            if isinstance(current_temp, (int, float)):
                if current_temp - last_focus_temp >= config.FOCUS_INTERVAL_TEMP:
                    need_focus = True
                    last_focus_temp = current_temp

            if loop_start - last_focus_time >= config.FOCUS_INTERVAL_TIME:
                need_focus = True
                last_focus_time = loop_start

            if best_sharpness > 0 and current_sharpness < best_sharpness * config.FOCUS_SHARPNESS_DROP:
                print(f"\n⚠️ 清晰度下降，触发调焦")
                need_focus = True

            if need_focus and isinstance(current_temp, (int, float)):
                best_z, focus_box, focus_ok = tracking.auto_focus(
                    inclusion_data, last_valid_box, current_temp,
                    stages[current_stage_idx]["target"]
                )
                last_focus_time = time.time()
                last_focus_temp = current_temp

                if not focus_ok:
                    print("   ⚠️ 调焦失败，终止测量")
                    break

                # 调焦后强制YOLO扫描
                time.sleep(0.5)
                img = imaging.take_screenshot()
                if img is not None:
                    candidates = models.detect_all_inclusions(img, conf_thresh=config.CONF_THRESH)
                    nearby = tracking._filter_candidates_by_distance(
                        candidates, last_valid_box, config.TARGET_MATCH_RADIUS_PX * 2
                    )

                    if nearby:
                        new_box, conf, source = models.choose_best_candidate(img, nearby, inclusion_data)
                        if new_box is None:
                            print("   ❌ 调焦后形态匹配失败，终止测量")
                            break
                        dist = tracking.center_distance(new_box, last_valid_box)
                        print(f"   调焦后重新锁定: 偏移={dist:.1f}px, 来源={source}")
                        last_valid_box = new_box
                    else:
                        print("   ❌ 调焦后未检测到目标，终止测量")
                        break
                else:
                    print("   ⚠️ 调焦后截图失败，终止测量")
                    break

                # 更新清晰度
                current_sharpness = tracking.evaluate_sharpness(img, last_valid_box)

            if current_sharpness > best_sharpness:
                best_sharpness = current_sharpness

            # ========== ROI分析与均一判定 ==========
            temp_str = f"{current_temp}" if isinstance(current_temp, (int, float)) else "NA"
            frame_name = f"frame_{frame_count:04d}_T{temp_str}"

            full_ss_path = os.path.join(full_ss_dir, f"{frame_name}_full.png")
            io_utils.save_full_screenshot(img, full_ss_path)
            full_det_path = os.path.join(full_det_dir, f"{frame_name}_detection.png")
            io_utils.save_detection_image(img, last_valid_box, full_det_path, source)

            vis_path = os.path.join(vis_dir, f"{frame_name}.png")
            roi_raw_path = os.path.join(roi_raw_dir, f"{frame_name}_roi_raw.png")

            ratio, gas_pixels, liquid_pixels, total_pixels, success = models.analyze_locked_roi(
                img, last_valid_box, vis_path, roi_raw_path
            )

            if success:
                timestamp = time.time() - start_time
                data_point = {
                    'time': round(timestamp, 2),
                    'temperature': current_temp,
                    'ratio': ratio if ratio is not None else 0,
                    'gas_pixels': gas_pixels,
                    'liquid_pixels': liquid_pixels,
                    'total_pixels': total_pixels,
                    'vis_path': vis_path,
                    'sharpness': round(current_sharpness, 2),
                    'z_pos': hardware.get_position()[2],
                }
                measurement_history.append(data_point)
                detector.add_data(current_temp, ratio if ratio is not None else 0,
                                 gas_pixels, liquid_pixels, timestamp)

                ratio_str = f"{ratio:.4f}" if ratio is not None and np.isfinite(ratio) else "N/A"
                stage_tag = "S1" if current_stage_idx == 0 else "S2"
                print(f"  [{frame_count:04d}|{stage_tag}] T={current_temp}°C | G/L={ratio_str} | "
                      f"G={gas_pixels} L={liquid_pixels} | 清晰度={current_sharpness:.1f} | "
                      f"定位={source}")

                # 检查均一
                is_homo, homo_type, homo_temp, confidence, reason = detector.check_homogenization()
                if is_homo:
                    print(f"\n" + "="*60)
                    print(f"🎉 检测到均一！")
                    print(f"   均一类型: {homo_type}")
                    print(f"   均一温度: {homo_temp}°C")
                    print(f"   置信度: {confidence:.2%}")
                    print(f"   判定依据: {reason}")
                    print("="*60)
                    homogenized = True
                    homo_result = {
                        'type': homo_type,
                        'temperature': homo_temp,
                        'confidence': confidence,
                        'reason': reason,
                        'time': timestamp
                    }
                    homo_vis_path = os.path.join(vis_dir, f"{frame_name}_HOMOGENIZATION.png")
                    os.rename(vis_path, homo_vis_path)
                    measurement_history[-1]['vis_path'] = homo_vis_path
                    break

            # 控制采样间隔
            elapsed = time.time() - loop_start
            if elapsed < config.SAMPLE_INTERVAL:
                time.sleep(config.SAMPLE_INTERVAL - elapsed)

        hardware.stop_heating()

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
        hardware.stop_heating()

    # ========== 保存结果 ==========
    result = io_utils.save_single_results(
        inc_dir, name, inclusion_data, homo_result,
        initial_temp, initial_ratio, measurement_history,
        spatial_match.z_um, spatial_match.match_score
    )

    return True, result