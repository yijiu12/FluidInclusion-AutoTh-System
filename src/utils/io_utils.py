"""数据持久化：CSV、JSON、图像、绘图、批次报告生成。"""
import csv
import json
import os
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config


def load_inclusion_list(csv_path: str):
    """加载包裹体数据库CSV。支持传入目录自动补全路径。"""
    inclusions = []
    
    if os.path.isdir(csv_path):
        csv_path = os.path.join(csv_path, "morphology_statistics", "inclusions_with_morphology.csv")
    
    if not os.path.exists(csv_path):
        return None
        
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            inclusions.append(row)
    return inclusions


def save_single_results(output_dir, name, inclusion_data, homo_result,
                        initial_temp, initial_ratio, measurement_history,
                        best_z, best_combined):
    """保存单个包裹体的所有结果（CSV曲线、JSON报告、曲线图）。"""
    
    # CSV曲线
    csv_path = os.path.join(output_dir, "temperature_ratio_curve.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['时间(s)', '温度(°C)', '气液比', '气相像素', '液相像素', '总像素', '清晰度', 'Z轴位置', '可视化图'])
        for dp in measurement_history:
            time_val = dp['time']
            vis_path = dp.get('vis_path', '')
            if vis_path:
                rel_path = os.path.relpath(vis_path, os.path.dirname(csv_path)).replace("\\", "/")
                hyperlink = f'=HYPERLINK("{rel_path}","{time_val:.2f}s")'
            else:
                hyperlink = f"{time_val:.2f}"
            writer.writerow([
                hyperlink, dp['temperature'], dp['ratio'],
                dp['gas_pixels'], dp['liquid_pixels'], dp['total_pixels'],
                dp.get('sharpness', ''), dp.get('z_pos', ''), vis_path
            ])
    print(f"\n📊 曲线数据: {csv_path}")
    
    # 绘图
    plot_single_curve(output_dir, measurement_history)
    
    # JSON报告
    report = {
        "package_name": name,
        "measurement_time": datetime.now().isoformat(),
        "inclusion_info": inclusion_data,
        "heating_parameters": {
            "stage1": {"rate": config.STAGE1_RATE, "target": config.STAGE1_TARGET},
            "stage2": {"rate": config.STAGE2_RATE, "target": config.TARGET_TEMP_MAX},
            "sample_interval": config.SAMPLE_INTERVAL
        },
        "initial_state": {"temperature": initial_temp, "ratio": initial_ratio},
        "localization": {
            "best_z": best_z,
            "best_combined_score": best_combined
        },
        "result": homo_result,
        "total_data_points": len(measurement_history),
        "data": measurement_history
    }
    json_path = os.path.join(output_dir, "report.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"📋 报告: {json_path}")
    
    print("\n" + "="*60)
    print("📈 测量结果")
    print("="*60)
    print(f"   包裹体: {name}")
    print(f"   初始温度: {initial_temp}°C")
    print(f"   数据点数: {len(measurement_history)}")
    if homo_result:
        print(f"   均一状态: ✅ 已均一 @ {homo_result['temperature']}°C")
    else:
        print(f"   均一状态: ❌ 未检测到均一")
    print("="*60)
    
    return report


def plot_single_curve(output_dir, measurement_history):
    """绘制单包裹体测温曲线（温度-时间、气液比-时间、气液比-温度、相像素-时间）。"""
    if len(measurement_history) < 2:
        return
    valid_data = [dp for dp in measurement_history
                  if isinstance(dp['temperature'], (int, float)) and np.isfinite(dp['ratio'])]
    if len(valid_data) < 2:
        return

    times = [dp['time'] for dp in valid_data]
    temps = [dp['temperature'] for dp in valid_data]
    ratios = [dp['ratio'] for dp in valid_data]
    gas_pixels = [dp['gas_pixels'] for dp in valid_data]
    liquid_pixels = [dp['liquid_pixels'] for dp in valid_data]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax1 = axes[0, 0]
    ax1.plot(times, temps, 'r-', lw=1.5, label='Temperature')
    switch_idx = None
    for i, t in enumerate(temps):
        if t >= config.STAGE1_TARGET - 5:
            switch_idx = i
            break
    if switch_idx and switch_idx > 0:
        ax1.axvline(x=times[switch_idx], color='orange', ls='--', alpha=0.7, label='Stage Switch')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_xlabel('Time (s)')
    ax1.set_title('Temperature vs Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.plot(times, ratios, 'b-', lw=1.5, label='Gas/Liquid Ratio')
    ax2.axhline(y=config.RATIO_THRESHOLD, color='g', ls='--', alpha=0.5, label=f'Threshold ({config.RATIO_THRESHOLD})')
    ax2.set_ylabel('Gas/Liquid Ratio')
    ax2.set_xlabel('Time (s)')
    ax2.set_title('Gas/Liquid Ratio vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    ax3.plot(temps, ratios, 'm-', lw=2, marker='o', markersize=3, label='G/L Ratio')
    ax3.axhline(y=config.RATIO_THRESHOLD, color='g', ls='--', alpha=0.5, label=f'Threshold ({config.RATIO_THRESHOLD})')
    ax3.axvline(x=config.STAGE1_TARGET, color='orange', ls='--', alpha=0.5, label='Stage 1→2')
    ax3.set_ylabel('Gas/Liquid Ratio')
    ax3.set_xlabel('Temperature (°C)')
    ax3.set_title('Gas/Liquid Ratio vs Temperature (Core Chart)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    ax4.plot(times, gas_pixels, 'r-', lw=1, label='Gas pixels')
    ax4.plot(times, liquid_pixels, 'b-', lw=1, label='Liquid pixels')
    ax4.set_ylabel('Pixel Count')
    ax4.set_xlabel('Time (s)')
    ax4.set_title('Phase Pixels vs Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "curve_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 曲线图: {plot_path}")


def save_full_screenshot(img_array: np.ndarray, save_path: str) -> bool:
    """保存全幅截图。"""
    try:
        pil_img = Image.fromarray(img_array)
        pil_img.save(save_path, quality=95)
        return True
    except Exception as e:
        print(f"⚠️ 保存截图失败: {e}")
        return False


def save_detection_image(img_array: np.ndarray, box: tuple | None, save_path: str, label: str = "inclusion") -> bool:
    """保存带检测框标注的图像。"""
    try:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        if box is not None:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_bgr, label, (x1, max(y1 - 10, 0)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        pil_img.save(save_path, quality=95)
        return True
    except Exception as e:
        print(f"⚠️ 保存检测图失败: {e}")
        return False


def generate_batch_report(batch_dir: str, all_results: list):
    """生成批次汇总报告（JSON、CSV）和均一温度频率直方图。"""
    successful = [r for s, r in all_results if s and r and r.get('result')]
    failed = [i for i, (s, r) in enumerate(all_results) if not s or not r or not r.get('result')]
    
    temps = [r['result']['temperature'] for r in successful]
    
    summary = {
        "batch_time": datetime.now().isoformat(),
        "total_requested": len(all_results),
        "total_success": len(successful),
        "total_failed": len(failed),
        "cooling_target": config.COOLING_TARGET_TEMP,
        "cooling_resume_threshold": config.COOLING_RESUME_THRESHOLD,
        "histogram_bin_width": config.HISTOGRAM_BIN_WIDTH,
        "successful_inclusions": [
            {
                "name": r['package_name'],
                "homogenization_temp": r['result']['temperature'],
                "confidence": r['result']['confidence'],
                "data_points": r['total_data_points']
            }
            for r in successful
        ],
        "failed_indices": failed,
        "statistics": {}
    }
    
    if temps:
        summary['statistics'] = {
            "mean": round(float(np.mean(temps)), 2),
            "std": round(float(np.std(temps)), 2),
            "median": round(float(np.median(temps)), 2),
            "min": round(float(np.min(temps)), 2),
            "max": round(float(np.max(temps)), 2),
            "count": len(temps)
        }
    
    json_path = os.path.join(batch_dir, "batch_report.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n📋 批次报告: {json_path}")
    
    # 汇总CSV
    csv_path = os.path.join(batch_dir, "batch_summary.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['序号', '包裹体名称', '均一温度(°C)', '置信度', '数据点数', '状态'])
        for i, (s, r) in enumerate(all_results):
            if s and r and r.get('result'):
                writer.writerow([
                    i+1, r['package_name'], r['result']['temperature'],
                    f"{r['result']['confidence']:.2%}", r['total_data_points'], '成功'
                ])
            else:
                name = r.get('package_name', 'unknown') if r else 'unknown'
                writer.writerow([i+1, name, 'N/A', 'N/A', 'N/A', '失败'])
    print(f"📊 汇总表格: {csv_path}")
    
    # 直方图
    if len(temps) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        min_temp = np.min(temps)
        max_temp = np.max(temps)
        bin_start = np.floor(min_temp / config.HISTOGRAM_BIN_WIDTH) * config.HISTOGRAM_BIN_WIDTH
        bin_end = np.ceil(max_temp / config.HISTOGRAM_BIN_WIDTH) * config.HISTOGRAM_BIN_WIDTH + config.HISTOGRAM_BIN_WIDTH
        bins = np.arange(bin_start, bin_end + config.HISTOGRAM_BIN_WIDTH, config.HISTOGRAM_BIN_WIDTH)
        
        n, bins_edges, patches = ax.hist(temps, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
        
        mean_t = np.mean(temps)
        std_t = np.std(temps)
        median_t = np.median(temps)
        
        ax.axvline(mean_t, color='red', linestyle='--', linewidth=2, label=f'Mean={mean_t:.1f}°C')
        ax.axvline(median_t, color='green', linestyle='--', linewidth=2, label=f'Median={median_t:.1f}°C')
        
        ax.set_xlabel('Homogenization Temperature (°C)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Homogenization Temperature Distribution (n={len(temps)}, σ={std_t:.1f}°C)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        for i, (count, left_edge) in enumerate(zip(n, bins_edges[:-1])):
            if count > 0:
                ax.text(left_edge + config.HISTOGRAM_BIN_WIDTH/2, count + 0.1, str(int(count)),
                       ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        hist_path = os.path.join(batch_dir, "homogenization_histogram.png")
        plt.savefig(hist_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 直方图: {hist_path}")
    
    print("\n" + "="*70)
    print("📈 批次测量汇总")
    print("="*70)
    print(f"   总请求数: {len(all_results)}")
    print(f"   成功: {len(successful)} | 失败: {len(failed)}")
    if temps:
        print(f"   均一温度统计:")
        print(f"      均值: {np.mean(temps):.2f}°C")
        print(f"      标准差: {np.std(temps):.2f}°C")
        print(f"      中位数: {np.median(temps):.2f}°C")
        print(f"      范围: [{np.min(temps):.2f}, {np.max(temps):.2f}]°C")
    print("="*70)