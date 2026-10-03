from ultralytics import YOLO
import os
import json
import time
from datetime import datetime

# 项目根目录（向上三级）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

if __name__ == '__main__':
    # ==================== 路径配置 ====================
    BEST_WEIGHTS = os.path.join(_PROJECT_ROOT, "weights", "yolo26n_best.pt")
    DATA_YAML = os.path.join(_PROJECT_ROOT, "dataset", "yolo_detect", "dataset.yaml")
    TEST_IMG_DIR = os.path.join(_PROJECT_ROOT, "dataset", "yolo_detect", "sample", "images")
    SAVE_DIR = os.path.join(_PROJECT_ROOT, "training_output", "yolo26n", "yolo_test_results")
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ==================== 直接加载 best.pt ====================
    print("🔄 加载模型中...")
    model = YOLO(BEST_WEIGHTS)
    print(f"✅ 模型加载成功：{BEST_WEIGHTS}")

    # ==================== 1. 测试集标准评估 ====================
    print("\n🚀 开始测试集评估...")
    metrics = model.val(
        data=DATA_YAML,
        split='test',
        batch=8,
        imgsz=1280,
        device=0,
        project=SAVE_DIR,
        name='v_final_end',
        exist_ok=True,
        plots=True,
        save_json=True,
        verbose=True,
    )

    # 提取指标
    map50 = metrics.box.map50
    map = metrics.box.map
    precision = metrics.box.mp
    recall = metrics.box.mr


    # ==================== 2. CPU 推理速度评估 ====================
    print("\n⏱️  计算 CPU 推理速度...")
    model_cpu = YOLO(BEST_WEIGHTS)
    test_images = [os.path.join(TEST_IMG_DIR, f) for f in os.listdir(TEST_IMG_DIR) if f.endswith(('jpg', 'png'))]
    test_images = test_images[:50]

    total_time = 0.0
    total_frames = len(test_images)

    for img_path in test_images:
        start = time.time()
        model_cpu.predict(img_path, imgsz=1280, device='cpu', verbose=False)
        total_time += time.time() - start

    avg_time_ms = (total_time / total_frames) * 1000
    fps = total_frames / total_time

    # ==================== 输出结果 ====================
    print("\n" + "="*65)
    print("📊 测试集评估结果")
    print("="*65)
    print(f"精确率 Precision:    {precision:.4f}")
    print(f"召回率 Recall:       {recall:.4f}")
    print(f"mAP@0.5:             {map50:.4f}")
    print(f"mAP@0.5:0.95:        {map:.4f}")
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
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "mAP@0.5": round(float(map50), 4),
            "mAP@0.5:0.95": round(float(map), 4)
        },
        "cpu_speed": {
            "avg_time_ms": round(avg_time_ms, 2),
            "fps": round(fps, 2)
        }
    }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    print(f"\n📄 评估报告已保存：{report_path}")