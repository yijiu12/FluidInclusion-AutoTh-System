from ultralytics import YOLO
import os
from datetime import datetime

# 项目根目录（向上三级）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

if __name__ == '__main__':
    # ==================== 路径配置 ====================
    WEIGHTS_PATH = os.path.join(_PROJECT_ROOT, "weights", "yolo26n_best.pt")  # 预训练权重
    DATA_YAML = os.path.join(_PROJECT_ROOT, "dataset", "yolo_detect", "dataset.yaml")  # 数据集配置
    PROJECT_DIR = os.path.join(_PROJECT_ROOT, "training_output", "yolo26n")  # 训练输出目录
    # 生成带时间戳的实验名称
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    EXPERIMENT_NAME = f'mbjc_v_end_{timestamp}'  
    # ==================== 模型加载 ====================
    print("🔄 正在加载 YOLO26n 模型...")
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"找不到权重文件: {WEIGHTS_PATH}")
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(f"找不到数据集配置: {DATA_YAML}")
    model = YOLO(WEIGHTS_PATH)    

    # ==================== 训练配置 ====================
    print("🚀 开始训练...")
    results = model.train(
        #  数据与输出
        data=DATA_YAML,
        project=PROJECT_DIR,
        name=EXPERIMENT_NAME,
        exist_ok=True,      
        #  训练轮次与批次
        epochs=120,              
        batch=8,                
        workers=4,               # 数据加载线程数    
        #  输入尺寸
        imgsz=1024,             
        device=0,                       
        #  数据增强
        degrees=30,              
        translate=0.1,           # 平移
        scale=0.3,               # 缩放
        shear=3,               # 剪切
        perspective=0.0,         # 透视变换      
        #  颜色增强
        hsv_h=0.01,             # 色调
        hsv_s=0.2,               # 饱和度
        hsv_v=0.2,               # 亮度
        #  翻转与拼接
        fliplr=0.5,              # 水平翻转
        flipud=0.2,              # 垂直翻转
        mosaic=0.5,              # Mosaic 增强
        mixup=0.1,               # Mixup 增强
        copy_paste=0,          # 关闭 CopyPaste
        #  优化器配置
        optimizer='AdamW',       
        lr0=0.001,                # 初始学习率
        lrf=0.1,                # 最终学习率比例
        momentum=0.9,            
        weight_decay=0.05,            
        warmup_epochs=3.0,     
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        #  损失函数权重
        box=8.0,                # 边界框损失
        cls=0.5,                 # 分类损失
        dfl=2.0,                 # 分布焦点损失
        #  正则化与早停
        dropout=0.0,             
        patience=50,             # 早停耐心值
        close_mosaic=10,         # 最后 10 轮关闭 Mosaic     
        #  验证与日志
        val=True,
        plots=True,              # 生成训练曲线图
        save=True,
        save_period=-1,          
        verbose=True,
        #  单类别检测优化
        single_cls=True,         # 单类别检测
        amp=True
    ) 
    # ==================== 训练完成 ====================
    print("\n" + "="*50)
    print("✅ 训练完成！")
    print("="*50)
    print(f"📁 权重保存路径：{results.save_dir}")
    print(f"📊 最佳权重：{os.path.join(results.save_dir, 'weights', 'best.pt')}")
    print(f"📈 训练曲线：{os.path.join(results.save_dir, 'results.png')}")
    print("="*50)