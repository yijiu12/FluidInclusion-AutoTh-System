"""
🎯 语义分割训练脚本 - U-Net++ (基于 segmentation_models_pytorch)
📌 特点：配置清晰 / 逻辑线性 / 功能完整 / 注释充分
🔧 适用于：流体包裹体气液相分割任务
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import logging
from sklearn.metrics import confusion_matrix

# 项目根目录（向上三级）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ========================
# 🎯 配置区域（所有参数集中管理）
# ========================
CONFIG = {
    # ========== 路径配置 ==========
    "train_img_dir": os.path.join(_PROJECT_ROOT, "dataset", "unet_segment", "train", "images"),
    "train_mask_dir": os.path.join(_PROJECT_ROOT, "dataset", "unet_segment", "train", "masks"),
    "val_img_dir": os.path.join(_PROJECT_ROOT, "dataset", "unet_segment", "val", "images"),
    "val_mask_dir": os.path.join(_PROJECT_ROOT, "dataset", "unet_segment", "val", "masks"),
    "save_dir": os.path.join(_PROJECT_ROOT, "training_output", "unetpp_res34", "unet_train_results"),
    
    # ========== 模型配置 ==========
    "model_name": "unetplusplus",      # 可选: unet, unetplusplus, deeplabv3plus
    "encoder": "resnet34",             # 骨干网络: resnet34/efficientnet-b0/mobileone_s0
    "encoder_weights": "imagenet",     # 预训练权重: imagenet/None
    "in_channels": 3,                  # 输入图像通道数
    "num_classes": 3,                  # 类别数: 0=背景, 1=液相, 2=气相
    
    # ========== 训练超参 ==========
    "epochs": 100,                     # 总训练轮数
    "batch_size": 8,                   # 每批样本数
    "lr": 1e-4,                        # 初始学习率
    "img_size": (256, 256),            # 输入图像尺寸 (宽, 高)
    
    # ========== 早停配置 ==========
    "early_stop_patience": 15,         # 验证 mIoU 多少轮不提升则停止训练
    "early_stop_metric": "miou",       # 早停监控指标: miou / val_loss
    
    # ========== 其他配置 ==========
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # 自动选择设备
    "num_workers": 4,                  # 数据加载线程数 
    "save_visual_every": 10,           # 每多少轮保存一次预测可视化图
}

# 创建保存目录
os.makedirs(CONFIG["save_dir"], exist_ok=True)

# 配置日志：同时输出到控制台和文件
log_path = os.path.join(CONFIG["save_dir"], "train.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ========================
# 🗃️ 自定义数据集类
# ========================
class FluidDataset(Dataset):
    """
    流体包裹体分割数据集
    - 图像和 Mask 文件名必须一致（如 001.png）

    """
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # 预扫描所有 .png 文件，避免每次 __getitem__ 都遍历目录
        self.file_list = [f for f in os.listdir(img_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        logger.info(f"📁 加载数据集: {img_dir} | 样本数: {len(self.file_list)}")
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        
        # 读取图像 (BGR→RGB)
        img_path = os.path.join(self.img_dir, filename)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 读取 Mask (灰度模式，像素值代表类别)
        mask_path = os.path.join(self.mask_dir, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 数据增强（训练时）/ 仅归一化（验证时）
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']      # Tensor [C, H, W]
            mask = augmented['mask']        # Tensor [H, W]
        
        # 返回: 图像张量 + Mask 类别索引 (必须为 long 类型)
        return image, mask.long()


# ========================
# 🎨 数据增强配置
# ========================
def get_train_transform():
    """训练集增强：几何变换 + 色彩扰动 + 归一化"""
    return A.Compose([
        A.Resize(*CONFIG["img_size"]),  # 统一输入尺寸
        A.HorizontalFlip(p=0.5),         # 50% 概率水平翻转
        A.VerticalFlip(p=0.2),           # 20% 概率垂直翻转
        A.Rotate(limit=30, p=0.5),       # ±30° 随机旋转
        A.ColorJitter(                   # 色彩扰动，提升泛化
            brightness=0.1, contrast=0.1, 
            saturation=0.2, hue=0.05, p=0.3
        ),
        A.Normalize(                     # ImageNet 归一化 (匹配预训练编码器)
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),                    # HWC→CHW + numpy→tensor
    ], additional_targets={'mask': 'mask'})


def get_val_transform():
    """验证集变换：仅 Resize + 归一化（无随机性，保证评估稳定）"""
    return A.Compose([
        A.Resize(*CONFIG["img_size"]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})


# ========================
# 🖼️ 可视化预测结果（用于调试）
# ========================
def save_prediction_visualization(image_tensor, mask_gt, mask_pred, save_path):
    """
    保存三图对比: 原图 | 真实 Mask | 预测 Mask
    image_tensor: Tensor [C, H, W], 已归一化
    mask_gt/pred: numpy array [H, W], 类别索引 0/1/2
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # 反归一化: 将 Tensor 转回可视化的 RGB 图像
    img = image_tensor.permute(1, 2, 0).cpu().numpy()  # CHW → HWC
    img = (img * np.array([0.229, 0.224, 0.225]) + 
           np.array([0.485, 0.456, 0.406])).clip(0, 1)
    
    # 绘制原图
    axes[0].imshow(img)
    axes[0].set_title("Input Image", fontsize=10)
    axes[0].axis("off")
    
    # 绘制真实 Mask (cmap='jet' 用颜色区分类别)
    axes[1].imshow(mask_gt, cmap='jet', vmin=0, vmax=2)
    axes[1].set_title("Ground Truth", fontsize=10)
    axes[1].axis("off")
    
    # 绘制预测 Mask
    axes[2].imshow(mask_pred, cmap='jet', vmin=0, vmax=2)
    axes[2].set_title("Prediction", fontsize=10)
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()  # 关闭 figure 防止内存泄漏


# ========================
# 📊 计算评估指标 (mIoU / Dice / Pixel Acc)
# ========================
def compute_metrics(preds, targets, num_classes):
    """
    基于混淆矩阵计算分割指标
    preds/targets: 1D numpy arrays (展平后的预测/真实类别)
    """
    # 计算混淆矩阵: cm[i][j] = 真实类 i 被预测为类 j 的像素数
    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    
    # 逐类别计算 IoU 和 Dice
    ious, dices = [], []
    for cls in range(num_classes):
        tp = cm[cls, cls]  # 真正例
        fp = cm[:, cls].sum() - tp  # 假正例: 其他类被预测为 cls
        fn = cm[cls, :].sum() - tp  # 假反例: cls 被预测为其他类
        
        union = tp + fp + fn
        iou = tp / union if union > 0 else 1.0  # IoU = TP / (TP+FP+FN)
        
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
        # Dice = 2TP / (2TP + FP + FN)
        
        ious.append(iou)
        dices.append(dice)
    
    return {
        'miou': np.mean(ious),  # 平均 IoU (所有类别)
        'dice': np.mean(dices),  # 平均 Dice
        'pixel_acc': np.diag(cm).sum() / cm.sum()  # 像素级准确率
    }


# ========================
# 🚀 训练主函数
# ========================
def train():
    logger.info("=" * 60)
    logger.info("🚀 开始训练 U-Net++ 语义分割模型")
    logger.info(f"📦 配置: encoder={CONFIG['encoder']}, classes={CONFIG['num_classes']}")
    logger.info(f"💻 设备: {CONFIG['device']}")
    logger.info("=" * 60)
    
    # ---------- 1. 构建模型 ----------
    logger.info("🔧 构建模型...")
    model = smp.UnetPlusPlus(
        encoder_name=CONFIG["encoder"],
        encoder_weights=CONFIG["encoder_weights"],  # 加载 ImageNet 预训练权重
        in_channels=CONFIG["in_channels"],
        classes=CONFIG["num_classes"],
        activation=None,  # 输出 logits，由 DiceLoss 内部处理
    ).to(CONFIG["device"])
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"📊 模型参数量: {total_params:.2f}M")
    
    # ---------- 2. 损失函数 + 优化器 + 学习率调度 ----------
    logger.info("⚙️ 配置优化器...")
    
    # 组合损失：Dice + CrossEntropy
    # Dice: 对类别不平衡鲁棒，适合小目标（气相/液相）
    # CE: 训练更稳定，对边界学习更好
    class CombinedLoss(nn.Module):
        def __init__(self, num_classes=3, dice_weight=0.5, ce_weight=0.5):
            super().__init__()
            self.dice = smp.losses.DiceLoss(mode='multiclass')
            self.ce = nn.CrossEntropyLoss()
            self.dice_weight = dice_weight
            self.ce_weight = ce_weight
            
        def forward(self, outputs, masks):
            # outputs: [B, C, H, W] - 模型输出的 logits
            # masks: [B, H, W] - 真实类别索引
            dice_loss = self.dice(outputs, masks)
            ce_loss = self.ce(outputs, masks)
            return self.dice_weight * dice_loss + self.ce_weight * ce_loss
    
    criterion = CombinedLoss(
        num_classes=CONFIG["num_classes"],
        dice_weight=0.5,    # Dice 损失权重
        ce_weight=0.5       # CE 损失权重
    )
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"],weight_decay=1e-4 )
    
    # 当验证损失不再下降时，自动降低学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6,verbose=True
    )
    
    # ---------- 3. 数据加载 ----------
    logger.info("📦 加载数据集...")
    train_dataset = FluidDataset(
        CONFIG["train_img_dir"], CONFIG["train_mask_dir"], 
        transform=get_train_transform()
    )
    val_dataset = FluidDataset(
        CONFIG["val_img_dir"], CONFIG["val_mask_dir"], 
        transform=get_val_transform()
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True, 
        num_workers=CONFIG["num_workers"]
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=False, 
        num_workers=CONFIG["num_workers"]
    )
    
    # 固定一张验证图像用于周期性可视化对比
    sample_img, sample_mask = next(iter(val_loader))
    
    # ---------- 4. 训练状态记录 ----------
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'miou': [],
        'dice': [],
        'pixel_acc': []
    }
    
    best_metric = 0  # 记录最佳验证指标
    patience_counter = 0  # 早停计数器
    
    logger.info("✨ 开始训练循环...")
    
    # ---------- 5. 训练循环 (核心) ----------
    for epoch in range(1, CONFIG["epochs"] + 1):
        # ====== 5.1 训练阶段 ======
        model.train()  # 开启训练模式 (启用 Dropout/BN)
        train_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(CONFIG["device"])  # [B, C, H, W]
            masks = masks.to(CONFIG["device"])    # [B, H, W]
            
            # 前向传播
            optimizer.zero_grad()  # 清空梯度
            outputs = model(images)  # [B, num_classes, H, W]
            loss = criterion(outputs, masks)
            
            # 反向传播 + 参数更新
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)  # 平均训练损失
        
        # ====== 5.2 验证阶段 ======
        model.eval()  # 关闭训练模式
        val_loss = 0.0
        all_preds, all_targets = [], []
        
        with torch.no_grad():  # 验证时不计算梯度，节省显存
            for images, masks in val_loader:
                images = images.to(CONFIG["device"])
                masks = masks.to(CONFIG["device"])
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                # 获取预测类别: logits → 概率 → 类别索引
                preds = torch.argmax(outputs, dim=1)  # [B, H, W]
                
                # 收集所有像素的预测/真实标签 (用于计算指标)
                all_preds.extend(preds.cpu().numpy().flatten())
                all_targets.extend(masks.cpu().numpy().flatten())
        
        val_loss /= len(val_loader)  # 平均验证损失
        
        # ====== 5.3 计算评估指标 ======
        metrics = compute_metrics(
            np.array(all_preds), 
            np.array(all_targets), 
            CONFIG["num_classes"]
        )
        
        # ====== 5.4 记录历史 + 打印日志 ======
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['miou'].append(metrics['miou'])
        history['dice'].append(metrics['dice'])
        history['pixel_acc'].append(metrics['pixel_acc'])
        
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(
            f"Epoch {epoch:3d}/{CONFIG['epochs']} | "
            f"Loss: {train_loss:.4f} / {val_loss:.4f} | "
            f"mIoU: {metrics['miou']:.4f} | "
            f"Dice: {metrics['dice']:.4f} | "
            f"Acc: {metrics['pixel_acc']:.4f} | "
            f"LR: {current_lr:.2e}"
        )
        
        # ====== 5.5 学习率调度 ======
        scheduler.step(val_loss)
        
        # ====== 5.6 保存最佳模型 + 早停判断 ======
        # 选择监控指标 (默认用 mIoU)
        current_metric = metrics[CONFIG["early_stop_metric"]]
        
        if current_metric > best_metric:
            best_metric = current_metric
            # 简单保存模型权重 (仅 state_dict)
            save_path = os.path.join(CONFIG["save_dir"], "best_model.pth")
            torch.save(model.state_dict(), save_path)
            patience_counter = 0  # 重置早停计数
            logger.info(f"✅ 保存最佳模型 @ epoch {epoch} ({CONFIG['early_stop_metric']}={best_metric:.4f})")
            
            # 周期性保存预测可视化图 (用于调试)
            if epoch % CONFIG["save_visual_every"] == 0:
                with torch.no_grad():
                    sample_out = model(sample_img.to(CONFIG["device"]))
                    sample_pred = torch.argmax(sample_out, dim=1).cpu().numpy()[0]
                vis_path = os.path.join(CONFIG["save_dir"], f"vis_epoch{epoch}.png")
                save_prediction_visualization(
                    sample_img[0], 
                    sample_mask[0].numpy(), 
                    sample_pred, 
                    vis_path
                )
        else:
            patience_counter += 1
            logger.info(f"⏳ 早停计数: {patience_counter}/{CONFIG['early_stop_patience']}")
            
            # 触发早停
            if patience_counter >= CONFIG["early_stop_patience"]:
                logger.info(f"🛑 早停触发: {CONFIG['early_stop_metric']} 连续 {patience_counter} 轮未提升")
                break
        
        # ====== 5.7 保存当前轮模型 (可选，用于断点续训) ======
        # torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "last_model.pth"))
    
    # ---------- 6. 训练结束: 绘制曲线 + 总结 ----------
    logger.info("📊 绘制训练曲线...")
    plot_training_history(history, CONFIG["save_dir"])
    
    logger.info("=" * 60)
    logger.info(f"🎉 训练完成！")
    logger.info(f"🏆 最佳 {CONFIG['early_stop_metric']}: {best_metric:.4f}")
    logger.info(f"📁 模型保存至: {CONFIG['save_dir']}")
    logger.info("=" * 60)
    
    return history, best_metric


# ========================
# 📈 绘制训练历史曲线
# ========================
def plot_training_history(history, save_dir):
    """绘制 Loss 和 指标 曲线并保存"""
    epochs = history['epoch']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # (0,0): Loss 曲线
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (0,1): mIoU 曲线
    ax = axes[0, 1]
    ax.plot(epochs, history['miou'], color='green', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mIoU')
    ax.set_title('Mean IoU')
    ax.grid(True, alpha=0.3)
    
    # (1,0): Dice 曲线
    ax = axes[1, 0]
    ax.plot(epochs, history['dice'], color='orange', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Dice Score')
    ax.set_title('Dice Coefficient')
    ax.grid(True, alpha=0.3)
    
    # (1,1): Pixel Accuracy 曲线
    ax = axes[1, 1]
    ax.plot(epochs, history['pixel_acc'], color='red', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Pixel Accuracy')
    ax.set_title('Pixel Accuracy')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Training History', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "training_history.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"📈 训练曲线已保存: {save_path}")


# ========================
# 🏁 程序入口
# ========================
if __name__ == "__main__":
    # 执行训练
    history, best_score = train()
