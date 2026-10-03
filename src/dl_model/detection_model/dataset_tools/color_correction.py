import cv2
import numpy as np
from pathlib import Path

class AutoWhiteBalance:
    """自动白平衡校正类"""
    
    def __init__(self, method='grayworld'):
        self.method = method
    
    def gray_world(self, img: np.ndarray) -> np.ndarray:
        """
        灰度世界假设自动白平衡
        假设图像平均颜色应该是灰色（R=G=B）
        """
        img = img.astype(np.float32)
        
        # 计算各通道均值
        mean_r = np.mean(img[:, :, 2])
        mean_g = np.mean(img[:, :, 1])
        mean_b = np.mean(img[:, :, 0])
        
        # 计算增益（以G通道为基准）
        kr = mean_g / mean_r if mean_r != 0 else 1.0
        kb = mean_g / mean_b if mean_b != 0 else 1.0
        kg = 1.0
        
        # 应用增益
        img[:, :, 2] *= kr  # R
        img[:, :, 1] *= kg  # G
        img[:, :, 0] *= kb  # B
        
        return np.clip(img, 0, 255).astype(np.uint8)
    
    def white_patch(self, img: np.ndarray, percentile=95) -> np.ndarray:
        """
        白点法：假设最亮的点是白色
        """
        img = img.astype(np.float32)
        
        # 找到每个通道的亮部百分位点作为"白点"
        white_r = np.percentile(img[:, :, 2], percentile)
        white_g = np.percentile(img[:, :, 1], percentile)
        white_b = np.percentile(img[:, :, 0], percentile)
        
        # 计算增益（使白点变为纯白）
        max_val = 255.0
        kr = max_val / white_r if white_r > 0 else 1.0
        kg = max_val / white_g if white_g > 0 else 1.0
        kb = max_val / white_b if white_b > 0 else 1.0
        
        # 归一化增益（防止过度校正）
        max_gain = max(kr, kg, kb)
        kr, kg, kb = kr/max_gain, kg/max_gain, kb/max_gain
        
        img[:, :, 2] *= kr
        img[:, :, 1] *= kg
        img[:, :, 0] *= kb
        
        return np.clip(img, 0, 255).astype(np.uint8)
    
    def process(self, img: np.ndarray) -> np.ndarray:
        if self.method == 'grayworld':
            return self.gray_world(img)
        elif self.method == 'whitepatch':
            return self.white_patch(img)
        else:
            # OpenCV 内置白平衡
            wb = cv2.xphoto.createGrayworldWB()
            return wb.balanceWhite(img)


def batch_white_balance(input_dir: str, output_dir: str, method='grayworld'):
    """批量白平衡处理"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    resizer = AutoWhiteBalance(method=method)
    image_paths = list(Path(input_dir).glob('*.jpg')) + \
                  list(Path(input_dir).glob('*.png'))
    
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # 白平衡校正
        img_corrected = resizer.process(img)
        
        # 保存
        output_path = Path(output_dir) / img_path.name
        cv2.imwrite(str(output_path), img_corrected)
        
        print(f"✓ 处理完成: {img_path.name}")


# 使用示例
if __name__ == "__main__":
    # 项目根目录（向上四级）
    import os
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    batch_white_balance(
        input_dir=os.path.join(_PROJECT_ROOT, "dataset", "raw", "input"),      # 偏蓝图像目录
        output_dir=os.path.join(_PROJECT_ROOT, "dataset", "raw", "output"),      # 校正后输出目录
        method='grayworld'                       # 或 'whitepatch'
    )