import cv2
import numpy as np
import os
import glob
from pathlib import Path
from tqdm import tqdm

# 项目根目录（向上四级）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

class EdgeOnlyPreprocessor:
    def __init__(self):
        self.input_folder = os.path.join(_PROJECT_ROOT, "dataset", "raw", "input")
        self.output_folder = os.path.join(_PROJECT_ROOT, "dataset", "raw", "output")
        
        # 创建输出文件夹
        os.makedirs(self.output_folder, exist_ok=True)
        print(f"✅ 输入文件夹: {self.input_folder}")
        print(f"✅ 输出文件夹: {self.output_folder}")
    
    def edge_enhancement(self, image):
        """
        边缘增强 - 使用拉普拉斯算子
        参数保持原样：strength=0.5
        """
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        sharp = image - 0.5 * laplacian
        sharp = np.clip(sharp, 0, 255).astype(np.uint8)
        return sharp
    
    def process_single_image(self, image_path):
        """处理单张图像"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"⚠️ 无法读取图像: {image_path}")
                return False
            
            # 应用边缘增强
            enhanced_image = self.edge_enhancement(image)
            
            # 保存结果
            filename = Path(image_path).name
            output_path = os.path.join(self.output_folder, f"{filename}")
            success = cv2.imwrite(output_path, enhanced_image)
            
            if success:
                print(f"✓ 已增强: {filename}")
            else:
                print(f"✗ 保存失败: {filename}")
                
            return success
            
        except Exception as e:
            print(f"❌ 处理 {image_path} 时出错: {str(e)}")
            return False
    
    def batch_process(self):
        """批量处理所有图像"""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        image_paths = []
        
        for ext in image_extensions:
            pattern = os.path.join(self.input_folder, ext)
            image_paths.extend(glob.glob(pattern))
            pattern_upper = os.path.join(self.input_folder, ext.upper())
            image_paths.extend(glob.glob(pattern_upper))
        
        image_paths = list(set(image_paths))
        
        if not image_paths:
            print(f"📁 在文件夹 {self.input_folder} 中未找到图像文件")
            return
        
        print(f"📦 找到 {len(image_paths)} 张图像，开始边缘增强处理...")
        
        success_count = 0
        for image_path in tqdm(image_paths, desc="边缘增强中"):
            if self.process_single_image(image_path):
                success_count += 1
        
        print(f"\n🎉 处理完成！成功增强 {success_count}/{len(image_paths)} 张图像")
        print(f"📂 增强后的图像保存在: {self.output_folder}")

# 🚀 主程序
if __name__ == "__main__":
    print("🚀 开始使用边缘增强处理流体包裹体图像...")
    print("=" * 60)
    
    processor = EdgeOnlyPreprocessor()
    processor.batch_process()
    
    print("=" * 60)
    print("🎯 本次处理说明:")
    print("   • 仅使用拉普拉斯边缘增强方法")
    print("   • 参数保持原样: strength=0.5")
    print("   • 输出文件命名格式: enhanced_原文件名")
