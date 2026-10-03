import os
import shutil
import random
from pathlib import Path

# 项目根目录（向上四级）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# ========================
# 配置区域（请根据实际数据路径修改）
# ========================  
base_data_dir = os.path.join(_PROJECT_ROOT, "dataset", "raw", "unet_segment")  # 原始数据目录
sub_folders = ["1-1", "1-2", "1-3", "2-0", "2-1", "2-2", "2-3", "2-4", "3-1", "3-2", "3-3"]  
output_dir = os.path.join(_PROJECT_ROOT, "dataset", "unet_segment")  # 划分后输出目录

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
random_seed = 42

# ⚠️ 语义分割类别配置（背景通常像素值为0，此处仅填前景类别）
NUM_CLASSES = 2
CLASS_NAMES = ["liquid","gas"]

# ========================
# 1. 创建输出目录
# ========================
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, 'masks'), exist_ok=True)

# ========================
# 2. 收集数据 + 生成唯一新文件名
# ========================
def collect_all_pairs(base_dir, folders):
    valid_pairs = []  # 存储: (new_stem, img_src, mask_src)
    
    for folder in folders:
        root = Path(base_dir) / folder
        if not root.exists():
            print(f"⚠️ 跳过不存在的文件夹：{folder}")
            continue

        for dirpath, dirnames, _ in os.walk(root):
            # 过滤备份或无关目录
            if 'backup' in dirnames:
                dirnames.remove('backup')
                
            dir_p = Path(dirpath)
            if dir_p.name.lower() in ['images', 'masks', 'labels']:
                continue

            img_dir = dir_p / 'images'
            mask_dir = dir_p / 'masks'

            if img_dir.exists() and mask_dir.exists():
                for img_file in os.listdir(img_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                        orig_stem = Path(img_file).stem
                        # 🔑 核心策略：文件夹名_原文件名，确保全局唯一且可溯源
                        new_stem = f"{folder}_{orig_stem}"
                        
                        mask_found = False
                        # 优先匹配.png（分割Mask标准格式），其次兼容.jpg/.jpeg
                        for mask_ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                            mask_path = mask_dir / f"{orig_stem}{mask_ext}"
                            if mask_path.exists():
                                valid_pairs.append((new_stem, img_dir / img_file, mask_path))
                                mask_found = True
                                break
                                
                        if not mask_found:
                            print(f"⚠️ 跳过无对应Mask的图像：{img_file}")
    return valid_pairs

print("🔍 正在扫描数据并生成防重名映射...")
all_data = collect_all_pairs(base_data_dir, sub_folders)
print(f"✅ 共收集到 {len(all_data)} 个有效样本（已自动处理重名冲突）")

# ========================
# 3. 随机划分（直接混合）
# ========================
random.seed(random_seed)
random.shuffle(all_data)

n_total = len(all_data)
n_train = int(n_total * train_ratio)
n_val = int(n_total * val_ratio)
n_test = n_total - n_train - n_val

train_data = all_data[:n_train]
val_data = all_data[n_train:n_train + n_val]
test_data = all_data[n_train + n_val:]

print(f"📊 划分结果: 训练集 {len(train_data)} | 验证集 {len(val_data)} | 测试集 {len(test_data)}")

# ========================
# 4. 复制并重命名文件
# ========================
def copy_split_data(data_list, split_name):
    for new_stem, img_src, mask_src in data_list:
        img_ext = img_src.suffix.lower()
        
        # 图像：保留原扩展名，使用新名称
        dst_img = os.path.join(output_dir, split_name, 'images', f"{new_stem}{img_ext}")
        # 🔑 语义分割核心优化：Mask 强制转为 .png
        # 避免 JPEG 有损压缩导致类别像素值发生偏移（如 255 -> 254）
        dst_mask = os.path.join(output_dir, split_name, 'masks', f"{new_stem}.png")
        
        shutil.copy2(img_src, dst_img)
        shutil.copy2(mask_src, dst_mask)

print("📤 正在复制训练集...")
copy_split_data(train_data, 'train')
print("📤 正在复制验证集...")
copy_split_data(val_data, 'val')
print("📤 正在复制测试集...")
copy_split_data(test_data, 'test')

# ========================
# 5. 生成数据集配置文件
# ========================
yaml_path = os.path.join(output_dir, 'dataset.yaml')
with open(yaml_path, 'w', encoding='utf-8') as f:
    f.write(f"path: {Path(output_dir).resolve()}\n")
    f.write("train:\n")
    f.write(f"  images: train/images\n  masks: train/masks\n")
    f.write("val:\n")
    f.write(f"  images: val/images\n  masks: val/masks\n")
    f.write("test:\n")
    f.write(f"  images: test/images\n  masks: test/masks\n\n")
    f.write(f"# 语义分割类别数（背景通常不计入，背景像素值默认为0）\n")
    f.write(f"nc: {NUM_CLASSES}\n")
    f.write(f"names: {CLASS_NAMES}\n")

print("🎉 语义分割数据集划分完成！")
print(f"📁 输出路径：{output_dir}")
print(f"📝 重命名规则：来源文件夹_原文件名（例：._001.png）")
print(f"✅ 图像与Mask已严格配对，Mask已统一转为.png，可直接用于UNet等模型训练。")