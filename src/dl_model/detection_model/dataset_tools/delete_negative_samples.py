import os
import shutil

def delete_negative_samples(images_dir, labels_dir):
    """
    删除负样本（没有对应标注文件的图像文件）
    
    Args:
        images_dir: 图像文件夹路径
        labels_dir: 标注文件夹路径
    """
    
    # 确保文件夹存在
    if not os.path.exists(images_dir):
        print(f"错误：图像文件夹不存在 - {images_dir}")
        return
    if not os.path.exists(labels_dir):
        print(f"错误：标注文件夹不存在 - {labels_dir}")
        return
    
    # 获取所有图像文件和标注文件
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    label_files = [f for f in os.listdir(labels_dir) if f.lower().endswith('.txt')]
    
    print(f"找到 {len(image_files)} 个图像文件")
    print(f"找到 {len(label_files)} 个标注文件")
    
    # 创建备份文件夹
    backup_dir = os.path.join(os.path.dirname(images_dir), "backup")
    os.makedirs(backup_dir, exist_ok=True)
    images_backup_dir = os.path.join(backup_dir, "images")
    labels_backup_dir = os.path.join(backup_dir, "labels")
    os.makedirs(images_backup_dir, exist_ok=True)
    os.makedirs(labels_backup_dir, exist_ok=True)
    
    deleted_count = 0
    kept_count = 0
    
    # 遍历所有图像文件
    for image_file in image_files:
        # 获取对应的标注文件名（假设图像和标注文件同名，只是扩展名不同）
        image_name = os.path.splitext(image_file)[0]
        label_file = image_name + '.txt'
        
        label_path = os.path.join(labels_dir, label_file)
        
        # 如果标注文件不存在，或者标注文件为空，则认为是负样本
        if not os.path.exists(label_path):
            # 备份并删除负样本
            shutil.copy2(os.path.join(images_dir, image_file), images_backup_dir)
            os.remove(os.path.join(images_dir, image_file))
            print(f"删除负样本图像: {image_file} (无对应标注文件)")
            deleted_count += 1
        else:
            # 检查标注文件是否为空或没有有效目标
            if is_empty_label_file(label_path):
                # 备份并删除负样本
                shutil.copy2(os.path.join(images_dir, image_file), images_backup_dir)
                shutil.copy2(label_path, labels_backup_dir)
                os.remove(os.path.join(images_dir, image_file))
                os.remove(label_path)
                print(f"删除负样本: {image_file} (标注文件为空)")
                deleted_count += 1
            else:
                kept_count += 1
    
    print(f"\n处理完成！")
    print(f"保留的样本数量: {kept_count}")
    print(f"删除的负样本数量: {deleted_count}")
    print(f"备份文件保存在: {backup_dir}")

def is_empty_label_file(label_path):
    """
    检查标注文件是否为空或没有有效目标
    
    Args:
        label_path: 标注文件路径
        
    Returns:
        bool: 如果是空文件或没有有效目标返回True，否则返回False
    """
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # 如果文件为空
        if not content:
            return True
            
        # 检查每一行是否有有效的数据
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line:
                parts = line.split()
                # 如果至少有类别ID和边界框坐标
                if len(parts) >= 5:
                    return False
                    
        return True
    except Exception as e:
        print(f"读取标注文件出错 {label_path}: {e}")
        return True

def main():
    # 设置文件夹路径（请根据实际数据路径修改）
    import os
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    images_dir = os.path.join(_PROJECT_ROOT, "dataset", "raw", "yolo_detect", "2-0", "images")
    labels_dir = os.path.join(_PROJECT_ROOT, "dataset", "raw", "yolo_detect", "2-0", "labels")
    
    # 确认操作
    print("即将删除负样本文件，建议先备份重要数据！")
    confirm = input("确定要继续吗？(y/n): ")
    
    if confirm.lower() == 'y':
        delete_negative_samples(images_dir, labels_dir)
    else:
        print("操作已取消")

if __name__ == "__main__":
    main()