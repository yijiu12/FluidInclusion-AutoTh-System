import os
import yaml
from collections import Counter
import glob

def analyze_dataset(data_yaml_path, results_dir=None):
    """
    分析YOLO数据集，统计标签数量、类别分布、负样本等信息
    
    参数:
        data_yaml_path: data.yaml文件路径
        results_dir: 结果保存目录（可选）
    """
    print("🔍 开始分析数据集...")
    
    # 读取data.yaml配置
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    # 获取路径信息
    train_dir = data_config.get('train')  # 这是目录路径
    val_dir = data_config.get('val')      # 这是目录路径
    names = data_config.get('names', {})
    
    print(f"📁 训练集图片目录: {train_dir}")
    print(f"📁 验证集图片目录: {val_dir}")
    print(f"🏷️  类别名称: {names}")
    
    def analyze_image_directory(image_dir, split_name):
        """分析图片目录中的数据集"""
        if not os.path.exists(image_dir):
            print(f"❌ {split_name}目录不存在: {image_dir}")
            return None
        
        # ✅ 修复：只使用小写扩展名，并去重
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files_set = set()
        for ext in image_extensions:
            # 使用 glob 查找，并统一转为小写路径以去重（防止大小写问题）
            found = glob.glob(os.path.join(image_dir, ext), recursive=False)
            found += glob.glob(os.path.join(image_dir, ext.upper()), recursive=False)
            for f in found:
                image_files_set.add(os.path.normpath(f))  # 标准化路径
        
        image_files = sorted(list(image_files_set))
        print(f"📸 在 {split_name} 目录中找到 {len(image_files)} 个图像文件")
        
        if len(image_files) == 0:
            print(f"⚠️  警告: {split_name}目录中没有找到图片文件!")
            return None
        
        total_images = len(image_files)
        total_boxes = 0
        class_counts = Counter()
        negative_samples = 0
        boxes_per_image = []
        missing_labels = 0
        
        print(f"\n📊 分析{split_name}...")
        
        for i, img_path in enumerate(image_files):
            # 获取对应的标签文件路径
            img_dir = os.path.dirname(img_path)
            img_filename = os.path.basename(img_path)
            label_filename = os.path.splitext(img_filename)[0] + '.txt'
            
            # 推断labels目录路径 (images -> labels)
            labels_dir = img_dir.replace('images', 'labels')
            label_path = os.path.join(labels_dir, label_filename)
            
            if not os.path.exists(label_path):
                missing_labels += 1
                negative_samples += 1
                boxes_per_image.append(0)
                continue
            
            # 读取标签文件
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except Exception as e:
                print(f"❌ 读取标签文件失败: {label_path}, 错误: {e}")
                negative_samples += 1
                boxes_per_image.append(0)
                continue
            
            # 检查是否为空文件（负样本）
            valid_lines = [line.strip() for line in lines if line.strip()]
            if len(valid_lines) == 0:
                negative_samples += 1
                boxes_per_image.append(0)
                continue
                
            # 统计标签信息
            image_boxes = 0
            for line in valid_lines:
                parts = line.split()
                if len(parts) >= 5:  # YOLO格式: class x_center y_center width height
                    try:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1
                        total_boxes += 1
                        image_boxes += 1
                    except ValueError:
                        print(f"⚠️  标签格式错误: {label_path}, 行: {line}")
            
            boxes_per_image.append(image_boxes)
            
            # 显示进度
            if (i + 1) % 50 == 0 or (i + 1) == total_images:
                print(f"   📊 已处理 {i + 1}/{total_images} 张图像...")
        
        # 统计结果
        stats = {
            'total_images': total_images,
            'total_boxes': total_boxes,
            'class_counts': dict(class_counts),
            'negative_samples': negative_samples,
            'missing_labels': missing_labels,
            'avg_boxes_per_image': total_boxes / total_images if total_images > 0 else 0,
            'boxes_per_image': boxes_per_image
        }
        
        print(f"✅ {split_name}分析完成:")
        print(f"   📸 总图像数: {stats['total_images']}")
        print(f"   📦 总标注框数: {stats['total_boxes']}")
        print(f"   🚫 负样本数: {stats['negative_samples']} ({stats['negative_samples']/stats['total_images']*100:.1f}%)")
        print(f"   📋 缺失标签文件: {stats['missing_labels']}")
        print(f"   📊 平均每图标注数: {stats['avg_boxes_per_image']:.2f}")
        
        # 显示类别分布
        if stats['class_counts']:
            print(f"   🏷️  类别分布:")
            for class_id, count in sorted(stats['class_counts'].items()):
                class_name = names[class_id] if isinstance(names, list) and class_id < len(names) else f'class_{class_id}'
                print(f"     {class_name}({class_id}): {count} 个")
        else:
            print(f"   🏷️  类别分布: 无标注框")
        
        return stats
    
    # 分析训练集和验证集
    train_stats = analyze_image_directory(train_dir, "训练集")
    val_stats = analyze_image_directory(val_dir, "验证集")
    
    # 汇总信息
    print("\n" + "="*60)
    print("📈 数据集汇总统计")
    print("="*60)
    
    if train_stats and val_stats:
        total_train_val = train_stats['total_images'] + val_stats['total_images']
        total_boxes_all = train_stats['total_boxes'] + val_stats['total_boxes']
        total_negative = train_stats['negative_samples'] + val_stats['negative_samples']
        total_missing = train_stats['missing_labels'] + val_stats['missing_labels']
        
        print(f"📸 总图像数: {total_train_val}")
        print(f"   - 训练集: {train_stats['total_images']}")
        print(f"   - 验证集: {val_stats['total_images']}")
        print(f"📦 总标注框数: {total_boxes_all}")
        print(f"   - 训练集: {train_stats['total_boxes']}")
        print(f"   - 验证集: {val_stats['total_boxes']}")
        print(f"🚫 总负样本数: {total_negative} ({total_negative/total_train_val*100:.1f}%)")
        print(f"📋 总缺失标签文件: {total_missing}")
        print(f"📊 整体平均每图标注数: {total_boxes_all/total_train_val:.2f}")
        
        # 类别汇总
        all_class_counts = Counter(train_stats['class_counts']) + Counter(val_stats['class_counts'])
        if all_class_counts:
            print(f"🏷️  总类别分布:")
            for class_id, count in sorted(all_class_counts.items()):
                class_name = names[class_id] if isinstance(names, list) and class_id < len(names) else f'class_{class_id}'
                print(f"     {class_name}({class_id}): {count} 个")
    
    elif train_stats:
        print(f"📸 总图像数: {train_stats['total_images']} (仅训练集)")
        print(f"📦 总标注框数: {train_stats['total_boxes']}")
        print(f"🚫 总负样本数: {train_stats['negative_samples']} ({train_stats['negative_samples']/train_stats['total_images']*100:.1f}%)")
    
    print("="*60)
    
    # 保存详细报告
    if results_dir:
        save_detailed_report(train_stats, val_stats, names, results_dir)
    
    return train_stats, val_stats

def save_detailed_report(train_stats, val_stats, names, results_dir):
    """保存详细分析报告"""
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, 'dataset_analysis_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("YOLO数据集分析报告\n")
        f.write("=" * 50 + "\n\n")
        
        if train_stats:
            f.write("训练集统计:\n")
            f.write(f"  总图像数: {train_stats['total_images']}\n")
            f.write(f"  总标注框数: {train_stats['total_boxes']}\n")
            f.write(f"  负样本数: {train_stats['negative_samples']} ({train_stats['negative_samples']/train_stats['total_images']*100:.1f}%)\n")
            f.write(f"  缺失标签文件: {train_stats['missing_labels']}\n")
            f.write(f"  平均每图标注数: {train_stats['avg_boxes_per_image']:.2f}\n")
            
            if train_stats['class_counts']:
                f.write("  类别分布:\n")
                for class_id, count in sorted(train_stats['class_counts'].items()):
                    class_name = names[class_id] if isinstance(names, list) and class_id < len(names) else f'class_{class_id}'
                    f.write(f"    {class_name}({class_id}): {count} 个\n")
            f.write("\n")
        
        if val_stats:
            f.write("验证集统计:\n")
            f.write(f"  总图像数: {val_stats['total_images']}\n")
            f.write(f"  总标注框数: {val_stats['total_boxes']}\n")
            f.write(f"  负样本数: {val_stats['negative_samples']} ({val_stats['negative_samples']/val_stats['total_images']*100:.1f}%)\n")
            f.write(f"  缺失标签文件: {val_stats['missing_labels']}\n")
            f.write(f"  平均每图标注数: {val_stats['avg_boxes_per_image']:.2f}\n")
            
            if val_stats['class_counts']:
                f.write("  类别分布:\n")
                for class_id, count in sorted(val_stats['class_counts'].items()):
                    class_name = names[class_id] if isinstance(names, list) and class_id < len(names) else f'class_{class_id}'
                    f.write(f"    {class_name}({class_id}): {count} 个\n")
    
    print(f"📄 详细报告已保存至: {report_path}")

if __name__ == "__main__":
    # 使用示例
    import os
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    data_yaml_path = os.path.join(_PROJECT_ROOT, "dataset", "yolo_detect", "dataset.yaml")
    results_dir = os.path.join(_PROJECT_ROOT, "dataset", "yolo_detect", "analysis_results")
    
    # 分析数据集
    analyze_dataset(data_yaml_path, results_dir)