import os
import glob
import shutil

def delete_unmatched_images(images_dir, labels_dir, none_dir):
    """
    将images文件夹中不能与labels文件夹中txt文件名称对应的图片移动到None文件夹
    
    参数:
        images_dir: 图片文件夹路径
        labels_dir: 标签文件夹路径
        none_dir: 存放未匹配图片的文件夹路径
    """
    # 创建None文件夹（如果不存在）
    if not os.path.exists(none_dir):
        os.makedirs(none_dir)
        print(f"创建文件夹: {none_dir}")
    
    # 获取所有图片文件和标签文件
    image_files = glob.glob(os.path.join(images_dir, "*.png"))
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    
    print(f"找到 {len(image_files)} 个图片文件")
    print(f"找到 {len(label_files)} 个标签文件")
    
    # 从标签文件名提取基础名称（去掉扩展名）
    label_basenames = set()
    for label_file in label_files:
        basename = os.path.basename(label_file)  # 如 "zone2 (4).txt"
        name_without_ext = os.path.splitext(basename)[0]  # 如 "zone2 (4)"
        label_basenames.add(name_without_ext)
    
    print(f"提取到 {len(label_basenames)} 个唯一的标签名称")
    
    # 检查每个图片文件是否有对应的标签文件
    moved_count = 0
    kept_count = 0
    
    for image_file in image_files:
        basename = os.path.basename(image_file)  # 如 "zone1 (1).png"
        name_without_ext = os.path.splitext(basename)[0]  # 如 "zone1 (1)"
        
        # 如果图片名称在标签名称集合中，则保留；否则移动到None文件夹
        if name_without_ext in label_basenames:
            kept_count += 1
            print(f"保留: {basename}")
        else:
            moved_count += 1
            # 构建目标路径
            destination = os.path.join(none_dir, basename)
            print(f"移动: {basename} -> {none_dir}")
            # 移动文件
            shutil.move(image_file, destination)
    
    print(f"\n处理完成!")
    print(f"保留图片: {kept_count} 个")
    print(f"移动图片: {moved_count} 个")
    print(f"未匹配图片保存在: {none_dir}")

if __name__ == "__main__":
    # 设置文件夹路径（请根据实际数据路径修改）
    import os
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    images_dir = os.path.join(_PROJECT_ROOT, "dataset", "raw", "yolo_detect", "3-3", "images")
    labels_dir = os.path.join(_PROJECT_ROOT, "dataset", "raw", "yolo_detect", "3-3", "labels")
    none_dir = os.path.join(_PROJECT_ROOT, "dataset", "raw", "yolo_detect", "3-3", "None")
    
    # 检查文件夹是否存在
    if not os.path.exists(images_dir):
        print(f"错误: 图片文件夹不存在 - {images_dir}")
    elif not os.path.exists(labels_dir):
        print(f"错误: 标签文件夹不存在 - {labels_dir}")
    else:
        # 执行移动操作
        delete_unmatched_images(images_dir, labels_dir, none_dir)