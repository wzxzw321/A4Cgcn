import os
import random
import shutil
from tqdm import tqdm
from configs.CONST import *


def split_dataset(source_dir, target_root):
    random.seed(SEED)
    # 配置目标路径
    splits = ['train', 'val', 'test']
    labels_dirs = [os.path.join(target_root, split, 'labels') for split in splits]
    images_dirs = [os.path.join(target_root, split, 'images') for split in splits]

    # 创建目录结构
    for split_dir in labels_dirs + images_dirs:
        os.makedirs(split_dir, exist_ok=True)

    # 获取并打乱所有txt文件路径
    txt_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    random.shuffle(txt_files)

    # 计算分割点
    total = len(txt_files)
    train_end = int(total * 0.8)
    val_end = train_end + int(total * 0.1)

    # 执行文件复制
    for i, txt_path in enumerate(tqdm(txt_files, desc='Processing')):
        # 确定目标目录
        if i < train_end:
            split_index = 0  # train
        elif i < val_end:
            split_index = 1  # val
        else:
            split_index = 2  # test

        # 复制标签文件
        txt_name = os.path.basename(txt_path)
        shutil.copy2(txt_path, os.path.join(labels_dirs[split_index], txt_name))

        # 复制对应的图片文件
        png_path = os.path.join(os.path.dirname(txt_path), txt_name.replace('.txt', '.png'))
        if os.path.exists(png_path):
            shutil.copy2(png_path, os.path.join(images_dirs[split_index], os.path.basename(png_path)))
        else:
            print(f"警告: 找不到对应的PNG文件 {png_path}")


if __name__ == "__main__":
    # 配置路径
    SOURCE_DIR = TRANSLABEL_PATH  # 替换为实际源路径
    TARGET_ROOT = YOLO_DATASET_PATH  # 替换为实际目标路径

    split_dataset(SOURCE_DIR, TARGET_ROOT)
    print("数据集分割完成！")