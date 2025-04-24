import os
import random
import shutil
from tqdm import tqdm

# 源目录配置
source_dir = "/home/ygz/ZC/HUS/HUS_ImgSeg/yolov8_dataset/CobotDataset_A4C"
train_dir = "/home/ygz/ZC/HUS/HUS_ImgSeg/yolov8_dataset/train/images"
val_dir = "/home/ygz/ZC/HUS/HUS_ImgSeg/yolov8_dataset/val/images"
test_dir = "/home/ygz/ZC/HUS/HUS_ImgSeg/yolov8_dataset/test/images"

# 删除旧目录（新增json目录清理）
dirs_to_remove = [
    train_dir, val_dir, test_dir,
    train_dir.replace("images", "img_json"),
    val_dir.replace("images", "img_json"),
    test_dir.replace("images", "img_json")
]
for dir_path in dirs_to_remove:
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)

# 创建目标目录（新增json目录创建）
os.makedirs(train_dir.replace("images", "img_json"), exist_ok=True)
os.makedirs(val_dir.replace("images", "img_json"), exist_ok=True)
os.makedirs(test_dir.replace("images", "img_json"), exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 获取所有PNG文件并打乱顺序
png_files = [f for f in os.listdir(source_dir) if f.endswith('.png')]
random.shuffle(png_files)

# 计算分割点
total = len(png_files)
train_end = int(total * 0.8)
val_end = train_end + int(total * 0.1)

# 文件复制流程
for i, png_file in enumerate(tqdm(png_files, desc='Processing files')):
    base_name = os.path.splitext(png_file)[0]
    json_file = f"{base_name}.json"

    src_png = os.path.join(source_dir, png_file)
    src_json = os.path.join(source_dir, json_file)

    # 确定目标路径
    if i < train_end:
        img_dest = train_dir
        json_dest = img_dest.replace("images", "img_json")
    elif i < val_end:
        img_dest = val_dir
        json_dest = img_dest.replace("images", "img_json")
    else:
        img_dest = test_dir
        json_dest = img_dest.replace("images", "img_json")

    # 复制图片到images目录
    shutil.copy(src_png, os.path.join(img_dest, png_file))

    # 复制JSON到img_json目录
    if os.path.exists(src_json):
        shutil.copy(src_json, os.path.join(json_dest, json_file))
    else:
        print(f"警告: 找不到对应的JSON文件 {json_file}")

print(f"数据集分割完成：\n训练集: {train_end} 个\n验证集: {val_end - train_end} 个\n测试集: {total - val_end} 个")
