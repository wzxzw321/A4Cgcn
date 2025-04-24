import os
import cv2
import numpy as np
from tqdm import tqdm
from configs.CONST import *

# 配置绝对路径
IMAGE_ROOT = "/home/ygz/ZC/HUS/HUS_ImgSeg/yolov8_dataset/val/images"  # 图片根目录
LABEL_ROOT = "/home/ygz/ZC/HUS/HUS_ImgSeg/yolov8_dataset/val/labels"  # 标签目录
VIS_ROOT = "/home/ygz/ZC/HUS/HUS_ImgSeg/yolov8_dataset/val/visualization"  # 输出目录

# 类别颜色映射（与my_traindata.yml中的names顺序一致）
COLORS = [
    (255, 0, 0),    # 0: 右室 - 红色
    (0, 255, 0),    # 1: 左室 - 绿色
    (0, 0, 255),    # 2: 右房 - 蓝色
    (255, 255, 0),  # 3: 左房 - 青色
    (255, 0, 255),  # 4: 室间隔 - 品红
    (0, 255, 255),  # 5: 房间隔 - 黄色
    (128, 0, 0),    # 6: 主动脉 - 深红
    (0, 128, 0),    # 7: 二尖瓣 - 深绿
    (0, 0, 128),    # 8: 三尖瓣 - 深蓝
    (128, 128, 0),  # 9: 肺动脉 - 橄榄绿
    (128, 0, 128)   # 10: 乳头肌 - 紫色
]

def denormalize_points(normalized_points, img_w, img_h):
    """将归一化坐标转换为像素坐标"""
    points = []
    for i in range(0, len(normalized_points), 2):
        x = int(normalized_points[i] * img_w)
        y = int(normalized_points[i+1] * img_h)
        points.append((x, y))
    return np.array(points, dtype=np.int32)

def visualize_label(image_path, label_path, output_dir):
    """可视化单个图片的标签"""
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"警告: 无法读取图片 {image_path}")
        return

    h, w = img.shape[:2]

    # 读取标签文件
    if not os.path.exists(label_path):
        print(f"警告: 找不到标签文件 {label_path}")
        return

    with open(label_path) as f:
        lines = f.readlines()

    # 绘制每个多边形
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue

        class_id = int(parts[0])
        points = list(map(float, parts[1:]))
        polygon = denormalize_points(points, w, h)

        # 绘制多边形
        cv2.polylines(img, [polygon], isClosed=True, color=COLORS[class_id % len(COLORS)], thickness=2)
        # 绘制类别标签
        cv2.putText(img, f"{class_id}", tuple(polygon[0]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # 保存结果
    rel_path = os.path.relpath(image_path, IMAGE_ROOT)
    output_path = os.path.join(output_dir, rel_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)

def batch_visualize():
    """批量处理所有图片"""
    for root, _, files in os.walk(IMAGE_ROOT):
        for file in tqdm(files, desc="可视化进度"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # 构建对应路径
                img_path = os.path.join(root, file)
                rel_path = os.path.relpath(img_path, IMAGE_ROOT)
                label_path = os.path.join(LABEL_ROOT, os.path.splitext(rel_path)[0] + ".txt")

                visualize_label(img_path, label_path, VIS_ROOT)

if __name__ == "__main__":
    batch_visualize()
    print(f"可视化完成！结果保存在 {VIS_ROOT}")
