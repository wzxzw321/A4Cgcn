import os
import math
import shutil
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from configs.CONST import *
from data_transform_cobot import is_closed, check_polygon

def resample_polygon(points, n_points):
    """均匀重采样多边形轮廓（新增格式验证和转换）"""
    # 转换为OpenCV需要的格式：CV_32F类型的二维数组
    try:
        # 转换为numpy数组并添加维度
        points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    except Exception as e:
        print(f"坐标转换错误: {str(e)}")
        return np.array([])

    # 复用已有的多边形验证逻辑（来自data_transform_cobot.py）
    if len(points) < 3 or not check_polygon(points.squeeze()):
        print("警告: 无效多边形，点数不足或无法构成闭合图形")
        return np.array([])

    # 复用已有的闭合性检查（来自data_transform_cobot.py）
    if not is_closed(points.squeeze().tolist()):
        print("自动闭合多边形")
        points = np.vstack([points, points[0]])

    perimeter = cv2.arcLength(points, closed=True)
    # 使用approxPolyDP进行重采样（保持形状）
    epsilon = 0.005 * perimeter
    approx = cv2.approxPolyDP(points, epsilon, closed=True)

    # 均匀间隔采样
    sampled_points = []
    for i in range(n_points):
        ratio = i / n_points
        idx = int(ratio * len(approx))
        sampled_points.append(approx[idx][0])

    return np.array(sampled_points)


def process_labels(source_labels_dir, source_images_dir, target_root, n_points=20):
    """处理标签并复制图片"""
    target_labels = Path(target_root) / 'labels'
    target_images = Path(target_root) / 'images'

    # 遍历所有标签文件
    for label_path in tqdm(list(Path(source_labels_dir).rglob('*.txt')), desc='Processing'):
        # 读取原始标签
        with open(label_path, 'r') as f:
            lines = f.readlines()

        # 获取对应图片路径
        img_name = label_path.stem + '.png'
        img_path = Path(source_images_dir) / img_name

        if not img_path.exists():
            print(f"警告: 图片 {img_path} 不存在")
            continue

        # 获取图片尺寸
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        # 处理每个多边形
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            class_id = parts[0]
            points = list(map(float, parts[1:]))

            # 转换为像素坐标
            pixel_points = [(x * w, y * h) for x, y in zip(points[::2], points[1::2])]

            # 重采样
            resampled = resample_polygon(pixel_points, n_points)

            # 闭合处理
            if len(resampled) > 0:
                # 添加首点实现强制闭合
                resampled = np.vstack([resampled, resampled[0]])

            # 转换回归一化坐标
            normalized = [[x / w, y / h] for x, y in resampled]

            # 生成新行
            flat_coords = [f"{coord:.6f}" for pair in normalized for coord in pair]
            new_lines.append(f"{class_id} " + " ".join(flat_coords))

        # 保存新标签
        rel_path = label_path.relative_to(source_labels_dir)
        output_label = target_labels / rel_path
        output_label.parent.mkdir(parents=True, exist_ok=True)

        with open(output_label, 'w') as f:
            f.write("\n".join(new_lines))

        # 复制图片
        output_img = target_images / rel_path.parent.name / img_name
        output_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, output_img)


if __name__ == "__main__":
    # 配置路径
    SOURCE_LABELS = TRANSLABEL_PATH
    SOURCE_IMAGES = TRANSLABEL_PATH
    TARGET_ROOT = YOLO_DATASET_ORG_PATH
    SAMPLE_POINTS = POINTS

    process_labels(SOURCE_LABELS, SOURCE_IMAGES, TARGET_ROOT, SAMPLE_POINTS)
    print("重采样完成！新数据集结构：")
    print(f"标签位置: {Path(TARGET_ROOT) / 'labels'}")
    print(f"图片位置: {Path(TARGET_ROOT) / 'images'}")