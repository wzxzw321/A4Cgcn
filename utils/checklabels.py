import os
import cv2
import numpy as np
from configs.CONST import *
def check_label_file(txt_path, num_classes=11):
    """检查单个标签文件的合法性"""
    errors = []

    try:
        with open(txt_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        return [f"文件读取失败: {str(e)}"]

    if not lines:
        errors.append("空文件警告: 文件不包含任何标注信息")

    for line_num, line in enumerate(lines, 1):
        parts = line.strip().split()

        # 检查基本格式
        if len(parts) < 3:
            errors.append(f"第{line_num}行: 数据不完整 (至少需要类别+1个坐标)")
            continue

        # 检查类别ID
        try:
            class_id = int(parts[0])
            if not (0 <= class_id < num_classes):
                errors.append(f"第{line_num}行: 类别ID {class_id} 超出有效范围(0-{num_classes-1})")
        except ValueError:
            errors.append(f"第{line_num}行: 无效的类别ID格式 '{parts[0]}'")
            continue

        # 检查坐标值
        coords = parts[1:]
        if len(coords) % 2 != 0:
            errors.append(f"第{line_num}行: 坐标点数量为奇数 ({len(coords)}个值)")

        for i, val in enumerate(coords):
            try:
                v = float(val)
                if not (0.0 <= v <= 1.0):
                    errors.append(f"第{line_num}行: 坐标值[{i}] {v} 超出[0,1]范围")
            except ValueError:
                errors.append(f"第{line_num}行: 无效的浮点数格式 '{val}'")

        if len(coords) >= 6:  # 3个点需要6个坐标值
            try:
                # 将归一化坐标转换为实际坐标（假设图像尺寸为640x640）
                points = np.array([[float(coords[i]) * 1150, float(coords[i + 1]) * 795]
                                   for i in range(0, len(coords), 2)])

                # 计算多边形方向
                area = cv2.contourArea(points.astype(np.float32))
                if area == 0:
                    errors.append(f"第{line_num}行: 多边形面积为零")
                elif area < 0:
                    errors.append(f"第{line_num}行: 多边形为顺时针方向 (建议使用逆时针)")
                # 正值表示逆时针方向，不需要报错

            except Exception as e:
                errors.append(f"第{line_num}行: 坐标解析失败 - {str(e)}")

    return errors

def check_label_dir(label_dir):
    """批量检查目录下的所有标签文件"""
    total_files = 0
    error_files = 0
    error_log = []

    for root, _, files in os.walk(label_dir):
        for file in files:
            if file.endswith('.txt'):
                total_files += 1
                file_path = os.path.join(root, file)
                errors = check_label_file(file_path)

                if errors:
                    error_files += 1
                    error_log.append(f"\n=== {file_path} ===")
                    error_log.extend(errors)

    # 生成统计报告
    report = [
        f"\n检查完成！共处理 {total_files} 个文件",
        f"发现问题的文件: {error_files} 个",
        f"有效文件比例: {(total_files - error_files)/total_files:.1%}"
    ]

    if error_files > 0:
        report.append("\n详细错误信息:")
        report.extend(error_log)

    return '\n'.join(report)

if __name__ == "__main__":
    LABEL_DIR = YOLO_DATASET_ORG_PATH + "/train/labels"
    print(check_label_dir(LABEL_DIR))
    LABEL_DIR = YOLO_DATASET_ORG_PATH + "/test/labels"
    print(check_label_dir(LABEL_DIR))
    LABEL_DIR = YOLO_DATASET_ORG_PATH + "/val/labels"
    print(check_label_dir(LABEL_DIR))
