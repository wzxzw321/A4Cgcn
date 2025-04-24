import json
import os
import shutil
import sys
from configs.CONST import *

def is_closed(points):
    # 检查首尾点是否重合
    return len(points) >= 3 and points[0] == points[-1]

def check_polygon(points):
    if len(points) < 3:
        return False
    return True


def json_to_yolo(json_path, output_dir):
    """将单个JSON标注文件转换为YOLO格式的TXT文件，返回是否成功"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        # 新增格式校验
        if "imageWidth" not in data or "imageHeight" not in data:
            print(f"错误: {json_path} 缺少imageWidth或imageHeight字段")
            return False
    except Exception as e:
        print(f"错误: 无法读取文件 {json_path} - {str(e)}")
        return False

    os.makedirs(output_dir, exist_ok=True)
    img_w = data["imageWidth"]
    img_h = data["imageHeight"]
    txt_content = []

    for shape in data["shapes"]:
        if len(shape["points"]) < 3:
            print(f"警告: {json_path} 中的多边形点数不足3个")
            continue

        if not is_closed(shape["points"]):
            print(f"注意: {json_path} 中的多边形未闭合，将自动闭合")
            shape["points"].append(shape["points"][0])

        first_char = shape["label"][0]
        label_map = {
            '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
            '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'a': 10
        }

        if (class_id := label_map.get(first_char)) is None:
            print(f"警告: 跳过未知标签 {shape['label']}")
            continue

        points = []
        valid = True  # 新增验证标志
        for x, y in shape["points"]:
            nx = max(0.0, min(round(x / img_w, 6), 1.0))
            ny = max(0.0, min(round(y / img_h, 6), 1.0))

            # 修改点：当坐标异常时标记为无效
            if nx < 0 or nx > 1 or ny < 0 or ny > 1:
                print(f"错误: {json_path} 坐标异常 ({x}/{img_w}={nx}, {y}/{img_h}={ny})")
                valid = False
                # 不再继续处理该多边形的其他点
                break

            points.extend([nx, ny])

        if not valid:
            print(f"警告: 跳过无效多边形 - {shape['label']}")
            continue  # 新增跳过逻辑

        txt_content.append(f"{class_id} " + " ".join(map(str, points)))

    try:
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.txt")
        with open(output_path, 'w') as f:
            f.write("\n".join(txt_content))
        return True  # 新增返回值
    except Exception as e:
        print(f"错误: 无法写入文件 {output_path} - {str(e)}")
        return False


def batch_convert(json_dir, output_root):
    """批量转换目录下的所有JSON文件"""
    json_dir = os.path.abspath(json_dir)
    output_root = os.path.abspath(output_root)

    # 新增初始检查
    json_files = []
    for root, _, files in os.walk(json_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    if not json_files:
        print("错误: JSON文件夹中没有JSON文件")
        sys.exit(1)

    # 新增目标文件夹检查
    if os.path.exists(output_root):
        if os.listdir(output_root):
            print(f"警告: 目标文件夹 {output_root} 不为空，将覆盖已有文件")
    else:
        os.makedirs(output_root, exist_ok=True)

    # 新增计数器
    json_count = 0
    txt_count = 0
    png_count = 0

    for json_path in json_files:
        json_count += 1
        relative_path = os.path.relpath(os.path.dirname(json_path), json_dir)
        output_dir = os.path.join(output_root, relative_path)

        # 转换JSON并计数
        if json_to_yolo(json_path, output_dir):
            txt_count += 1

        # 复制PNG文件
        png_path = os.path.splitext(json_path)[0] + '.png'
        if os.path.exists(png_path):
            os.makedirs(output_dir, exist_ok=True)
            shutil.copy2(png_path, os.path.join(output_dir, os.path.basename(png_path)))
            png_count += 1

    # 新增统计输出
    print(f"\n转换结果:")
    print(f"成功处理 {json_count} 个JSON文件")
    print(f"生成 {txt_count} 个TXT文件")
    print(f"复制 {png_count} 个PNG文件")

if __name__ == "__main__":
    # 使用绝对路径配置
    JSON_ROOT = LABELME_PATH
    OUTPUT_ROOT = TRANSLABEL_PATH

    batch_convert(JSON_ROOT, OUTPUT_ROOT)
    print(f"转换完成！结果保存在 {OUTPUT_ROOT}")