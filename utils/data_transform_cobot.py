import json
import os

def is_closed(points):
    # 检查首尾点是否重合
    return len(points) >= 3 and points[0] == points[-1]

def check_polygon(points):
    if len(points) < 3:
        return False
    return True

def json_to_yolo(json_path, output_dir):
    """将单个JSON标注文件转换为YOLO格式的TXT文件"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"错误: 无法读取文件 {json_path} - {str(e)}")
        return

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 获取图片尺寸
    img_w = data["imageWidth"]
    img_h = data["imageHeight"]

    # 生成YOLO格式内容
    txt_content = []
    for shape in data["shapes"]:
        # 添加多边形有效性检查
        if len(shape["points"]) < 3:
            print(f"警告: {json_path} 中的多边形点数不足3个")
            continue

        # 检查闭合性（可选）
        if not is_closed(shape["points"]):
            print(f"注意: {json_path} 中的多边形未闭合，将自动闭合")
            shape["points"].append(shape["points"][0])  # 自动闭合
        # 提取标签首字符并映射
        first_char = shape["label"][0]
        label_map = {
            '0':0, '1':1, '2':2, '3':3, '4':4,
            '5':5, '6':6, '7':7, '8':8, '9':9, 'a':10
        }

        if (class_id := label_map.get(first_char)) is None:
            print(f"警告: 跳过未知标签 {shape['label']}")
            continue

        # 转换并归一化坐标
        points = []
        for x, y in shape["points"]:
            nx = max(0.0, min(round(x / img_w, 6), 1.0))  # 确保坐标在[0,1]之间
            ny = max(0.0, min(round(y / img_h, 6), 1.0))
            points.extend([nx, ny])

            # 添加坐标值验证
            if nx < 0 or nx > 1 or ny < 0 or ny > 1:
                print(f"错误: {json_path} 归一化坐标异常 ({x}/{img_w}={nx}, {y}/{img_h}={ny})")
                continue

        txt_content.append(f"{class_id} " + " ".join(map(str, points)))

    # 写入文件
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.txt")
    with open(output_path, 'w') as f:
        f.write("\n".join(txt_content))

def batch_convert(json_dir, output_root):
    """批量转换目录下的所有JSON文件"""
    json_dir = os.path.abspath(json_dir)
    output_root = os.path.abspath(output_root)

    for root, _, files in os.walk(json_dir):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                # 保持原始目录结构
                relative_path = os.path.relpath(root, json_dir)
                output_dir = os.path.join(output_root, relative_path)
                json_to_yolo(json_path, output_dir)

if __name__ == "__main__":
    # 使用绝对路径配置
    JSON_ROOT = "/home/ygz/ZC/HUS/HUS_ImgSeg/yolov8_dataset/train_test/img_json"  # JSON源目录
    OUTPUT_ROOT = "/home/ygz/ZC/HUS/HUS_ImgSeg/yolov8_dataset/train_test/labels"         # 输出目录

    batch_convert(JSON_ROOT, OUTPUT_ROOT)
    print(f"转换完成！结果保存在 {OUTPUT_ROOT}")
