import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class QualitySegDatasetYOLO(Dataset):
    """
    自定义 Dataset，用于加载 YOLO 格式的分割标签 + 质量分数

    假设数据组织如下（以 train 为例）：
      train/
        images/          # 存放 *.png 图像
            0001.png
            0002.png
            ...
        labels/          # 存放 YOLO segmentation 格式的 txt
            0001.txt
            0002.txt
            ...
        quality/         # 存放单个整数 1–5，每张图一个 txt
            0001.txt
            0002.txt
            ...

    每个分割标签 txt 内容示例（多行，每行一个实例）：
      类别 x1 y1 x2 y2 x3 y3 ... xn yn
      0     0.123 0.234 0.456 0.789 0.678 0.345 ... 0.321 0.654
      0     0.456 0.012 0.789 0.345 0.901 0.567 ... 0.222 0.111
    其中 (x, y) 都是相对于原图（width, height）的归一化坐标，范围[0,1]。

    质量分数 txt 里直接写一个 1–5 的整数，比如 “4”。
    """

    def __init__(self, img_dir, label_dir, quality_dir, img_size=640, transform=None):
        """
        Args:
          img_dir     (str): 存放 *.png 图像的目录，比如 "/…/train/images/"
          label_dir   (str): 存放 YOLO segmentation txt 的目录，比如 "/…/train/labels/"
          quality_dir (str): 存放质量分数 txt 的目录，比如 "/…/train/quality/"
          img_size    (int): 最终把图都 resize 成 (img_size, img_size)
          transform   (callable 或 None): 如果需要对 img+mask 做额外的数据增强，可传入一个 callable，
                                         接受 (img_tensor, mask_tensor)，返回新 (img_tensor, mask_tensor)。
        """
        super().__init__()
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.quality_dir = quality_dir
        self.img_size = img_size
        self.transform = transform

        # 列出所有图片名称（不含后缀），假定只处理 .png
        self.img_names = [
            f.split('.')[0]
            for f in os.listdir(img_dir)
            if f.lower().endswith('.png')
        ]
        self.img_names.sort()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        """
        返回：img_tensor, mask_tensor, quality_tensor
          img_tensor:   torch.float32, shape [3, img_size, img_size], 值范围 [0,1]
          mask_tensor:  torch.long,    shape [1, img_size, img_size], 值为 {0,1}
          quality_tensor: torch.float32, shape [1], 值为 1-5
        """
        name = self.img_names[idx]
        # ——————————— 1. 读取并 resize 图像 ——————————— #
        img_path = os.path.join(self.img_dir, name + '.png')
        img_bgr = cv2.imread(img_path)  # BGR
        if img_bgr is None:
            raise FileNotFoundError(f"找不到图片: {img_path}")
        # 原始宽高（用于计算坐标时，不直接使用，而是用 resize 后的大小）
        # 但由于我们会把所有图像 resize 到 (img_size, img_size)，可以直接在归一化坐标基础上乘以 img_size。
        # 下面先做 resize：
        img_resized = cv2.resize(img_bgr, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        # BGR -> RGB -> 归一化 -> CHW
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0  # H×W×3, float32
        img_chw = np.transpose(img_float, (2, 0, 1))  # 3×H×W
        img_tensor = torch.from_numpy(img_chw)  # torch.float32

        # ——————————— 2. 构建分割 mask ——————————— #
        # 先给一个全背景的 mask (0 表示背景)
        mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)  # H×W, uint8

        label_path = os.path.join(self.label_dir, name + '.txt')
        if not os.path.exists(label_path):
            # 如果没有对应的分割标签，则默认全背景 mask (或报错)
            # raise FileNotFoundError(f"找不到分割标签: {label_path}")
            pass
        else:
            with open(label_path, 'r') as f:
                lines = [x.strip() for x in f.readlines() if x.strip() != ""]
            for line in lines:
                # 每行格式："class x1 y1 x2 y2 ... xn yn"
                parts = line.split()
                # 第一个值是类别，不拆也可，只要给 mask 标记前景即可
                cls_id = int(float(parts[0]))
                # 后面就是坐标序列，假设坐标是归一化的 [0,1]
                coords = [float(x) for x in parts[1:]]
                if len(coords) % 2 != 0:
                    raise ValueError(f"{label_path} 中这一行坐标数不是偶数: {line}")
                # 坐标对数
                num_points = len(coords) // 2
                pts = []
                for i in range(num_points):
                    x_norm = coords[2 * i]
                    y_norm = coords[2 * i + 1]
                    # 这里把归一化坐标 * img_size 得到像素坐标
                    # 注意：OpenCV fillPoly 需要整型坐标
                    x_pix = int(round(x_norm * self.img_size))
                    y_pix = int(round(y_norm * self.img_size))
                    # 同时 clamp 到 [0, img_size-1]
                    x_pix = max(0, min(self.img_size - 1, x_pix))
                    y_pix = max(0, min(self.img_size - 1, y_pix))
                    pts.append([x_pix, y_pix])
                if len(pts) >= 3:
                    # 构造一个 (N, 1, 2) 的整数数组给 fillPoly
                    polygon = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                    # fillPolygon，把该多边形区域置为 1
                    cv2.fillPoly(mask, [polygon], color=1)
                else:
                    # 如果点数小于 3，则无法构成多边形，可以忽略或者打印警告
                    # print(f"警告: {label_path} 中多边形点数 < 3，已跳过: {pts}")
                    pass

        # 转成 torch tensor，[1, H, W], long
        mask_tensor = torch.from_numpy(mask).long().unsqueeze(0)

        # ——————————— 3. 读取质量分数 ——————————— #
        quality_path = os.path.join(self.quality_dir, name + '.txt')
        if not os.path.exists(quality_path):
            raise FileNotFoundError(f"找不到质量分数文件: {quality_path}")
        with open(quality_path, 'r') as f:
            score_str = f.read().strip()
        try:
            score_val = float(score_str)
        except:
            raise ValueError(f"{quality_path} 中无法解析 float: '{score_str}'")
        # 约定质量分数在 [1,5] 之间
        quality_tensor = torch.tensor([score_val], dtype=torch.float32)  # [1]

        # ——————————— 4. （可选）数据增强/标准化 ——————————— #
        if self.transform is not None:
            img_tensor, mask_tensor = self.transform(img_tensor, mask_tensor)

        return img_tensor, mask_tensor, quality_tensor
