import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# from ultralytics.utils.ops import resize, segment2mask


class SegmentationWithQualityDataset(Dataset):
    """
    用于 YOLOv8 分割任务的自定义 Dataset，同时读取 quality 文件夹的质量分数。
    数据目录结构示例：
        root/
        ├── train/
        │   ├── images/       # *.jpg / *.png
        │   ├── labels/       # *.txt，每行：class x1 y1 x2 y2 ... （所有坐标为归一化 [0,1]）
        │   └── quality/      # *.txt，每行单个数字（1–5）
        ├── val/  (同上)
        └── test/ (同上)

    每个标签文件（如 train/labels/0001.txt）示例：
        2 0.1 0.2 0.15 0.25 0.2 0.2 0.15 0.15
        0 0.6 0.7 0.65 0.75 0.7 0.7 0.65 0.65
    表示有两个人工实例：
        - 第一行：类别=2，后面 (0.1,0.2),(0.15,0.25),(0.2,0.2),(0.15,0.15) 是归一化边界多边形点
        - 第二行：类别=0，后面是另一组多边形点。

    本 Dataset 在 __getitem__ 中返回：
        {
          "img":      Tensor [3, img_size, img_size]，float32，[0,1]
          "masks":    Tensor [N, img_size, img_size]，uint8，N 是实例数
          "cls":      Tensor [N]，long，类别
          "bboxes":   Tensor [N, 4]，float32，xyxy 像素坐标
          "im_file":  str，图像原始路径
          "batch_idx":Tensor( index )，int64
          "quality":  Tensor( score )，float32，图像质量分数
        }
    """

    def __init__(self, root: str, split: str = "train", img_size: int = 640):
        self.root = Path(root)
        assert split in ("train", "val", "test"), "split 必须是 'train' / 'val' / 'test'"
        self.split = split
        self.img_size = img_size

        self.img_dir = self.root / split / "images"
        self.label_dir = self.root / split / "labels"
        self.quality_dir = self.root / split / "quality"

        exts = (".jpg", ".jpeg", ".png", ".bmp")
        self.img_files = sorted([p for p in self.img_dir.iterdir() if p.suffix.lower() in exts])
        if len(self.img_files) == 0:
            raise FileNotFoundError(f"{self.img_dir} 下没有图像文件")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index: int):
        # 1. 读取并缩放图像
        img_path = self.img_files[index]
        stem = img_path.stem
        img0 = cv2.imread(str(img_path))  # BGR
        assert img0 is not None, f"无法读取图像: {img_path}"
        h0, w0 = img0.shape[:2]

        img_resized, ratio, pad = resize(img0, (self.img_size, self.img_size), auto=False)
        img_resized = img_resized[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
        img_tensor = torch.from_numpy(img_resized).float() / 255.0

        # 2. 读取分割标签 (“class x1 y1 x2 y2 ...”)
        label_path = self.label_dir / f"{stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"未找到标签: {label_path}")

        with open(label_path) as f:
            lines = [x.strip().split() for x in f.read().splitlines() if x.strip()]

        bboxes = []
        classes = []
        masks = []

        for parts in lines:
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))  # [x1, y1, x2, y2, ...], 归一化
            if len(coords) % 2 != 0:
                raise ValueError(f"{label_path} 中坐标数不是偶数: {coords}")

            # 将归一化多边形点映射到 pad 后图像上 (img_size x img_size)
            poly_pts = []
            it = iter(coords)
            for x_norm, y_norm in zip(it, it):
                x_px = x_norm * w0 * ratio + pad[0]
                y_px = y_norm * h0 * ratio + pad[2]
                poly_pts.extend([x_px, y_px])

            # 生成 mask: segment2mask 接受一个 (K,2) numpy 数组
            coords_arr = np.array(poly_pts, dtype=np.float32).reshape(-1, 2)
            mask_np = segment2mask(coords_arr, (self.img_size, self.img_size))  # uint8, HxW
            masks.append(torch.from_numpy(mask_np).to(torch.uint8))  # [H, W]

            # 计算该多边形的 bounding box (xyxy)
            xs = coords_arr[:, 0]
            ys = coords_arr[:, 1]
            x1, x2 = float(xs.min()), float(xs.max())
            y1, y2 = float(ys.min()), float(ys.max())
            bboxes.append([x1, y1, x2, y2])
            classes.append(cls)

        if masks:
            masks = torch.stack(masks, dim=0)  # [N, H, W]
            bboxes = torch.tensor(bboxes, dtype=torch.float32)   # [N, 4]
            classes = torch.tensor(classes, dtype=torch.long)    # [N]
        else:
            masks = torch.zeros((0, self.img_size, self.img_size), dtype=torch.uint8)
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            classes = torch.zeros((0,), dtype=torch.long)

        # 3. 读取 quality 分数
        quality_path = self.quality_dir / f"{stem}.txt"
        if not quality_path.exists():
            raise FileNotFoundError(f"未找到质量标签: {quality_path}")
        with open(quality_path) as f:
            line = f.read().strip()
        try:
            score = float(line)
        except:
            raise ValueError(f"{quality_path} 内容无法解析为 float: '{line}'")
        quality = torch.tensor(score, dtype=torch.float32)

        # 4. 返回 dict
        return {
            "img": img_tensor,            # [3, img_size, img_size]
            "masks": masks,               # [N, img_size, img_size]
            "cls": classes,               # [N]
            "bboxes": bboxes,             # [N, 4]
            "im_file": str(img_path),     # 原始路径
            "batch_idx": torch.tensor(index, dtype=torch.int64),
            "quality": quality,           # 单标量
        }
