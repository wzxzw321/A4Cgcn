import torch
import torch.nn as nn
from ultralytics.nn.modules import Detect       # 原有分割 head 继承自 Detect
from ultralytics.nn.modules import Segment      # 原有 Segment 类
from ultralytics.nn.modules import Proto        # Proto 模块

class MultiHead(nn.Module):
    """同时输出 segmentation mask 和 quality 分数的多任务 head."""
    def __init__(self, nc=80, nm=32, npr=256, q_hidden=128, ch=()):
        super().__init__()
        # 原始分割 head（保留所有功能）
        self.seg = Segment(nc=nc, nm=nm, npr=npr, ch=ch)

        # Quality head: 从通道维度为 ch[-1]（即32）的特征上做一个小网络
        self.quality = nn.Sequential(
            nn.Conv2d(ch[-1], q_hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),    # 全局池化
            nn.Flatten(),
            nn.Linear(q_hidden, 1)      # 输出单个标量质量分数
        )

    def forward(self, x):
        """
        x: list of 4 张量，对应 model.yaml 里传入的
           [feat17, feat22, feat27, feat32]
        """
        # 1) 分割任务：只取前 3 路特征
        seg_feats = x[:3]
        # 2) 质量任务：取第 4 路特征
        q_feat = x[3]

        # 得到分割输出 (x_seg, mc, proto) 或者导出格式
        seg_out = self.seg(seg_feats)

        # 得到质量分数
        q_out = self.quality(q_feat)  # (batch_size, 1)

        # 训练和推理时，返回值稍有差别，可根据 seg_out 格式拼接/解包
        if self.training:
            # seg_out = (detect_outputs, mc, proto)
            return (*seg_out, q_out)  # tuple: (y, mc, p, q)
        else:
            # 推理时 seg_out 格式可能是 ((y, mc), p) 或者其他，视 export 决定
            # 这里演示最常见的情况：
            y, mc, p = seg_out
            return ( (y, mc), p, q_out )
