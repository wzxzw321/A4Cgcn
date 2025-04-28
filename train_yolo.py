import random
import numpy as np
import torch
from configs.CONST import *
from engine import YOLO
from torch.backends import cudnn
# cudnn.deterministic = True

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)  # set random seed for CPU
torch.cuda.manual_seed(SEED)  # set random seed for one GPU
torch.cuda.manual_seed_all(SEED)  # set random seed for all GPU
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


model = YOLO(model=YOLO_GCN_MODEL_PATH, task="segment")

results = model.train(
    data="/home/ubuntu/WZX/A4C_GCN/my_traindata.yml",  # 替换为你的data.yaml路径
    epochs=150,                     # 训练轮数
    batch=16,                       # 批次大小（根据GPU显存调整）
    imgsz=640,                      # 输入图像尺寸
    device="0",                     # 使用GPU 0（"cpu"表示CPU）
    optimizer="AdamW",              # 优化器
    lr0=0.0005,                     # 初始学习率
    name="/home/ubuntu/WZX/A4C_GCN/results/result",
    rect=True,                      # 矩形训练
    # shuffle=False,                  # 不打乱数据集
    val=True,                       # 在每个epoch结束时进行验证
    close_mosaic=10,                # 关闭mosaic数据增强的epoch
    mosaic=0.8,                     # mosaic数据增强的概率
    save=True,                      # 保存训练结果
)
