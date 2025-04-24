from configs.CONST import *
from ultralytics import YOLO

# 加载模型（可选择预训练权重）

model = YOLO("yolov8.yaml")
# model = YOLO("yolov8n.pt")
# model = YOLO("yolov8n-seg.pt")

# 开始训练
results = model.train(
    data="/home/ubuntu/WZX/A4C_GCN/my_traindata.yml",  # 替换为你的data.yaml路径
    epochs=100,                     # 训练轮数
    batch=16,                       # 批次大小（根据GPU显存调整）
    imgsz=640,                      # 输入图像尺寸
    device="0",                     # 使用GPU 0（"cpu"表示CPU）
    optimizer="Adam",               # 优化器
    lr0=0.001,                      # 初始学习率
    name="/home/ubuntu/WZX/A4C_GCN/result",
    save=True,                      # 保存训练结果
)
