from configs.CONST import *
from ultralytics import YOLO

model = YOLO(model=YOLO_GCN_MODEL_PATH, task="segment")

results = model.train(
    data="/home/ubuntu/WZX/A4C_GCN/my_traindata.yml",  # 替换为你的data.yaml路径
    epochs=150,                     # 训练轮数
    batch=16,                       # 批次大小（根据GPU显存调整）
    imgsz=640,                      # 输入图像尺寸
    device="0",                     # 使用GPU 0（"cpu"表示CPU）
    optimizer="AdamW",               # 优化器
    lr0=0.0005,                     # 初始学习率
    name="/home/ubuntu/WZX/A4C_GCN/results/result",
    save=True,                      # 保存训练结果
)
