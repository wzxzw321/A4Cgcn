# -*- coding: utf-8 -*-
# using yolov8-04 for prod

from ultralytics.ultralytics import YOLO
# import matplotlib.pyplot as plt
#
# # 设置字体为支持中文的字体
# plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题


# Load a model
model = YOLO("yolov8n-seg.yaml")  # build a new model from YAML

# # Train the model
# results = model.train(data="/home/ygz/ZC/HUS/HUS_ImgSeg/0_InsSeg_yolov8/my_traindata.yml", epochs=50, imgsz=640)

model = YOLO("yolov8n-seg.yaml")
results = model.train(
    data="/home/ygz/ZC/HUS/HUS_ImgSeg/0_InsSeg_yolov8/my_traindata.yml",
    epochs=100,
    imgsz=640,
    lr0=0.001,  # 降低初始学习率
    optimizer='AdamW',  # 明确指定优化器
)