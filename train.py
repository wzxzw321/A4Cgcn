from configs import CONST
# from ultralytics import YOLO

# # 加载模型（可选择预训练权重）
# model = YOLO("yolov8n.pt")  # 从官方预训练模型开始
# # 或 model = YOLO("yolov8.yaml")  # 从零开始训练
#
# # 开始训练
# results = model.train(
#     data="path/to/your/data.yaml",  # 替换为你的data.yaml路径
#     epochs=100,                     # 训练轮数
#     batch=16,                       # 批次大小（根据GPU显存调整）
#     imgsz=640,                      # 输入图像尺寸
#     device="0",                     # 使用GPU 0（"cpu"表示CPU）
#     name="a4c_model",               # 实验名称（可选）
#     save=True,                      # 保存训练结果
# )
