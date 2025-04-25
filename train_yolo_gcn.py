from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG
from configs.CONST import *

# 初始化配置
cfg = DEFAULT_CFG.copy()
cfg.update({
    'data': '/home/ubuntu/WZX/A4C_GCN/my_traindata.yml',
    'model': YOLO_GCN_MODEL_PATH,
    'epochs': 100,
    'batch': 16,
    'imgsz': 640,
    'device': 'cuda:0',
    'optimizer' : 'AdamW',
    'lr0' : 0.0005
})

trainer = SegmentationTrainer(overrides=cfg)

class CustomModel(SegmentationModel):
    def __init__(self, cfg, ch=3, nc=None, verbose=True):
        super().__init__(cfg, ch, nc, verbose)
        # 修改模型结构示例：替换主干网络
        self.model.backbone = YourCustomBackbone()  # 替换为你自定义的主干网络


model = CustomModel(cfg='yolov8-seg.yaml', nc=11)  # nc为类别数

trainer.setup_model(model=model)
trainer.setup_train()

# 手动执行训练循环
for epoch in range(cfg.epochs):
    for batch in trainer.train_loader:
        # 前向传播
        preds = model(batch['img'])

        # 计算损失
        loss = model.loss(preds, batch)[0]  # 获取总损失

        # 反向传播
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()

    # 验证流程
    if epoch % cfg.val_interval == 0:
        trainer.validate()