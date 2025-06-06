import random
import numpy as np
import torch
from configs.CONST import *
from engine import YOLO
from torch.backends import cudnn
import os
from pathlib import Path
import logging
import warnings
class WarningFilter(logging.Filter):
    def filter(self, record):
        # 屏蔽包含 "Limiting validation plots"以及"Gradients will be None" 的警告
        return ("Limiting validation plots" not in record.getMessage())
logger = logging.getLogger('ultralytics')
logger.addFilter(WarningFilter())
warnings.filterwarnings("ignore","None of the inputs have requires_grad=True. Gradients will be None")
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)  # set random seed for CPU
torch.cuda.manual_seed(SEED)  # set random seed for one GPU
torch.cuda.manual_seed_all(SEED)  # set random seed for all GPU
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)

model = YOLO(model=YOLO_GCN_MODEL_PATH, task="segment with score")

results = model.train(
    data="/home/ubuntu/WZX/A4C_GCN/my_traindata.yml",  # 替换为你的data.yml路径
    epochs=220,                     # 训练轮数
    batch=8,                        # 批次大小（根据GPU显存调整）
    imgsz=640,                      # 输入图像尺寸
    device="0",                     # 使用GPU
    optimizer="AdamW",              # 优化器
    lr0=0.0005,                     # 初始学习率
    name="/home/ubuntu/WZX/A4C_GCN/results/result",
    rect=True,                      # 矩形训练
    # shuffle=False,                # rect=true时默认不打乱数据集
    val=True,                       # 在每个epoch结束时进行验证
    close_mosaic=11,                # 关闭mosaic数据增强的epoch
    mosaic=0.8,                     # mosaic数据增强的概率
    save=True,                      # 保存训练结果
    amp=False,                      # 不使用混合精度训练
)

train_root: Path = Path(results.save_dir)
weights_dir: Path = train_root / "weights"

best_pt = weights_dir / "best.pt"
last_pt = weights_dir / "last.pt"

if not best_pt.exists():
    raise FileNotFoundError(f"best.pt not found in {best_pt}")
if not last_pt.exists():
    raise FileNotFoundError(f"last.pt not found in {last_pt}")

print(f"\nbest.pt found: {best_pt}")
print(f"last.pt found: {last_pt}\n")

for mode,weight_path in [("BEST",best_pt)]:
    print(f"Evaluating {mode} set...")
    metrics = YOLO(str(weight_path)).val(
        data="/home/ubuntu/WZX/A4C_GCN/my_traindata.yml",
        split="test",)
    print(f"====={mode} finished=====\n")

print("Done")