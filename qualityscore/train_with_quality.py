import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from seg_with_q import YOLOv8SegWithQuality
from quality_dataset_yolo import QualitySegDatasetYOLO
from configs.CONST import *

# ----------------------- 1. 超参数 -----------------------
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
img_size = 640
batch_size = 4
epochs = 200
learning_rate = 1e-4
alpha = 1.0  # 分割 Loss 权重（已在模型内部使用）
beta = 0.5   # 质量 Loss 权重（已在模型内部使用）

# ----------------------- 2. 数据集 & DataLoader -----------------------
# 这里假设你的训练/验证数据组织如下：
# /home/ubuntu/WZX/A4C_GCN/quality_test/
#   ├─ train/
#   │    ├─ images/      （*.png）
#   │    ├─ labels/      （YOLO segmentation 格式 *.txt）
#   │    └─ quality/     （单个整数 1–5 的 *.txt）
#   └─ val/
#        ├─ images/
#        ├─ labels/
#        └─ quality/

source_dir = '/home/ubuntu/WZX/quality_test'

train_img_dir     = os.path.join(source_dir, 'train/images')
train_labels_dir  = os.path.join(source_dir, 'train/labels')
train_quality_dir = os.path.join(source_dir, 'train/quality')

val_img_dir       = os.path.join(source_dir, 'val/images')
val_labels_dir    = os.path.join(source_dir, 'val/labels')
val_quality_dir   = os.path.join(source_dir, 'val/quality')

# 创建 Dataset
train_dataset = QualitySegDatasetYOLO(
    img_dir     = train_img_dir,
    label_dir   = train_labels_dir,
    quality_dir = train_quality_dir,
    img_size    = img_size,
    transform   = None  # 如果需要数据增强，可传入一个 callable
)
val_dataset = QualitySegDatasetYOLO(
    img_dir     = val_img_dir,
    label_dir   = val_labels_dir,
    quality_dir = val_quality_dir,
    img_size    = img_size,
    transform   = None
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# ----------------------- 3. 初始化模型 & 优化器 -----------------------
# 指向一个预训练或中间训练好的分割模型权重
base_model_yaml = QUALITY_TEST_PATH
model = YOLOv8SegWithQuality(
    base_weights_path=base_model_yaml,  # 指向 .yaml 而不是 .pt
    num_quality_hidden=256
)
model = model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# ----------------------- 4. 训练主循环 -----------------------
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    running_seg_loss = 0.0
    running_quality_loss = 0.0

    for imgs, masks, qualities in train_loader:
        imgs = imgs.to(device)             # [B,3,H,W], float32
        masks = masks.to(device)           # [B,1,H,W], long (0/1)
        qualities = qualities.to(device)   # [B,1], float32

        outputs = model(imgs, seg_targets=masks, quality_targets=qualities)
        loss = outputs['loss']
        loss_seg = outputs['loss_seg']
        loss_quality = outputs['loss_quality']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bsz = imgs.size(0)
        running_loss += loss.item() * bsz
        running_seg_loss += loss_seg.item() * bsz
        running_quality_loss += loss_quality.item() * bsz

    epoch_loss = running_loss / len(train_dataset)
    epoch_seg = running_seg_loss / len(train_dataset)
    epoch_qual = running_quality_loss / len(train_dataset)

    print(f"Epoch [{epoch+1}/{epochs}]  "
          f"Total_Loss: {epoch_loss:.4f}  "
          f"Seg_Loss: {epoch_seg:.4f}  "
          f"Qual_Loss: {epoch_qual:.4f}")

    scheduler.step()

    # 每 5 个 epoch 或最后一个 epoch 进行一次验证
    if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
        model.eval()
        val_loss = 0.0
        val_seg_loss = 0.0
        val_quality_loss = 0.0

        with torch.no_grad():
            for imgs, masks, qualities in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                qualities = qualities.to(device)

                outputs = model(imgs, seg_targets=masks, quality_targets=qualities)
                loss = outputs['loss']
                loss_seg = outputs['loss_seg']
                loss_quality = outputs['loss_quality']

                bsz = imgs.size(0)
                val_loss += loss.item() * bsz
                val_seg_loss += loss_seg.item() * bsz
                val_quality_loss += loss_quality.item() * bsz

        val_loss /= len(val_dataset)
        val_seg_loss /= len(val_dataset)
        val_quality_loss /= len(val_dataset)
        print(f"—— 验证 —— Epoch [{epoch+1}/{epochs}]  "
              f"Val_Loss: {val_loss:.4f}  "
              f"Val_Seg: {val_seg_loss:.4f}  "
              f"Val_Qual: {val_quality_loss:.4f}")

        # 保存当前 epoch 的模型权重
        save_dir = '/home/ubuntu/WZX/A4C_GCN/qualityscore/results_seg_quality/'
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            model.state_dict(),
            os.path.join(save_dir, f'epoch_{epoch+1}.pth')
        )

print("训练完毕！")
