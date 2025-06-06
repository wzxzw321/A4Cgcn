import torch
import torch.nn.functional as F
from gcncode_phis import ARGRModule

# ==============================
# 1. 参数设置
# ==============================
# 假设我们想要测试的特征图维度是：
B, C, H, W = 2, 16, 8, 8
#   B: batch size = 2
#   C: 输入通道数 = 16（与 ARGRModule 构造时的 in_channels 保持一致）
#   H = W = 8（特征图高宽都设为 8，纯粹演示用）

# ==============================
# 2. 构造随机输入与目标
# ==============================
# 随机输入 tensor，形状 [B, C, H, W]
feat_map = torch.randn(B, C, H, W)

# 随机 target（比如我们用 4D 回归损失来示范），形状也 [B, C, H, W]
target = torch.randn(B, C, H, W)

# ==============================
# 3. 初始化 ARGRModule
# ==============================
# 由于输入通道数是 C=16，所以 in_channels 也传 16
model = ARGRModule(in_channels=C, num_heads=3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ==============================
# 4. 前向 + 反向 + 参数更新
# ==============================
num_epochs = 2000
for epoch in range(1, num_epochs+1):
    model.train()
    optimizer.zero_grad()
    out_map = model(feat_map)
    loss = F.mse_loss(out_map, target)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:>3d}  loss = {loss.item():.4f}")

