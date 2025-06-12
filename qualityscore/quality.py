import torch
import torch.nn as nn
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.loss import v8SegmentationLoss

class QualityHead(nn.Module):
    """
    全局回归分支：
    - 输入: 某一个尺度的特征图 [B, C, H, W]；
    - 结构: 全局平均池化 → FC → BN → ReLU → FC → 回归一个标量。
    """
    def __init__(self, in_channels: int, hidden_dim: int = 128, out_dim: int = 1):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))    # [B, C, 1, 1] → view 为 [B, C]
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.global_pool(x).view(B, C)       # [B, C]
        x = self.fc1(x)                          # [B, hidden_dim]
        x = self.bn1(x)
        x = self.relu(x)
        quality = self.fc2(x)                    # [B, out_dim]
        return quality.view(B)                   # 如果 out_dim=1 → [B]


class SegmentationModelWithQuality(SegmentationModel):
    def __init__(self,
                 cfg: str = "yolo8g-seg-quality.yaml",
                 ch: int = 3,
                 nc: int = None,
                 verbose: bool = True):
        # ——— 第一步：在父类 __init__ 阶段，不启用 QualityHead，保证 stride 推断正确 ———
        self._quality_initialized = False
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

        # ——— 第二步：父类构造完毕后（此时已经跑过一次 forward(zeros) 来推断 stride），
        #            再根据你贴出的模型结构，“索引 32 层输出通道数”是 256，就直接写死 256。
        neck_out_channels = 256
        self.quality_head = QualityHead(in_channels=neck_out_channels, hidden_dim=128, out_dim=1)
        self._initialize_quality_head()

        # ——— 第三步：标记 QualityHead 已初始化，后续真正 forward 时要启用它 ———
        self._quality_initialized = True

    def _initialize_quality_head(self):
        """
        用 Kaiming 初始化 QualityHead 的全连接层，BN 层保持默认初始化。
        """
        for m in self.quality_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self,
                x: torch.Tensor,
                augment: bool = False,
                visualize: bool = False) -> tuple:
        # ——— 父类 __init__ 阶段的那次 forward(zeros)（用于自动推断 stride），QualityHead 还没初始化，直接走父类逻辑 ———
        if not self._quality_initialized:
            return super().forward(x, augment=augment, visualize=visualize)

        # ——— QualityHead 已就绪，下面演示“用 forward hook 截第 32 层输出，然后跑完整分割前向” ———

        # 1. 定义一个容器来存第 32 层的输出
        feat32_holder = {}

        # 2. 写一个 hook：把模块输出存到 feat32_holder['feat'] 里
        def hook_fn(module, input, output):
            feat32_holder['feat'] = output

        # 3. 给 self.model.model[32] 注册 forward hook
        #    注意：你的模型结构里，索引 32 对应的正是 “- [-1, 3, C2f, [256]]  # 32” 这一层。
        handle = self.model.model[32].register_forward_hook(hook_fn)

        # 4. 真正调用父类的 forward（这会顺序地把 x 传进整个 Sequential 网络）
        seg_out = super().forward(x, augment=augment, visualize=visualize)

        # 5. hook 用完之后，要及时注销，否则下一次前向还会把 feat32_holder 覆盖
        handle.remove()

        # 6. 从 holder 里拿到第 32 层输出
        feat32 = feat32_holder.get('feat', None)
        if feat32 is None:
            # 万一没挂到或者出错，用一个显式报错帮助定位
            raise RuntimeError("前向钩子没有捕获到第 32 层输出。请确保 self.model.model[32] 真的是 C2f([256]) 这一层。")

        # 7. 用 feat32 喂给 QualityHead，得到 [B] 形状的质量分数
        #    （假设 batch_size = B）
        quality_pred = self.quality_head(feat32).view(x.shape[0])  # → [B]

        # 8. 把 seg_out 解包成 feats, pred_masks, proto
        if isinstance(seg_out, tuple) and len(seg_out) == 3 and isinstance(seg_out[0], list):
            feats, pred_masks, proto = seg_out
        else:
            # 训练模式下，seg_out 通常是 (loss, (feats, pred_masks, proto), ...)
            _, inner = seg_out
            feats, pred_masks, proto = inner

        # 9. 最终把“(feats, pred_masks, proto)”和 “quality_pred” 打包返回
        return (feats, pred_masks, proto), quality_pred

    def init_criterion(self):
        # 训练时只用原生分割 loss，QualityHead 的回归 loss 可以在外部 Trainer 里自行加
        return v8SegmentationLoss(self)
