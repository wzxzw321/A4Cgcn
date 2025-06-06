import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.tasks import DetectionModel,SegmentationModel
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
        # out_dim=1 时, 直接回归; 若 out_dim>1，可用于分类并配合 CrossEntropyLoss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        return: [B], 质量分数
        """
        B, C, H, W = x.shape
        # 全局平均池化: [B, C, 1, 1] → view 为 [B, C]
        x = self.global_pool(x).view(B, C)
        x = self.fc1(x)               # [B, hidden_dim]
        x = self.bn1(x)
        x = self.relu(x)
        quality = self.fc2(x)         # [B, out_dim]
        return quality.view(B)        # 如果 out_dim=1, return [B]


class SegmentationModelWithQuality(SegmentationModel):
    """
    在原生 SegmentationModel 基础上，增加一个 QualityHead 分支。
    - 继承自 SegmentationModel(cfg, ch, nc, verbose)；
    - 在 __init__ 中构造质量分支；
    - 在 forward 中，调用父类 forward 获取常规分割输出 (feats, pred_masks, proto)，
      同时从 feats 中选取一个尺度(feature map)喂给 quality_head，得到 [B] 的质量分数。
    """
    def __init__(self,
                 cfg: str = "yolo8g-seg-gcn.yaml",
                 ch: int = 3,
                 nc: int = None,
                 verbose: bool = True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

        # ---------- Step 1: 确定要喂给 QualityHead 的特征图通道数 ----------
        # SegmentationModel.forward(x) 返回的 preds 中，第一项 feats 是一个 List[Tensor]，
        # 每个 Tensor 对应不同下采样尺度的特征 (如 strides = [8,16,32] 对应的 feature maps)。
        # 我们通常挑选较「高层」的那张尺度较小、语义丰富的特征图，例如 feats[-1]。
        #
        # 由于具体实现里 feats 的顺序、通道数由 model.yaml 决定，
        # 这里以「取最后一个 feat」为例去推断通道数：
        #
        #    1. 用一张 dummy tensor 过一次 backbone+neck，拿到 feats list。
        #    2. 取 feats[-1].shape[1] 作为 in_channels。
        #
        # 为了避免在 __init__ 中实际跑一次前向，我们可以通过简单方式：先假设
        # feats[-1].shape[1] 等于 self.model.model[-3].cv2.out_channels —— 出自对 Ultralytics v8 源码的常见观察。
        # （如果你本地版本里顺序不同，请自行打印 self.model.model 的结构确认索引。）
        #
        # 下面演示获取 in_channels 的示例写法：
        try:
            # 常见 YOLOv8: self.model.model 是一个 nn.Sequential 列表，
            # 倒数第三层通常是 neck 输出 conv；它有属性 .cv2.out_channels。
            neck_out_channels = self.model.model[-2].cv2.out_channels
        except Exception:
            # 如果索引不对，请打印 self.model.model，找到「neck 最后输出层」的通道数，再手动替换以下数字。
            raise RuntimeError(
                "无法自动推断 neck_out_channels。请检查 self.model.model 的结构，确认哪个层的 out_channels 对应 neck 输出。"
            )

        # ---------- Step 2: 构造 QualityHead ----------
        self.quality_head = QualityHead(in_channels=neck_out_channels, hidden_dim=128, out_dim=1)
        # 可选：使用 Kaiming 初始化 QualityHead
        self._initialize_quality_head()

    def _initialize_quality_head(self):
        """
        使用 Kaiming 正态初始化 QualityHead 的全连接层，BN 层保持默认初始化。
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
        """
        重写 forward：
        - x: [B, ch, H, W]
        - 返回: ((feats, pred_masks, proto), quality_pred)
          其中 (feats, pred_masks, proto) 用于原生的 segmentation loss / postprocess，
          quality_pred 是 [B] 的质量回归分数。

        注意：
        - 若只想要 seg 输出，可从返回值中取前半部分；
        - 若想要质量分数，则取第二项 quality_pred。
        """
        # ------------------------------------------
        # 1. 调用父类 SegmentationModel.forward，得到 segmentation 相关 preds
        #    其返回值 preds 形如 (feats, pred_masks, proto)，满足 v8SegmentationLoss 所需格式。
        #    有时 parent.forward 返回的是一个更复杂的结构 (如 (tuple_of_feats, …), proto)；
        #    这里假设 len(preds) == 3，对应 feats、pred_masks、proto。
        # ------------------------------------------
        preds = super().forward(x, augment=augment, visualize=visualize)
        # preds 里:
        #   feats: List[Tensor]，例如 [[B, C1, H1, W1], [B, C2, H2, W2], [B, C3, H3, W3]]
        #   pred_masks: [B, N_anchors, 32]（或等价形状）
        #   proto: [B, 32, mask_h, mask_w]
        if isinstance(preds, tuple) and len(preds) == 3:
            feats, pred_masks, proto = preds
        else:
            # 如果 parent.forward 返回更深一层的嵌套，可以根据实际 unpack
            # 例如: preds = (None, (feats, pred_masks, proto)) → unpack 为 preds[1]
            try:
                _, (feats, pred_masks, proto) = preds
            except Exception:
                raise RuntimeError("无法解析 SegmentationModel.forward 的返回值，请检查父类实现。")

        # ------------------------------------------
        # 2. 从 feats 中挑一个尺度喂给 QualityHead
        #    这里我们选用 feats[-1]（即下采样最大、语义最丰富的特征图）
        # ------------------------------------------
        feature_for_quality = feats[-1]  # Tensor shape: [B, C_q, Hq, Wq]
        quality_pred = self.quality_head(feature_for_quality)  # [B]

        # ------------------------------------------
        # 3. 返回 (preds, quality_pred)
        #    上层 Trainer/Validator/Predicate 会根据 TASKS 映射去使用 quality_pred
        # ------------------------------------------
        return (feats, pred_masks, proto), quality_pred

    def init_criterion(self):
        """
        覆盖父类 init_criterion，将原 v8SegmentationLoss 包装到一个自定义的 Loss 中，
        以便在计算 loss 时同时用到 quality_pred 和 quality_gt。
        这里我们暂时返回原始的 seg loss；具体 quality loss 应在 Trainer 中完成叠加。
        """
        return v8SegmentationLoss(self)
