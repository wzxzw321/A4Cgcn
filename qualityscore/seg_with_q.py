import torch
import torch.nn as nn
from ultralytics import YOLO

class YOLOv8SegWithQuality(nn.Module):
    """
    在 Ultralytics YOLOv8-seg 模型后，拼接一个“从第 32 层抽特征 → 1×1 Conv → ReLU → GAP → FC”分支，
    用于预测质量分数（回归到 [1]）。

    - base_weights_path: 你事先训练好的分割权重，比如 'yolov8s-seg.pt' 或 自己的 best.pt
    - num_quality_hidden: 1×1 Conv 之后的中间通道数（这里示例用 256）
    """

    def __init__(self, base_weights_path, num_quality_hidden=256):
        super().__init__()
        # ——————————————————————
        # 1) 加载 Ultralytics YOLOv8-seg （这里自动把 YAML + 权重都加载进来）
        #    seg_model.model 是 DetectSeg 类，它内部有一个 .model 属性存着具体的 ModuleList
        #    .model.model[idx] 就是我们 YAML 里第 idx 行对应的 submodule
        self.seg_model = YOLO(base_weights_path, task='segment')
        # DetectSeg 的底层 nn.ModuleList 存在于：
        #    self.seg_model.model.model
        # （注意：seg_model.model 指向 DetectSeg；seg_model.model.model 才是 ModuleList）
        self._layers = self.seg_model.model.model  # ModuleList 中保存了所有逐层构建的子模块

        # ——————————————————————
        # 2) 找到“第 32 层”对应的子模块，以便拿它的输出通道数 c_feat
        #    在你贴的 YAML 里，Index 从 0 开始数到 32 对应的正好就是：
        #      #32: [-1, 3, C2f, [256]]  <-- 这是我们想要的“P3/P4/P5 上采样后融合”输出
        #    在 ModuleList 中，正好 .model[32] 就是那个 C2f 层。
        #
        feat_idx = 32
        feat_layer = self._layers[feat_idx]  # 理论上是一个 C2f 模块，输出通道数应该是 256
        # 根据不同版 Ultralytics 源码，C2f 里可能有 .cv1/.cv2/.conv 等属性，下面尝试几种常见写法：
        if hasattr(feat_layer, 'cv2') and hasattr(feat_layer.cv2, 'conv'):
            c_feat = feat_layer.cv2.conv.out_channels
        elif hasattr(feat_layer, 'conv'):
            c_feat = feat_layer.conv.out_channels
        else:
            raise ValueError(f"无法从第 {feat_idx} 层模块获取输出通道数，请打印 feat_layer 检查结构：\n{feat_layer}")

        # ——————————————————————
        # 3) 根据 c_feat 构造质量分支：1×1 Conv -> ReLU -> GAP -> FC -> [B,1]
        self.quality_conv = nn.Conv2d(
            in_channels=c_feat,
            out_channels=num_quality_hidden,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化到 1×1
        self.quality_fc = nn.Linear(num_quality_hidden, 1)

        # 初始化质量分支参数
        nn.init.kaiming_normal_(self.quality_conv.weight, mode='fan_out', nonlinearity='relu')
        if self.quality_conv.bias is not None:
            nn.init.constant_(self.quality_conv.bias, 0.0)
        nn.init.xavier_uniform_(self.quality_fc.weight)
        nn.init.constant_(self.quality_fc.bias, 0.0)

        # 换一个名字，方便后面调用
        self.feat_idx = feat_idx
        # Segment Head 最后一个模块的 index（不同版本 Ultralytics 可能略微有差别，但通常 Segment 是最后一条）
        # 这里我们下面会直接调用 seg_model(...) 来算分割 loss，所以不需要手动拆出 seg_head。
    def train(self, mode: bool = True):
        self.training = mode  # 先给自己打标志

        # —— 不要调用 self.seg_model.train() ——
        # 而是对子模块 seg_model.model（DetectSeg 底层 PyTorch ModuleList）调用 .train(mode)
        self.seg_model.model.train(mode)

        # —— 再对子类里除了 seg_model 以外的子模块调用 .train(mode) ——
        # 这些子模块主要是 quality_conv、quality_fc 等
        for name, module in self.named_children():
            if name == 'seg_model':
                continue
            # 这会对 quality_conv / relu / pool / quality_fc 依次调用 train(mode)
            module.train(mode)

        return self
    def forward(self, imgs, seg_targets=None, quality_targets=None):
        """
        imgs: Tensor [B,3,H,W]
        seg_targets: None 或 [B,1,H_seg,W_seg]（0/1 mask）—— 如果训练，需要传入真实分割 mask
        quality_targets: None 或 [B,1]（浮点，1–5 之间）—— 如果训练，需要传入真实质量分数

        返回：
          - 训练模式 (self.training==True) 下，返回 dict，包括：
              {
                'loss': total_loss,
                'loss_seg': loss_seg.detach(),
                'loss_quality': loss_quality.detach(),
                'pred_quality': pred_q.detach()  # 方便调试
              }
          - 推理/验证模式 (self.training==False) 下，返回：
              seg_mask_pred, pred_q
        """
        # ——————————————————————————————————————————————————————
        # 1) 先让 Ultralytics seg_model 自动跑一次“完整的分割前向 + 分割 Loss”
        #    注意：当你传入 masks=seg_targets 时，seg_model(...) 会返回一个 Results 对象，里面有 .loss、.masks、.boxes 等属性。
        #    例如：res = seg_model(imgs, masks=seg_targets)
        #            loss_seg = res.loss
        #            seg_preds = res.masks  # 这是经过 sigmoid 之后的 mask 概率
        #
        #    如果你只传 imgs 而不传 masks，seg_model(imgs) 则只做推理，返回 .masks 但 loss=0。
        #
        #    这里我们先调用一次以获得分割结果 + loss_seg：
        # if self.training:
        #     # 训练时传入 seg_targets，Ultralytics 会自动在内部 reshape、计算 loss
        #     res = self.seg_model(imgs, masks=seg_targets)
        #     loss_seg = res.loss  # Ultralytics 里已经包含了 BCE+Dice 等所有分割 loss
        #     # 需要保留 seg_preds? 如果要在画图或调试时可视化，可以用：
        #     # seg_preds = res.masks  # shape [B,1,H_seg,W_seg]，值在 [0,1]
        # 1. 获取分割损失
        seg_model = self.seg_model.model  # 获取底层模型（DetectSeg）

        if self.training:
            # 直接调用 DetectSeg 的 forward 函数，传入图像和目标
            outputs = seg_model(imgs, seg_targets)
            loss_seg = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
        else:
            # 推理/验证时，单纯得到分割预测
            res = self.seg_model(imgs)
            loss_seg = torch.tensor(0.0, device=imgs.device)
            # 典型地，res.masks 是 List[Tensor] 或 Tensor，视 Ultralytics 版本而定
            # 假设 res.masks 直接就是 [B,1,H_seg,W_seg]
            seg_preds = res.masks

        # ——————————————————————————————————————————————————————
        # 2) 单独做一路“到第 32 层”的前向，拿到 feats 供质量头使用
        #    Ultralytics 底层的 DetectSeg.forward 并没有暴露“中间层输出”，
        #    所以这里绕一圈：直接按顺序调用 ModuleList 里第 0..feat_idx（共 feat_idx+1 个模块）来手动前向一次。
        #
        #    “手动跑一次”其实和上面 seg_model(imgs) 重复了 backbone+neck+head[0..32] 的计算，
        #    导致多了一次开销。若显存/速度敏感，可以考虑改写 DetectSeg.forward，
        #    但为了示例，这里先直接双路做：一次给 seg_model 算 loss，一次手动跑到第 32 层拿特征。
        #
        x = imgs
        for i, layer in enumerate(self._layers[: self.feat_idx + 1]):  # 0..32 层（共 33 层）
            x = layer(x)
        feats = x  # [B, c_feat, H_feat, W_feat]  其中 c_feat=256

        # 质量分支：1×1 Conv -> ReLU -> GAP -> FC -> [B,1]
        q = self.quality_conv(feats)  # [B, num_quality_hidden, H_feat, W_feat]
        q = self.relu(q)
        q = self.pool(q).flatten(1)  # [B, num_quality_hidden]
        pred_q = self.quality_fc(q)  # [B,1]  —— 这就是网络对质量的预测值

        # ——————————————————————————————————————————————————————
        # 3) 如果是训练阶段，则再计算质量 Loss（MSE 或 L1），并与分割 Loss 合并
        if self.training:
            if quality_targets is None:
                raise ValueError("训练时必须同时传入 seg_targets 和 quality_targets")
            # MSE Loss（也可换成 L1Loss、SmoothL1 等）
            mse_fn = nn.MSELoss()
            loss_quality = mse_fn(pred_q, quality_targets)

            # 举例统一加权：分割 Loss 权重 α=1.0，质量 Loss 权重 β=0.5
            alpha = 1.0
            beta = 0.5
            total_loss = alpha * loss_seg + beta * loss_quality

            return {
                'loss': total_loss,
                'loss_seg': loss_seg.detach(),
                'loss_quality': loss_quality.detach(),
                'pred_quality': pred_q.detach()
            }

        else:
            # 推理或验证模式：直接返回分割预测 + 质量预测
            # seg_preds 从 res.masks 来；pred_q 形状 [B,1]
            return seg_preds, pred_q

# ——————————————————————————————————————————————————————————————————————————————————
# 使用示例：
#
# from seg_with_quality import YOLOv8SegWithQuality
# model = YOLOv8SegWithQuality(base_weights_path='best_seg_model.pt', num_quality_hidden=256)
# model = model.to(device)
#
# # 训练时：
# imgs = torch.randn(4,3,640,640).to(device)
# seg_gt = torch.randint(0,2,(4,1,160,160)).to(device)  # 假设 seg head 下采样到 1/4 → 160×160
# qual_gt = torch.tensor([[3.0],[4.0],[2.0],[5.0]],dtype=torch.float32).to(device)
#
# outputs = model(imgs, seg_targets=seg_gt, quality_targets=qual_gt)
# # outputs['loss'], outputs['loss_seg'], outputs['loss_quality'], outputs['pred_quality']
#
# # 推理时：
# model.eval()
# with torch.no_grad():
#     seg_pred, q_pred = model(imgs)
#     # seg_pred: [B,1, H_out, W_out]  分割 mask 概率； q_pred: [B,1] 质量分数
#
# ——————————————————————————————————————————————————————————————————————————————————
