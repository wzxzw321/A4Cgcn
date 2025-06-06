import torch
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class QualityPredictor(SegmentationPredictor):
    """
    在原生 SegmentPredictor 基础上，额外输出质量分数 (quality_score)。
    假设模型的 forward 返回 ((seg_preds, proto), quality_pred)：
      - seg_preds: 由分割 head 输出的预测（保留给父类后处理）；
      - proto: 分割原型，用于 mask 解码；
      - quality_pred: 形状 [B] 的质量回归分数。

    本类重写 postprocess，使其把 quality_pred 拆出并附加到每个 Results 对象上。
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        # 指定任务为 "segment with score"
        self.args.task = "segment with score"

    def postprocess(self, preds, img, orig_imgs):
        """
        重写后处理：
        - preds: 模型前向输出，形如 ((seg_preds, proto), quality_pred)
        - img: 预处理后的输入 tensor，[B, C, H, W]
        - orig_imgs: 原始图像列表或数组

        返回: List[Results]。每个 Results 对象包含原生分割结果，同时带 .quality_score 属性。
        """
        # 1. 解包模型输出
        try:
            (seg_preds, proto), quality_pred = preds
        except Exception:
            raise RuntimeError(
                "QualityPredictor.postprocess: 模型输出应为 ((seg_preds, proto), quality_pred)。"
            )

        # 2. 调用父类后处理分割部分，得到 List[Results]
        #    父类 postprocess 期待 (seg_preds, img, orig_imgs, protos=proto)
        results = super().postprocess((seg_preds, proto), img, orig_imgs)

        # 3. 将 quality_pred ([B]) 对应到每个结果并附加属性
        #    quality_pred 可能是 CPU/GPU Tensor，将其转到 CPU numpy 并转为 float
        if isinstance(quality_pred, torch.Tensor):
            quality_pred = quality_pred.detach().cpu().tolist()
        else:
            # 如果 already a list/ndarray，确保能索引
            quality_pred = list(quality_pred)

        for idx, res in enumerate(results):
            try:
                q = float(quality_pred[idx])
            except Exception:
                q = None
            # 给 Results 对象新增一个属性 quality_score
            setattr(res, "quality_score", q)

        return results
