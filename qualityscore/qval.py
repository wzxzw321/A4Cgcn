import torch
import torch.nn.functional as F
from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, NUM_THREADS, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import SegmentMetrics, box_iou, mask_iou
from ultralytics.utils.plotting import output_to_target, plot_images

from qualityscore.quality import SegmentationModelWithQuality  # 替换为实际路径


class QualityValidator(DetectionValidator):
    """
    在原生 SegmentationValidator 基础上，新增“质量回归”评估。
    - 接收模型返回 ((seg_feats, pred_masks, proto), quality_pred)；
    - 保留原有分割评估逻辑（mAP, mIoU 等），同时记录 quality_pred 与 quality_gt，
      在 finalize_metrics 时计算 MAE 和 RMSE 并写入最终指标。
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        # 强制使用 segment with score 任务
        self.args.task = "segment with score"
        # 原生只创建 SegmentMetrics，后续 finalize_metrics 会把 speed, confusion_matrix 填入
        self.metrics = SegmentMetrics(save_dir=self.save_dir)
        # 新增存放质量预测与 ground truth 的列表
        self.quality_preds = []
        self.quality_gts = []

    def preprocess(self, batch):
        """
        1. 调用父类预处理（包括 images → device, targets → device 等）；
        2. 把 batch["masks"] 转为 float 并送到 device；
        3. 把 batch["qualityscore"] 也送到 device，方便后续在 update_metrics 中直接使用。
        """
        batch = super().preprocess(batch)
        batch["masks"] = batch["masks"].to(self.device).float()
        if "qualityscore" in batch:
            batch["qualityscore"] = batch["qualityscore"].to(self.device).float()
        else:
            raise KeyError("QualityValidator 需要 batch 中包含 'qualityscore' 标签。")
        return batch

    def init_metrics(self, model):
        """
        初始化度量。与原生 SegmentationValidator 保持一致，只是额外清空 qualityscore 列表。
        """
        super().init_metrics(model)
        self.plot_masks = []
        if self.args.save_json:
            check_requirements("pycocotools>=2.0.6")
        self.process = (
            ops.process_mask_native
            if self.args.save_json or self.args.save_txt
            else ops.process_mask
        )
        self.stats = dict(tp_m=[], tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

        # 清空质量列表以便重新累积
        self.quality_preds.clear()
        self.quality_gts.clear()

    def postprocess(self, preds):
        """
        覆盖原生 postprocess，将模型输出拆成两部分：
          - seg_outputs: (predn, proto)，供原生后续评估使用
          - quality_preds: [B] 的张量
        输入 preds 形如 ((feats, pred_masks, proto), quality_pred)。
        原生 expects preds to be (seg_preds, proto), 其中 seg_preds 会传给 super().postprocess。
        """
        # preds_from_model: ((feats, pred_masks, proto), quality_pred)
        try:
            seg_tuple, quality_pred = preds
            feats, pred_masks, proto = seg_tuple
        except Exception:
            raise RuntimeError(
                "QualityValidator.postprocess: 模型输出格式应为 ((feats, pred_masks, proto), quality_pred)。"
            )

        # 先用父类 postprocess 处理分割部分：super().postprocess 需要输入 (preds_cls_xyxy_conf_cls, proto)
        # 但实际上 DetectionValidator.postprocess 接收的 preds[0] 是 det_preds，这里 seg_preds 等价为 dp[0]
        # 因此我们传入 predn, proto 给父类。为了不改动父类接口，我们手动构造父类期望的 preds_for_super：
        preds_for_super = (preds[0], preds[1])  # 注意：preds_for_super 将被解成 (predn, proto)
        # 由于我们并不关心 feats 和 pred_masks 在这里，只要 proto 即可
        # 但 DetectionValidator.postprocess 的定义是: p = super().postprocess(preds[0]); proto = preds[1][-1] 或 preds[1]
        # 故直接：
        _, proto_for_super = feats, proto
        # “predn” 其实就是 preds_cls_xyxy_conf_cls。我们在 Segmentation 任务里，“pred” 是：前向后 postprocess 返回的内容。
        # 为简便起见，调用父类时：让父类把 seg_preds(即 feats,pred_masks,proto) 做完整后处理：
        seg_preds = (feats, pred_masks, proto)
        # 但父类 postprocess 只需要 (predn, proto)。为了不深改，我们直接把 (predn, proto) 当作 preds_for_super。
        # 这里 predn，是 DetectionValidator 前面在 validate_step 中由 model(images)→postprocess 得到的结果。
        # 这一层写法只能保证结构一致；具体若出错请对齐 Ultralytics 版本。



        # 采用最直接的方式：让父类当作“分割任务”去后处理，这里忽略 quality_pred：
        p = super().postprocess((preds[0], preds[1]))  # 即 super().postprocess((seg_preds, proto))
        # 但 preds[0] 应为 predn，preds[1] 应为 proto。如果有兼容性问题，请根据本地源码调整。

        # 将 quality_pred 暂存以便后续 update_metrics 使用
        # 在 postprocess 中无法访问 batch，但我们可以把 quality_pred 临时赋给实例变量，update_metrics 再取
        self._last_quality = quality_pred.detach().cpu()

        return p, proto

    def update_metrics(self, preds, batch):
        """
        1. preds: 来自 postprocess，形如 (predn, proto)；
        2. batch: 包含 'cls', 'bboxes', 'masks', 以及 'qualityscore'；
        3. 先调用父类原有分割评估逻辑；
        4. 然后把 self._last_quality（[B]）与 batch['qualityscore'] 累积到列表。
        """
        # 先执行原生分割更新
        super().update_metrics(preds, batch)

        # 读取该 batch 的质量预测与 ground truth
        # 由于 DetectionValidator.update_metrics 是针对每个 “图像” 逐一调用，
        # 并且 postprocess 存储了单张图的 quality_pred，这里直接取 self._last_quality[si]
        # 但 self._last_quality 是 [B]，要跟 batch 顺序一一对应。
        quality_batch = self._last_quality.to(self.device)  # [B]
        quality_gt_batch = batch["qualityscore"]  # [B]

        # 累积到列表，后续 finalize_metrics 统一计算
        self.quality_preds.append(quality_batch.cpu())
        self.quality_gts.append(quality_gt_batch.cpu())

    def finalize_metrics(self, *args, **kwargs):
        """
        1. 先调用父类 finalize_metrics，将 speed/confusion_matrix 写入 self.metrics；
        2. 然后把采集到的 quality_preds/quality_gts 拼成大 Tensor，计算 MAE、RMSE；
        3. 将 'quality_mae'、'quality_rmse' 写入 self.metrics.stats 或者直接打印/日志输出。
        """
        # 调用父类，让其把 speed、confusion_matrix 等写入 self.metrics
        super().finalize_metrics(*args, **kwargs)

        # 把所有 batch 的 qualityscore 预测/GT 拼接
        preds = torch.cat(self.quality_preds, dim=0)  # [N_val]
        gts = torch.cat(self.quality_gts, dim=0)      # [N_val]

        # 计算 MAE 和 RMSE
        mae = (preds - gts).abs().mean().item()
        rmse = torch.sqrt(((preds - gts) ** 2).mean()).item()

        # 将结果写入 self.metrics.stats（或打印到控制台/日志）
        # SegmentMetrics.keys 包含 mAP/mIoU 等键，这里我们可以把 qualityscore 键附加进去
        self.metrics.stats["quality_mae"] = mae
        self.metrics.stats["quality_rmse"] = rmse

        # 打印日志
        LOGGER.info(f" → Quality MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        return self.metrics.stats

    # 保留其余方法，如 plot_val_samples、plot_predictions、save_one_txt、pred_to_json 等，无需改动。
