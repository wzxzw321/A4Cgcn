import torch
import torch.nn.functional as F
from copy import copy
from pathlib import Path

from ultralytics.models import yolo
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.plotting import plot_images, plot_results

from qualityscore.quality import SegmentationModelWithQuality
from qualityscore.qval import QualityValidator
from qualityscore.SegmentationWithQualityDataset import SegmentationWithQualityDataset


class QualityTrainer(yolo.segment.SegmentationTrainer):
    """
    在原生 SegmentationTrainer 基础上，加入“质量回归”分支的训练逻辑。
    1. __init__: 将 task 改为 "segment with score"；
    2. get_model: 返回 SegmentationModelWithQuality；
    3. get_validator: 返回 QualityValidator，并将 loss_names 增加 "quality_loss"；
    4. compute_loss: 先计算原生分割 loss，再计算 qualityscore 回归 loss（MSE），最后加权合并；
    5. plot_metrics: 在绘制原生分割指标后，额外打印 qualityscore 回归指标到 TensorBoard/控制台。
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        # 使用自定义任务名称，保证 Ultralytics 调用 QualityTrainer 而非 SegmentationTrainer
        overrides["task"] = "segment with score"
        super().__init__(cfg, overrides, _callbacks)

        # 从超参或 overrides 中读取分割 loss 权重与质量 loss 权重（可选）
        hyp = self.args.get("hyp", {}) or {}
        self.weight_seg = hyp.get("seg_loss", 1.0)
        self.weight_quality = hyp.get("quality_loss", 1.0)

        # 额外的 qualityscore 回归损失函数：MSELoss
        self.quality_loss_fn = torch.nn.MSELoss(reduction="mean")

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        覆盖原本 SegmentationTrainer.get_model → 使用 SegmentationModelWithQuality。
        """
        # cfg: 可以是 YAML 路径或者 dict。nc 与 ch 由 self.data 决定。
        model = SegmentationModelWithQuality(
            cfg if cfg is not None else self.args["model"],
            ch=self.data["channels"],
            nc=self.data["nc"],
            verbose=verbose and RANK == -1
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """
        返回 QualityValidator，同时把 loss_names 增加 "quality_loss"。
        """
        # 原本 SegmentationTrainer 会写 self.loss_names = ("box_loss", "seg_loss", "cls_loss", "dfl_loss")
        # 这里我们在最后加一个 "quality_loss"
        self.loss_names = ("box_loss", "seg_loss", "cls_loss", "dfl_loss", "quality_loss")
        return QualityValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks
        )

    def get_dataloader(self, mode):
        """
        重写此方法，让它根据 YAML 中的 path, train, val, test，
        构造 SegmentationWithQualityDataset，再用 create_dataloader 生成 DataLoader。
        """
        assert mode in ("train", "val"), "模式只能是 'train' 或 'val'"

        data_path = Path(self.data["path"])  # 例如 Path("/home/ubuntu/WZX/quality_test")
        imgsz = self.args.imgsz
        batch_size = self.args.batch

        if mode == "train":
            dataset = SegmentationWithQualityDataset(
                root=str(data_path), split="train", img_size=imgsz
            )
            loader, _ = self.get_dataloader(
                dataset=dataset,
                imgsz=imgsz,
                batch_size=batch_size,
                stride=self.model.stride.max(),
                single_cls=self.args.single_cls,
                pad=0.0,
                rect=self.args.rect,
                workers=self.args.workers,
                prefix="train: ",
            )
            return loader
        else:  # mode == "val"
            dataset = SegmentationWithQualityDataset(
                root=str(data_path), split="val", img_size=imgsz
            )
            loader, _ = self.get_dataloader(
                dataset=dataset,
                imgsz=imgsz,
                batch_size=batch_size,
                stride=self.model.stride.max(),
                single_cls=self.args.single_cls,
                pad=0.0,
                rect=self.args.rect,
                workers=self.args.workers,
                prefix="val: ",
            )
            return loader
    def compute_loss(self, images, batch):
        """
        计算训练时的总 loss，包括：
          1. 调用父类 compute_loss 得到原生分割相关 loss（已包含检测 + 分割）；
          2. 拿到 quality_pred → 与 batch["qualityscore"] 做 MSE → 得到 quality_loss；
          3. 按权重相加： total_loss = weight_seg * seg_loss + weight_quality * quality_loss；
        返回一个字典：
          {
            "box_loss": …,
            "seg_loss": …,
            "cls_loss": …,
            "dfl_loss": …,
            "quality_loss": …,
            "loss": total_loss
          }
        """
        # images: [B,3,H,W]
        # batch: dict，除了包含 "img", "cls", "bboxes", "masks" 外，还要包含 "qualityscore": [B]
        # 注意：Ultralytics 内部可能会把 "img" 重命名为 "img" 或 "images"。这里假设 batch["img"] 就是 images。

        imgs = images
        # 前向：SegmentationModelWithQuality.forward(imgs) → ((feats, pred_masks, proto), quality_pred)
        (feats, pred_masks, proto), quality_pred = self.model(imgs)

        # ---------- 1. 计算分割相关 loss ----------
        # 父类 SegmentationTrainer.compute_loss 只返回 seg 相关的 loss 值，而不是 dict。为了获取各项 loss，我们
        # 先调用父类 compute_loss → 得到 seg_loss_tensor。然后要从 self.model.init_criterion() 再手动解构每个分项。
        # 但最直接的做法是：调用原 v8SegmentationLoss 计算一次，得到完整的分割 loss 向量，再拆成 (box, seg, cls, dfl)。
        #
        crit = self.model.init_criterion()  # v8SegmentationLoss(self.model)
        # v8SegmentationLoss 的 __call__ 返回 (loss_tensor * batch_size, loss_components.detach())
        #   loss_tensor: shape [4] 包含 (box_loss, seg_loss, cls_loss, dfl_loss)
        #   loss_components.detach(): 同上但已 detach
        loss_vector, _ = crit((feats, pred_masks, proto), batch)
        # loss_vector: [4]，对应 [box, seg, cls, dfl]（已经乘以各自 gain，但未除以 batch_size）
        # 除以 batch_size 得到平均值
        bs = imgs.shape[0]
        box_loss = loss_vector[0] / bs
        seg_loss = loss_vector[1] / bs
        cls_loss = loss_vector[2] / bs
        dfl_loss = loss_vector[3] / bs

        # ---------- 2. 计算 qualityscore 回归 loss ----------
        if "qualityscore" not in batch:
            raise KeyError(
                "QualityTrainer 需要从 batch 中读取 'qualityscore' 标签，请检查 Dataset 是否返回了 batch['qualityscore']。"
            )
        quality_gt = batch["qualityscore"].to(quality_pred.device)  # [B]
        quality_loss = self.quality_loss_fn(quality_pred, quality_gt)  # MSE

        # ---------- 3. 加权合并所有 loss ----------
        total_loss = (
            self.weight_seg * (box_loss + seg_loss + cls_loss + dfl_loss)
            + self.weight_quality * quality_loss
        )

        # ---------- 4. 返回各项 loss，供 logging & backward ----------
        return {
            "box_loss": box_loss.detach(),
            "seg_loss": seg_loss.detach(),
            "cls_loss": cls_loss.detach(),
            "dfl_loss": dfl_loss.detach(),
            "quality_loss": quality_loss.detach(),
            "loss": total_loss
        }

    def plot_training_samples(self, batch, ni):
        """
        绘制训练样本：保留原本分割可视化，同时在标题或旁边加上 quality_gt 值，方便调试。
        """
        # 原版 SegmentationTrainer.plot_training_samples:
        # plot_images(
        #     batch["img"], batch["batch_idx"], batch["cls"].squeeze(-1),
        #     batch["bboxes"], masks=batch["masks"], paths=batch["im_file"],
        #     fname=self.save_dir / f"train_batch{ni}.jpg", on_plot=self.on_plot,
        # )
        #
        # 我们在可视化时，把 qualityscore 标签也当做文字 overlay 到图上。

        imgs = batch["img"]
        batch_idx = batch["batch_idx"]
        cls = batch["cls"].squeeze(-1)
        bboxes = batch["bboxes"]
        masks = batch["masks"]
        paths = batch.get("im_file", None)
        quality_gt = batch["qualityscore"]  # [B]

        # 直接调用原版 plot_images，可通过 on_plot 回调在标注区打印 qualityscore：
        def on_plot_with_quality(im, ax, idx):
            """
            在 ultralytics.utils.plotting.plot_images 的 on_plot 回调基础上叠加 qualityscore 文本。
            idx: 当前图在 batch 中的索引
            """
            q = float(quality_gt[idx].item())
            ax.text(
                0.02, 0.95,
                f"Q={q:.3f}",
                color="yellow", fontsize=10, transform=ax.transAxes,
                bbox=dict(facecolor="black", alpha=0.5, pad=2, edgecolor="none")
            )

        plot_images(
            imgs,
            batch_idx,
            cls,
            bboxes,
            masks=masks,
            paths=paths,
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=on_plot_with_quality,  # 替换回调
        )

    def plot_metrics(self):
        """
        绘制训练 & 验证指标图：
        - 原版 plot_results 会绘制 box/seg/cls/dfl 三类 loss 随 epoch 的变化，以及 mAP、mIoU 等指标。
        - 我们在此基础上，从 self.val_metrics 或 CSV 文件中获取 quality_mae 和 quality_rmse，然后打印到 TensorBoard/控制台。
        """
        # 先调用父类做原生分割指标绘制
        super().plot_metrics()

        # 再读取训练过程生成的 metrics.csv，在其中查找 quality_mae、quality_rmse 列
        try:
            import pandas as pd
            csv_path = self.csv  # self.csv 通常指向 "runs/exp*/metrics.csv"
            df = pd.read_csv(csv_path)
            # 假设在 QualityValidator.finalize_metrics 中，把 'quality_mae' 和 'quality_rmse' 写入了 metrics.csv
            if "quality_mae" in df.columns and "quality_rmse" in df.columns:
                # 仅打印到控制台，TensorBoard 已由父类写入
                latest_row = df.iloc[-1]
                q_mae = latest_row["quality_mae"]
                q_rmse = latest_row["quality_rmse"]
                self.on_plot(f"Latest Quality MAE: {q_mae:.4f}", txt=True)
                self.on_plot(f"Latest Quality RMSE: {q_rmse:.4f}", txt=True)
        except Exception:
            # 若无法读取或列不存在，则静默忽略
            pass
