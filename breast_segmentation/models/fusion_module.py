"""Lightning module for fusion-based breast segmentation training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import lightning.pytorch as L
import numpy as np
import cv2
from typing import Dict, Any, Optional, List, Union, Tuple
from monai.inferers import sliding_window_inference
import segmentation_models_pytorch as smp

from .architectures import get_model, VENUS
from ..metrics.losses import (
    get_loss_function,
    SurfaceLossBinary,
    CABFL,
    CrossEntropy2d,
    compute_class_weight,
)
from ..metrics.evaluation import (
    compute_iou,
    compute_dice,
    compute_metrics_from_confusion_matrix,
    compute_iou_from_metrics,
    compute_dice_from_metrics,
    compute_dice_score,
    compute_dice_score_from_cm,
    compute_mean_precision,
    compute_mean_recall,
    class_specific_accuracy_score,
)


class BreastFusionModel(L.LightningModule):
    """Lightning module for fusion-based breast segmentation model training with 3 inputs."""

    def __init__(
        self,
        arch: str = "venus",
        encoder_name: Optional[str] = None,
        in_channels: int = 1,
        out_classes: int = 1,
        batch_size: int = 24,
        len_train_loader: int = 100,
        threshold: float = 0.4,
        base_channels: int = 64,
        loss_function: str = "asymmetric_unified_focal",
        loss_kwargs: Optional[Dict[str, Any]] = None,
        use_boundary_loss: bool = False,
        use_decoder_attention: bool = True,
        use_simple_fusion: bool = False,
        img_size: int = 256,
        augment_inference: bool = False,
        **kwargs,
    ):
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters()

        self.model = VENUS(
            n_channels=in_channels,
            n_classes=out_classes,
            use_simple_fusion=use_simple_fusion,
            use_decoder_attention=use_decoder_attention,
            base_channels=base_channels,
        )

        # Loss function
        loss_kwargs = loss_kwargs or {}
        if isinstance(loss_function, str):
            self.loss_fn = get_loss_function(loss_function, **loss_kwargs)
        else:
            self.loss_fn = loss_function

        # Training parameters
        self.threshold = threshold
        self.batch_size = batch_size
        self.len_train_loader = len_train_loader
        self.use_boundary_loss = use_boundary_loss
        self.augment_inference = augment_inference

        # Metrics storage
        self.train_outputs = []
        self.val_outputs = []
        self.test_outputs = []

        self.boundaryloss = use_boundary_loss
        self.t_loss = self.loss_fn  # Alias for compatibility

    def forward(
        self, image1: torch.Tensor, image2: torch.Tensor, image3: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with 3 inputs."""
        mask = self.model(image1, image2, image3)
        return mask

    def step(
        self,
        batch: Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]],
        batch_idx: int,
        stage: str,
    ) -> torch.Tensor:
        """Training/validation/test step for fusion model."""

        image1 = batch[0]["image"].to("cuda")
        image2 = batch[1]["image"].to("cuda")
        image3 = batch[2]["image"].to("cuda")

        assert image1.ndim == 4
        h, w = image1.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch[0]["label"].to("cuda")
        assert mask.ndim == 4
        assert mask.max() <= 1 and mask.min() >= 0

        if stage == "test" and self.augment_inference:
            # Use test-time augmentation
            prob_mask = self.ttaug(self.model, image1, image2, image3)
            logits_mask = self.forward(image1, image2, image3)
        else:
            logits_mask = self.forward(image1, image2, image3)
            prob_mask = logits_mask.sigmoid()

        if self.boundaryloss and isinstance(self.t_loss, CABFL):
            dist_map = batch[0]["boundary"].to("cuda")
            t_loss = self.t_loss(logits_mask, prob_mask, dist_map, mask, self.current_epoch)
        else:
            t_loss = self.t_loss(logits_mask, mask)

        loss = t_loss
        pred_mask = (prob_mask > self.threshold).float()

        if stage == "test" and self.augment_inference:
            pred_mask = (prob_mask > self.threshold).int()
            # Add post-processing here if needed
            pred_mask = torch.Tensor(pred_mask).to("cuda")

        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        # Convert to float to avoid division issues
        tp = tp.float()
        fp = fp.float()
        fn = fn.float()
        tn = tn.float()

        iou_per_image_mass = compute_iou(
            y_true=mask, y_pred=pred_mask, class_id=1, reduction="micro_image_wise"
        )
        iou_per_image_background = compute_iou(
            y_true=mask, y_pred=pred_mask, class_id=0, reduction="micro_image_wise"
        )

        iou_per_image_mass_no_empty = compute_iou(
            y_true=mask,
            y_pred=pred_mask,
            class_id=1,
            reduction="micro_image_wise",
            exclude_empty=True,
        )
        iou_per_image_background_no_empty = compute_iou(
            y_true=mask,
            y_pred=pred_mask,
            class_id=0,
            reduction="micro_image_wise",
            exclude_empty=True,
        )

        iou_per_dataset_mass = compute_iou(
            y_true=mask, y_pred=pred_mask, class_id=1, reduction="micro"
        )
        iou_per_dataset_background = compute_iou(
            y_true=mask, y_pred=pred_mask, class_id=0, reduction="micro"
        )

        iou_per_dataset_mass_no_empty = compute_iou(
            y_true=mask, y_pred=pred_mask, class_id=1, reduction="micro", exclude_empty=True
        )
        iou_per_dataset_background_no_empty = compute_iou(
            y_true=mask, y_pred=pred_mask, class_id=0, reduction="micro", exclude_empty=True
        )

        dice_per_image_mass = compute_dice_score(
            y_true=mask, y_pred=pred_mask, class_id=1, reduction="micro_image_wise"
        )
        dice_per_image_background = compute_dice_score(
            y_true=mask, y_pred=pred_mask, class_id=0, reduction="micro_image_wise"
        )

        dice_per_image_mass_no_empty = compute_dice_score(
            y_true=mask,
            y_pred=pred_mask,
            class_id=1,
            reduction="micro_image_wise",
            exclude_empty=True,
        )
        dice_per_image_background_no_empty = compute_dice_score(
            y_true=mask,
            y_pred=pred_mask,
            class_id=0,
            reduction="micro_image_wise",
            exclude_empty=True,
        )

        dice_per_dataset_mass = compute_dice_score(
            y_true=mask, y_pred=pred_mask, class_id=1, reduction="micro"
        )
        dice_per_dataset_background = compute_dice_score(
            y_true=mask, y_pred=pred_mask, class_id=0, reduction="micro"
        )

        dice_per_dataset_mass_no_empty = compute_dice_score(
            y_true=mask, y_pred=pred_mask, class_id=1, reduction="micro", exclude_empty=True
        )
        dice_per_dataset_background_no_empty = compute_dice_score(
            y_true=mask, y_pred=pred_mask, class_id=0, reduction="micro", exclude_empty=True
        )

        acc_per_image_mass = class_specific_accuracy_score(
            preds=mask, targets=pred_mask, class_id=1, averaging="micro_image_wise"
        )
        acc_per_image_background = class_specific_accuracy_score(
            preds=mask, targets=pred_mask, class_id=0, averaging="micro_image_wise"
        )

        acc_per_dataset_mass = class_specific_accuracy_score(
            preds=mask, targets=pred_mask, class_id=1, averaging="micro"
        )
        acc_per_dataset_background = class_specific_accuracy_score(
            preds=mask, targets=pred_mask, class_id=0, averaging="micro"
        )

        loss = loss.to("cpu")

        output = {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "iou_per_image_mass": iou_per_image_mass,
            "iou_per_image_background": iou_per_image_background,
            "iou_per_dataset_mass": iou_per_dataset_mass,
            "iou_per_dataset_background": iou_per_dataset_background,
            "iou_per_image_mass_no_empty": iou_per_image_mass_no_empty,
            "iou_per_image_background_no_empty": iou_per_image_background_no_empty,
            "iou_per_dataset_mass_no_empty": iou_per_dataset_mass_no_empty,
            "iou_per_dataset_background_no_empty": iou_per_dataset_background_no_empty,
            "dice_per_image_mass": dice_per_image_mass,
            "dice_per_image_background": dice_per_image_background,
            "dice_per_dataset_mass": dice_per_dataset_mass,
            "dice_per_dataset_background": dice_per_dataset_background,
            "dice_per_image_mass_no_empty": dice_per_image_mass_no_empty,
            "dice_per_image_background_no_empty": dice_per_image_background_no_empty,
            "dice_per_dataset_mass_no_empty": dice_per_dataset_mass_no_empty,
            "dice_per_dataset_background_no_empty": dice_per_dataset_background_no_empty,
            "acc_per_image_mass": acc_per_image_mass,
            "acc_per_image_background": acc_per_image_background,
            "acc_per_dataset_mass": acc_per_dataset_mass,
            "acc_per_dataset_background": acc_per_dataset_background,
        }

        if stage == "train":
            self.train_outputs.append(output)
        if stage == "valid":
            self.val_outputs.append(output)
        if stage == "test":
            self.test_outputs.append(output)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, batch_size=self.batch_size)
        return loss

    def training_step(self, batch, batch_idx):
        if batch:
            return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        if batch:
            return self.step(batch, batch_idx, "valid")

    def test_step(self, batch, batch_idx):
        if batch:
            return self.step(batch, batch_idx, "test")

    def shared_epoch_end(self, outputs, stage):
        """Aggregate step metrics at the end of each epoch."""
        if not outputs:
            return

        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        iou_per_image_mass = torch.nanmean(torch.Tensor([x["iou_per_image_mass"] for x in outputs]))
        iou_per_image_background = torch.nanmean(
            torch.Tensor([x["iou_per_image_background"] for x in outputs])
        )
        iou_per_dataset_image = torch.nanmean(
            torch.Tensor([x["iou_per_dataset_mass"] for x in outputs])
        )
        iou_per_dataset_background = torch.nanmean(
            torch.Tensor([x["iou_per_dataset_background"] for x in outputs])
        )

        iou_per_image_mass_no_empty = torch.nanmean(
            torch.Tensor([x["iou_per_image_mass_no_empty"] for x in outputs])
        )
        iou_per_image_background_no_empty = torch.nanmean(
            torch.Tensor([x["iou_per_image_background_no_empty"] for x in outputs])
        )
        iou_per_dataset_image_no_empty = torch.nanmean(
            torch.Tensor([x["iou_per_dataset_mass_no_empty"] for x in outputs])
        )
        iou_per_dataset_background_no_empty = torch.nanmean(
            torch.Tensor([x["iou_per_dataset_background_no_empty"] for x in outputs])
        )

        dice_per_image_mass = torch.nanmean(
            torch.Tensor([x["dice_per_image_mass"] for x in outputs])
        )
        dice_per_image_background = torch.nanmean(
            torch.Tensor([x["dice_per_image_background"] for x in outputs])
        )
        dice_per_dataset_image = torch.nanmean(
            torch.Tensor([x["dice_per_dataset_mass"] for x in outputs])
        )
        dice_per_dataset_background = torch.nanmean(
            torch.Tensor([x["dice_per_dataset_background"] for x in outputs])
        )

        dice_per_image_mass_no_empty = torch.nanmean(
            torch.Tensor([x["dice_per_image_mass_no_empty"] for x in outputs])
        )
        dice_per_image_background_no_empty = torch.nanmean(
            torch.Tensor([x["dice_per_image_background_no_empty"] for x in outputs])
        )
        dice_per_dataset_image_no_empty = torch.nanmean(
            torch.Tensor([x["dice_per_dataset_mass_no_empty"] for x in outputs])
        )
        dice_per_dataset_background_no_empty = torch.nanmean(
            torch.Tensor([x["dice_per_dataset_background_no_empty"] for x in outputs])
        )

        acc_per_image_mass = torch.nanmean(torch.Tensor([x["acc_per_image_mass"] for x in outputs]))
        acc_per_image_background = torch.nanmean(
            torch.Tensor([x["acc_per_image_background"] for x in outputs])
        )
        acc_per_dataset_image = torch.nanmean(
            torch.Tensor([x["acc_per_dataset_mass"] for x in outputs])
        )
        acc_per_dataset_background = torch.nanmean(
            torch.Tensor([x["acc_per_dataset_background"] for x in outputs])
        )

        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        per_dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        per_image_dice = compute_dice_score_from_cm(tp, fp, fn, tn, reduction="micro-imagewise")
        per_dataset_dice = compute_dice_score_from_cm(tp, fp, fn, tn, reduction="micro")

        # MACRO AVG
        precision = compute_mean_precision(tp, fp, fn, tn)
        recall = compute_mean_recall(tp, fp, fn, tn)

        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro-imagewise")

        # MACRO IMAGEWISE MEAN DICE WITH EMPTY
        dice1_per_image = compute_dice_from_metrics(tp, fp, tn, fn, reduction="none")
        dice0_per_image = compute_dice_from_metrics(tn, fn, tp, fp, reduction="none")
        mean_dice_per_image = np.mean(
            np.nanmean(
                np.array([dice0_per_image.cpu().numpy(), dice1_per_image.cpu().numpy()]), axis=0
            )
        )

        # MACRO MEAN DICE WITH EMPTY
        mean_dice1_per_dataset = compute_dice_from_metrics(
            tp, fp, tn, fn, reduction="micro-imagewise"
        )
        mean_dice0_per_dataset = compute_dice_from_metrics(
            tn, fn, tp, fp, reduction="micro-imagewise"
        )
        mean_dice_per_dataset = np.nanmean(
            np.array([mean_dice0_per_dataset.cpu().numpy(), mean_dice1_per_dataset.cpu().numpy()])
        )

        # MACRO IMAGEWISE MEAN DICE NO EMPTY
        dice1_per_image_no_empty = compute_dice_from_metrics(
            tp, fp, tn, fn, reduction="none", exclude_empty=True
        )
        dice0_per_image_no_empty = compute_dice_from_metrics(
            tn, fn, tp, fp, reduction="none", exclude_empty=True
        )
        combined_dice_scores = np.hstack((dice0_per_image_no_empty, dice1_per_image_no_empty))
        valid_pairs = ~np.isnan(combined_dice_scores).any(axis=1)
        mean_dice_per_image_no_empty = np.mean(
            np.nanmean(combined_dice_scores[valid_pairs], axis=1)
        )

        if mean_dice_per_image_no_empty.size == 0:
            mean_dice_per_image_no_empty = float("nan")

        # MACRO MEAN DICE NO EMPTY
        mean_dice1_per_dataset_no_empty = compute_dice_from_metrics(
            tp, fp, tn, fn, reduction="micro-imagewise", exclude_empty=True
        )
        mean_dice0_per_dataset_no_empty = compute_dice_from_metrics(
            tn, fn, tp, fp, reduction="micro-imagewise", exclude_empty=True
        )
        mean_dice_per_dataset_no_empty = np.mean(
            np.array(
                [
                    mean_dice0_per_dataset_no_empty.cpu().numpy(),
                    mean_dice1_per_dataset_no_empty.cpu().numpy(),
                ]
            )
        )

        # MACRO IMAGEWISE MEAN IOU WITH EMPTY
        iou1_per_image = compute_iou_from_metrics(tp, fp, tn, fn, reduction="none")
        iou0_per_image = compute_iou_from_metrics(tn, fn, tp, fp, reduction="none")
        mean_iou_per_image = np.mean(
            np.nanmean(
                np.array([iou0_per_image.cpu().numpy(), iou1_per_image.cpu().numpy()]), axis=0
            )
        )

        # MACRO MEAN IOU WITH EMPTY
        mean_iou1_per_dataset = compute_iou_from_metrics(
            tp, fp, tn, fn, reduction="micro-imagewise"
        )
        mean_iou0_per_dataset = compute_iou_from_metrics(
            tn, fn, tp, fp, reduction="micro-imagewise"
        )
        mean_iou_per_dataset = np.nanmean(
            np.array([mean_iou0_per_dataset.cpu().numpy(), mean_iou1_per_dataset.cpu().numpy()])
        )

        # MACRO IMAGEWISE MEAN IOU NO EMPTY
        iou1_per_image_no_empty = compute_iou_from_metrics(
            tp, fp, tn, fn, reduction="none", exclude_empty=True
        )
        iou0_per_image_no_empty = compute_iou_from_metrics(
            tn, fn, tp, fp, reduction="none", exclude_empty=True
        )

        combined_iou_scores = np.hstack((iou0_per_image_no_empty, iou1_per_image_no_empty))
        valid_pairs = ~np.isnan(combined_iou_scores).any(axis=1)
        mean_iou_per_image_no_empty = np.mean(np.nanmean(combined_iou_scores[valid_pairs], axis=1))

        if mean_iou_per_image_no_empty.size == 0:
            mean_iou_per_image_no_empty = float("nan")

        # MACRO MEAN IOU NO EMPTY
        mean_iou1_per_dataset_no_empty = compute_iou_from_metrics(
            tp, fp, tn, fn, reduction="micro-imagewise", exclude_empty=True
        )
        mean_iou0_per_dataset_no_empty = compute_iou_from_metrics(
            tn, fn, tp, fp, reduction="micro-imagewise", exclude_empty=True
        )
        mean_iou_per_dataset_no_empty = np.mean(
            np.array(
                [
                    mean_iou0_per_dataset_no_empty.cpu().numpy(),
                    mean_iou1_per_dataset_no_empty.cpu().numpy(),
                ]
            )
        )

        self.log(
            f"{stage}_per_image_iou",
            per_image_iou,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            f"{stage}_per_dataset_iou",
            per_dataset_iou,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )

        self.log(
            f"{stage}_per_image_dice",
            per_image_dice,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            f"{stage}_per_dataset_dice",
            per_dataset_dice,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )

        self.log(
            f"{stage}_mean_iou_per_image",
            mean_iou_per_image,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            f"{stage}_mean_iou_per_dataset",
            mean_iou_per_dataset,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )

        self.log(
            f"{stage}_mean_iou_per_image_no_empty",
            mean_iou_per_image_no_empty,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            f"{stage}_mean_iou_per_dataset_no_empty",
            mean_iou_per_dataset_no_empty,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )

        self.log(
            f"{stage}_mean_dice_per_image",
            mean_dice_per_image,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            f"{stage}_mean_dice_per_dataset",
            mean_dice_per_dataset,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )

        self.log(
            f"{stage}_mean_dice_per_image_no_empty",
            mean_dice_per_image_no_empty,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            f"{stage}_mean_dice_per_dataset_no_empty",
            mean_dice_per_dataset_no_empty,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )

        self.log(
            f"{stage}_precision",
            precision,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            f"{stage}_recall", recall, sync_dist=True, prog_bar=True, batch_size=self.batch_size
        )
        self.log(
            f"{stage}_accuracy", accuracy, sync_dist=True, prog_bar=True, batch_size=self.batch_size
        )

        self.log(
            f"{stage}_iou_per_image_mass",
            iou_per_image_mass,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            f"{stage}_iou_per_image_background",
            iou_per_image_background,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log(
            f"{stage}_iou_per_dataset_mass",
            iou_per_dataset_image,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log(
            f"{stage}_iou_per_dataset_background",
            iou_per_dataset_background,
            sync_dist=True,
            batch_size=self.batch_size,
        )

        self.log(
            f"{stage}_iou_per_image_mass_no_empty",
            iou_per_image_mass_no_empty,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            f"{stage}_iou_per_image_background_no_empty",
            iou_per_image_background_no_empty,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log(
            f"{stage}_iou_per_dataset_mass_no_empty",
            iou_per_dataset_image_no_empty,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log(
            f"{stage}_iou_per_dataset_background_no_empty",
            iou_per_dataset_background_no_empty,
            sync_dist=True,
            batch_size=self.batch_size,
        )

        self.log(
            f"{stage}_dice_per_image_mass",
            dice_per_image_mass,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            f"{stage}_dice_per_image_background",
            dice_per_image_background,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log(
            f"{stage}_dice_per_dataset_mass",
            dice_per_dataset_image,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log(
            f"{stage}_dice_per_dataset_background",
            dice_per_dataset_background,
            sync_dist=True,
            batch_size=self.batch_size,
        )

        self.log(
            f"{stage}_dice_per_image_mass_no_empty",
            dice_per_image_mass_no_empty,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            f"{stage}_dice_per_image_background_no_empty",
            dice_per_image_background_no_empty,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log(
            f"{stage}_dice_per_dataset_mass_no_empty",
            dice_per_dataset_image_no_empty,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log(
            f"{stage}_dice_per_dataset_background_no_empty",
            dice_per_dataset_background_no_empty,
            sync_dist=True,
            batch_size=self.batch_size,
        )

        self.log(
            f"{stage}_acc_per_image_mass",
            acc_per_image_mass,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            f"{stage}_acc_per_image_background",
            acc_per_image_background,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            f"{stage}_acc_per_dataset_mass",
            acc_per_dataset_image,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            f"{stage}_acc_per_dataset_background",
            acc_per_dataset_background,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )

    def on_train_epoch_end(self):
        self.shared_epoch_end(outputs=self.train_outputs, stage="train")
        self.train_outputs.clear()

    def on_validation_epoch_end(self):
        self.shared_epoch_end(outputs=self.val_outputs, stage="valid")
        self.val_outputs.clear()

    def on_test_epoch_end(self):
        self.shared_epoch_end(outputs=self.test_outputs, stage="test")
        self.test_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-4)

        iterations_per_epoch = self.len_train_loader
        step_size_up = iterations_per_epoch // 2
        gamma = 0.99

        base_lr = 3e-5
        max_lr = 9e-4

        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=step_size_up,
            mode="triangular",
            cycle_momentum=False,
        )

        return [optimizer], [scheduler]

    def ttaug(self, model, image1, image2, image3):
        """Test-time augmentation for 3-input model."""

        # Simple implementation for now
        # Define picklable transform functions
        def identity_transform(x):
            return x

        def horizontal_flip_transform(x):
            return torch.flip(x, dims=[3])

        def vertical_flip_transform(x):
            return torch.flip(x, dims=[2])

        transforms = [
            identity_transform,
            horizontal_flip_transform,  # Horizontal flip
            vertical_flip_transform,  # Vertical flip
        ]

        model.eval()
        outputs = []

        for transform in transforms:
            aug_img1 = transform(image1).to("cuda")
            aug_img2 = transform(image2).to("cuda")
            aug_img3 = transform(image3).to("cuda")

            model_output = model(aug_img1, aug_img2, aug_img3)

            # Reverse augmentation
            if transform == transforms[1]:  # Horizontal flip
                deaug_mask = torch.flip(model_output, dims=[3])
            elif transform == transforms[2]:  # Vertical flip
                deaug_mask = torch.flip(model_output, dims=[2])
            else:
                deaug_mask = model_output

            outputs.append(deaug_mask)

        masks = torch.stack(outputs).mean(dim=0)
        return masks

    def single_predict_sliding_window(
        self, image1, image2, image3, roi_size=(128, 128), sw_batch_size=128, overlap=0
    ):
        """Sliding window inference for 3-input model."""
        # This would need to be implemented for proper sliding window inference with 3 inputs
        # For now, just do regular inference
        return self.forward(image1, image2, image3)
