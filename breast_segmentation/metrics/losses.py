"""Loss functions for breast segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import numpy as np
from numpy import einsum

from monai.losses import DiceLoss as MonaiDiceLoss, DiceCELoss
from torch.autograd import Variable


class AsymmetricFocalLoss(nn.Module):
    """Asymmetric Focal Loss for handling class imbalance."""

    def __init__(self, delta: float = 0.7, gamma: float = 2.0):
        super().__init__()
        self.delta = delta
        self.gamma = gamma

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: Predicted logits [B, C, H, W]
            y_true: Ground truth labels [B, C, H, W]

        Returns:
            Loss value
        """
        y_pred = y_pred.float()
        y_true = y_true.float()

        # Apply sigmoid to get probabilities
        y_pred = torch.sigmoid(y_pred)

        # Compute cross entropy
        cross_entropy = -y_true * torch.log(y_pred + 1e-7)

        # Compute back/fore ground losses
        back_ce = (
            (1 - self.delta)
            * (1 - y_true)
            * torch.pow(y_pred, self.gamma)
            * torch.log(1 - y_pred + 1e-7)
        )
        fore_ce = self.delta * y_true * torch.pow(1 - y_pred, self.gamma) * cross_entropy

        loss = torch.mean(fore_ce - back_ce)

        return loss


class AsymmetricFocalTverskyLoss(nn.Module):
    """Asymmetric Focal Tversky Loss."""

    def __init__(
        self, delta: float = 0.7, gamma: float = 0.75, alpha: float = 0.5, beta: float = 0.5
    ):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: Predicted logits [B, C, H, W]
            y_true: Ground truth labels [B, C, H, W]

        Returns:
            Loss value
        """
        y_pred = torch.sigmoid(y_pred)

        # Tversky index components
        tp = torch.sum(y_true * y_pred, dim=(2, 3))
        fp = torch.sum((1 - y_true) * y_pred, dim=(2, 3))
        fn = torch.sum(y_true * (1 - y_pred), dim=(2, 3))

        tversky = (tp + 1e-7) / (tp + self.alpha * fp + self.beta * fn + 1e-7)
        focal_tversky = torch.pow((1 - tversky), self.gamma)

        # Asymmetric component
        back_dice = (
            (1 - self.delta)
            * torch.sum((1 - y_true) * y_pred**2, dim=(2, 3))
            / (
                torch.sum((1 - y_true) * y_pred**2 + (1 - y_true) * (1 - y_pred) ** 2, dim=(2, 3))
                + 1e-7
            )
        )
        fore_dice = self.delta * (2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7)

        loss = torch.mean(focal_tversky - back_dice + (1 - fore_dice))

        return loss


class AsymmetricUnifiedFocalLoss(nn.Module):
    """Asymmetric Unified Focal Loss combining Focal and Focal Tversky losses."""

    def __init__(self, weight: float = 0.5, delta: float = 0.6, gamma: float = 0.2):
        super().__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted logits [B, C, H, W]
            y_true: Ground truth labels [B, C, H, W]

        Returns:
            Loss value
        """
        asymmetric_ftl = AsymmetricFocalTverskyLoss(delta=self.delta, gamma=self.gamma)(
            logits, y_true
        )
        asymmetric_fl = AsymmetricFocalLoss(delta=self.delta, gamma=self.gamma)(logits, y_true)

        if self.weight is not None:
            return (self.weight * asymmetric_ftl) + ((1 - self.weight) * asymmetric_fl)
        else:
            return asymmetric_ftl + asymmetric_fl


class SoftDiceLoss(nn.Module):
    """Soft Dice Loss for segmentation."""

    def __init__(self, smooth: float = 1.0, dims: Tuple[int, ...] = (2, 3)):
        super().__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: Predicted logits [B, C, H, W]
            y_true: Ground truth labels [B, C, H, W]

        Returns:
            Loss value
        """
        y_pred = torch.sigmoid(y_pred)

        # Compute intersection and union
        intersection = torch.sum(y_true * y_pred, dim=self.dims)
        union = torch.sum(y_true, dim=self.dims) + torch.sum(y_pred, dim=self.dims)

        # Compute Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Return loss
        return 1.0 - torch.mean(dice)


class FocalTverskyIOULoss(nn.Module):
    """Combined Focal Tversky and IoU Loss."""

    def __init__(
        self, alpha: float = 0.5, beta: float = 0.5, gamma: float = 0.75, iou_weight: float = 0.5
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.iou_weight = iou_weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: Predicted logits [B, C, H, W]
            y_true: Ground truth labels [B, C, H, W]

        Returns:
            Loss value
        """
        y_pred = torch.sigmoid(y_pred)

        # Tversky components
        tp = torch.sum(y_true * y_pred, dim=(2, 3))
        fp = torch.sum((1 - y_true) * y_pred, dim=(2, 3))
        fn = torch.sum(y_true * (1 - y_pred), dim=(2, 3))

        # Tversky index
        tversky = (tp + 1e-7) / (tp + self.alpha * fp + self.beta * fn + 1e-7)
        focal_tversky = torch.pow((1 - tversky), self.gamma)

        # IoU
        intersection = tp
        union = tp + fp + fn
        iou = (intersection + 1e-7) / (union + 1e-7)
        iou_loss = 1 - iou

        # Combine losses
        loss = torch.mean(focal_tversky + self.iou_weight * iou_loss)

        return loss


class SurfaceLossBinary(nn.Module):
    """Surface Loss for boundary-aware segmentation."""

    def __init__(self, idc: List[int] = [1]):
        super().__init__()
        self.idc = idc
        print(f"Initialized {self.__class__.__name__} with {idc}")

    def forward(self, probs: torch.Tensor, dist_maps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            probs: Predicted probabilities [B, C, H, W]
            dist_maps: Distance maps [B, C, H, W]

        Returns:
            Loss value
        """
        pc = probs[:, 0, ...].type(torch.float32)
        dc = dist_maps[:, 1, ...].type(torch.float32)

        multipled = einsum("bwh,bwh->bwh", pc, dc)
        loss = multipled.mean()

        return loss


class CABFL(nn.Module):
    """Combined Asymmetric Boundary Focal Loss - exact implementation from original."""

    def __init__(
        self, idc: List[int], weight_aufl: float = 0.5, delta: float = 0.6, gamma: float = 0.2
    ):
        super().__init__()
        self.boundaryLoss = SurfaceLossBinary(idc=idc)
        self.aufl = AsymmetricUnifiedFocalLoss(delta=delta, gamma=gamma, weight=weight_aufl)
        self.alpha = 0.01
        self.current_epoch = 0

    def norm_distmap(self, distmap: torch.Tensor) -> torch.Tensor:
        """Normalize distance map."""
        _m = torch.abs(distmap).max()
        return distmap / _m

    def forward(
        self,
        logits: torch.Tensor,
        probs: torch.Tensor,
        dist_maps: torch.Tensor,
        gts: torch.Tensor,
        current_epoch: int,
    ) -> torch.Tensor:
        """
        Args:
            logits: Predicted logits [B, C, H, W]
            probs: Predicted probabilities [B, C, H, W]
            dist_maps: Distance maps [B, C, H, W]
            gts: Ground truth labels [B, C, H, W]
            current_epoch: Current training epoch

        Returns:
            Loss value
        """
        if current_epoch != self.current_epoch:
            self.current_epoch = current_epoch
            self.alpha = min(self.alpha + 0.01, 0.99)

        bl = self.boundaryLoss(probs, self.norm_distmap(dist_maps))
        aufl = self.aufl(logits, gts)

        return (1 - self.alpha) * aufl + self.alpha * bl


def CrossEntropy2d(input, target, weight=None, reduction="mean"):
    """
    Binary cross entropy for 2D inputs with shape (B, H, W).
    `weight` is applied per class.
    """
    # Ensure input and target have the correct dimensions
    target = target.float()  # Convert target to float for BCEWithLogitsLoss

    # Handle class weights
    if weight is not None:
        # Expand weight to match the shape of the input/target
        weight = weight[target.long()]  # Select weights based on the class labels

    return F.binary_cross_entropy_with_logits(input, target, weight=weight, reduction=reduction)


def compute_class_weight(labels):
    """
    Compute class weights for binary classification with labels of shape (B, 1, C, H, W).
    """
    labels = labels.cpu()
    labels = labels.squeeze(1).reshape(-1)  # Flatten labels to (B * C * H * W)
    unique_labels = np.unique(labels.numpy())  # Get unique class labels

    class_freq = {}
    for label in unique_labels:
        class_freq[label] = np.sum(labels.numpy() == label)  # Count occurrences of each class

    total_samples = labels.numel()
    n_classes = len(unique_labels)

    # Compute class weights: inverse frequency normalized
    class_weights = {}
    for label in unique_labels:
        class_weights[label] = total_samples / (n_classes * class_freq[label])

    # Convert to a PyTorch tensor
    return torch.tensor([class_weights[label] for label in sorted(class_weights.keys())]).float()


def get_loss_function(loss_name: str, **kwargs) -> nn.Module:
    """Factory function to get loss function by name."""
    # Create loss functions directly to avoid pickling issues with lambdas
    if loss_name == "dice":
        return MonaiDiceLoss(**kwargs)
    elif loss_name == "dice_ce":
        return DiceCELoss(**kwargs)
    elif loss_name == "soft_dice":
        return SoftDiceLoss(**kwargs)
    elif loss_name == "focal":
        return AsymmetricFocalLoss(**kwargs)
    elif loss_name == "focal_tversky":
        return AsymmetricFocalTverskyLoss(**kwargs)
    elif loss_name == "focal_tversky_iou":
        return FocalTverskyIOULoss(**kwargs)
    elif loss_name == "surface":
        return SurfaceLossBinary(**kwargs)
    elif loss_name == "cabfl":
        return CABFL(**kwargs)
    elif loss_name == "asymmetric_unified_focal":
        return AsymmetricUnifiedFocalLoss(**kwargs)
    elif loss_name == "crossentropy2d":
        return CrossEntropy2d
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
