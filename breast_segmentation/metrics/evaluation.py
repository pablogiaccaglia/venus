"""Evaluation metrics for breast segmentation."""

import torch
import numpy as np
from typing import Tuple, Optional, Union, List
from monai.metrics import DiceMetric


def compute_metrics_from_confusion_matrix(
    tp: torch.Tensor, 
    fp: torch.Tensor, 
    tn: torch.Tensor, 
    fn: torch.Tensor,
    eps: float = 1e-7
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute precision, recall, F1 score, and accuracy from confusion matrix components.
    
    Args:
        tp: True positives
        fp: False positives
        tn: True negatives
        fn: False negatives
        eps: Small epsilon to avoid division by zero
    
    Returns:
        Tuple of (precision, recall, f1_score, accuracy)
    """
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1_score = 2 * (precision * recall) / (precision + recall + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    
    return precision, recall, f1_score, accuracy


def compute_iou_from_metrics(tp, fp, tn, fn, reduction='micro', exclude_empty=False):
    denominator = tp + fp + fn
    with torch.no_grad():  # Avoid tracking these operations in the autograd graph
        if reduction == 'micro':
            # Sum the counts across all samples and compute IoU
            iou = tp.sum() / (denominator.sum())
        elif reduction == 'micro-imagewise':
            # Compute IoU per sample, then average across samples
            valid = denominator != 0
            iou_per_sample = torch.zeros_like(tp, dtype=torch.float32)
            iou_per_sample[valid] = tp[valid].float() / denominator[valid].float()
            
            if exclude_empty:
                iou_per_sample[~valid] = torch.tensor(float('nan'))
                iou = torch.nanmean(iou_per_sample)
            else:
                iou_per_sample[~valid] = torch.tensor(1.0)
                iou = torch.mean(iou_per_sample)
        elif reduction == 'none':
            # Return IoU for each sample without averaging
            valid = denominator != 0
            iou = torch.zeros_like(tp, dtype=torch.float32)
            iou[valid] = tp[valid].float() / denominator[valid].float()
            if exclude_empty:
                iou[~valid] = torch.tensor(float('nan'))
            else:
                iou[~valid] = torch.tensor(1.0)
        else:
            raise ValueError("Reduction method must be either 'micro', 'micro-imagewise', or 'none'.")

    return iou


def compute_dice_from_metrics(tp, fp, tn, fn, reduction='micro', exclude_empty=False):
    dice_denominator = 2 * tp + fp + fn
    dice_numerator = 2 * tp

    if reduction == 'micro':
        dice_score = dice_numerator.sum() / dice_denominator.sum()
    elif reduction == 'micro-imagewise':
        dice_per_sample = dice_numerator / dice_denominator
        if exclude_empty:
            dice_per_sample = torch.where(dice_denominator == 0, torch.tensor(float('nan')), dice_per_sample)
            dice_score = torch.nanmean(dice_per_sample)
        else:
            dice_per_sample = torch.where(dice_denominator == 0, 1, dice_per_sample)
            dice_score = torch.nanmean(dice_per_sample)
    elif reduction == 'none':
        dice_score = dice_numerator / dice_denominator
        if exclude_empty:
            dice_score = torch.where(dice_denominator == 0, torch.tensor(float('nan')), dice_score)
        else:
            dice_score = torch.where(dice_denominator == 0, 1, dice_score)
    else:
        raise ValueError("Reduction method must be either 'micro', 'micro-imagewise', or 'none'.")

    # Ensure Dice scores are within the [0, 1] range
    dice_score = torch.clamp(dice_score, min=0, max=1)

    return dice_score


def compute_iou(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    class_id: int = 1,
    reduction: str = 'micro',
    exclude_empty: bool = False
) -> float:
    """
    Compute Intersection over Union for a specific class.
    
    Args:
        y_true: Ground truth tensor [B, H, W] or [B, C, H, W]
        y_pred: Prediction tensor [B, H, W] or [B, C, H, W]
        class_id: Class ID to compute IoU for
        reduction: Reduction method ('micro' or 'micro_image_wise')
        exclude_empty: Whether to exclude empty masks
    
    Returns:
        IoU score
    """
    def compute_iou_single(y_true_single, y_pred_single, class_id_single, exclude_empty=False):
        y_true_class = (y_true_single == class_id_single).float()
        y_pred_class = (y_pred_single == class_id_single).float()
        
        intersection = torch.logical_and(y_true_class, y_pred_class)
        union = torch.logical_or(y_true_class, y_pred_class)
        
        union_sum = torch.sum(union)
        if union_sum == 0:
            # Both prediction and ground truth are empty
            if exclude_empty:
                return torch.tensor(float('nan'))
            else:
                return torch.tensor(1.0)
        else:
            return torch.sum(intersection).float() / union_sum.float()
    
    if reduction == 'micro':
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        return compute_iou_single(y_true, y_pred, class_id, exclude_empty).item()
    
    elif reduction == 'micro_image_wise':
        iou_scores = torch.stack([
            compute_iou_single(y, p, class_id, exclude_empty) 
            for y, p in zip(y_true, y_pred)
        ])
        if exclude_empty:
            # Filter out NaN values when excluding empty masks
            valid_scores = iou_scores[~torch.isnan(iou_scores)]
            if len(valid_scores) == 0:
                return float('nan')
            return torch.mean(valid_scores).item()
        else:
            return torch.mean(iou_scores).item()
    
    else:
        raise ValueError(f"Unknown reduction method: {reduction}")


def compute_dice(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    class_id: int = 1,
    reduction: str = 'mean'
) -> float:
    """
    Compute Dice coefficient for a specific class.
    
    Args:
        y_true: Ground truth tensor
        y_pred: Prediction tensor
        class_id: Class ID to compute Dice for
        reduction: Reduction method
    
    Returns:
        Dice score
    """
    y_true_class = (y_true == class_id).float()
    y_pred_class = (y_pred == class_id).float()
    
    intersection = torch.sum(y_true_class * y_pred_class)
    union = torch.sum(y_true_class) + torch.sum(y_pred_class)
    
    dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
    
    return dice.item()


def compute_iou_imagewise_from_cumulator(
    TPs: List[torch.Tensor],
    FPs: List[torch.Tensor],
    FNs: List[torch.Tensor],
    TNs: List[torch.Tensor],
    exclude_empty: bool = False,
    exclude_empty_only_gt: bool = False,
    return_std: bool = False
):
    """Compute image-wise IoU from accumulated metrics."""
    # Concatenate tensors for each metric
    try:
        tp = torch.cat([tp for tp in TPs])
        fp = torch.cat([fp for fp in FPs])
        fn = torch.cat([fn for fn in FNs])
        tn = torch.cat([tn for tn in TNs])
    except:
        tp = TPs
        fp = FPs
        fn = FNs
        tn = TNs
    

    if return_std:

        mean_iou, std_iou = compute_iou_from_metrics(tp, fp, tn, fn, reduction='micro-imagewise',exclude_empty=exclude_empty, exclude_empty_only_gt =exclude_empty_only_gt, return_std=return_std)
        return mean_iou.item(), std_iou.item()

    else:

        return compute_iou_from_metrics(tp, fp, tn, fn, reduction='micro-imagewise',exclude_empty=exclude_empty).item()


def compute_dice_imagewise_from_cumulator(
    TPs: List[torch.Tensor],
    FPs: List[torch.Tensor],
    FNs: List[torch.Tensor],
    TNs: List[torch.Tensor],
    exclude_empty: bool = False,
    exclude_empty_only_gt: bool = False,
    return_std: bool = False
):
    """Compute image-wise Dice from accumulated metrics."""
    # Concatenate tensors for each metric
    try:
        tp = torch.cat([tp for tp in TPs])
        fp = torch.cat([fp for fp in FPs])
        fn = torch.cat([fn for fn in FNs])
        tn = torch.cat([tn for tn in TNs])
    except:
        tp = TPs
        fp = FPs
        fn = FNs
        tn = TNs
    
    
    if return_std:
        mean_dice, std_dice =  compute_dice_from_metrics(tp, fp, tn, fn, reduction='micro-imagewise',exclude_empty=exclude_empty, exclude_empty_only_gt=exclude_empty_only_gt, return_std=return_std)
        return mean_dice.item(), std_dice.item()

    else:

        return compute_dice_from_metrics(tp, fp, tn, fn, reduction='micro-imagewise',exclude_empty=exclude_empty).item()


def compute_mean_iou_imagewise_from_cumulator(
    TPs: List[torch.Tensor],
    FPs: List[torch.Tensor],
    FNs: List[torch.Tensor],
    TNs: List[torch.Tensor],
    exclude_empty: bool = False,
    return_std: bool = False,
    reduce_mean: bool = True
):
    """Compute mean IoU across classes from accumulated metrics."""
    # Concatenate tensors for each metric
    try:
        tp = torch.cat([tp for tp in TPs])
        fp = torch.cat([fp for fp in FPs])
        fn = torch.cat([fn for fn in FNs])
        tn = torch.cat([tn for tn in TNs])
    except:
        tp = TPs
        fp = FPs
        fn = FNs
        tn = TNs

    if exclude_empty:
        # Calculate IOU per image excluding empty cases
        iou1_per_image_no_empty = compute_iou_from_metrics(tp, fp, tn, fn, reduction='none', exclude_empty=True)
        iou0_per_image_no_empty = compute_iou_from_metrics(tn, fn, tp, fp, reduction='none', exclude_empty=True)
        
        # Combine and filter valid IOU scores
        combined_iou_scores = np.hstack((iou0_per_image_no_empty.cpu().numpy(), iou1_per_image_no_empty.cpu().numpy()))
        valid_pairs = ~np.isnan(combined_iou_scores).any(axis=1)
        
        # Compute mean and optionally standard deviation
        mean_iou_per_image_no_empty = np.nanmean(combined_iou_scores[valid_pairs], axis=1)

        if not reduce_mean:
            return mean_iou_per_image_no_empty
        if return_std:
            std_iou_per_image_no_empty = np.nanstd(mean_iou_per_image_no_empty)
            return np.mean(mean_iou_per_image_no_empty), std_iou_per_image_no_empty
        else:
            return np.mean(mean_iou_per_image_no_empty)

    else:
        # Calculate IOU per image including empty cases
        iou1_per_image = compute_iou_from_metrics(tp, fp, tn, fn, reduction='none')
        iou0_per_image = compute_iou_from_metrics(tn, fn, tp, fp, reduction='none')
        
        # Compute mean and optionally standard deviation
        combined_iou_scores = np.array([iou0_per_image.cpu().numpy(), iou1_per_image.cpu().numpy()])
        mean_iou_per_image = np.nanmean(combined_iou_scores, axis=0)
        
        if not reduce_mean:
            return mean_iou_per_image
        if return_std:
            std_iou_per_image = np.nanstd(mean_iou_per_image)
            return np.mean(mean_iou_per_image), std_iou_per_image
        else:
            return np.mean(mean_iou_per_image)


def compute_mean_dice_imagewise_from_cumulator(
    TPs: List[torch.Tensor],
    FPs: List[torch.Tensor],
    FNs: List[torch.Tensor],
    TNs: List[torch.Tensor],
    exclude_empty: bool = False,
    return_std: bool = False,
    reduce_mean: bool = True
):
    """Compute mean Dice across classes from accumulated metrics."""
    # Concatenate tensors for each metric
    try:
        tp = torch.cat([tp for tp in TPs])
        fp = torch.cat([fp for fp in FPs])
        fn = torch.cat([fn for fn in FNs])
        tn = torch.cat([tn for tn in TNs])
    except:
        tp = TPs
        fp = FPs
        fn = FNs
        tn = TNs

    if exclude_empty:
        # Calculate Dice per image excluding empty cases
        dice1_per_image_no_empty = compute_dice_from_metrics(tp, fp, tn, fn, reduction='none', exclude_empty=True)
        dice0_per_image_no_empty = compute_dice_from_metrics(tn, fn, tp, fp, reduction='none', exclude_empty=True)
        
        # Combine and filter valid Dice scores
        combined_dice_scores = np.hstack((dice0_per_image_no_empty.cpu().numpy(), dice1_per_image_no_empty.cpu().numpy()))
        valid_pairs = ~np.isnan(combined_dice_scores).any(axis=1)
        
        # Compute mean and optionally standard deviation
        mean_dice_per_image_no_empty = np.nanmean(combined_dice_scores[valid_pairs], axis=1)

        if not reduce_mean:
            return mean_dice_per_image_no_empty
        if return_std:
            std_dice_per_image_no_empty = np.nanstd(mean_dice_per_image_no_empty)
            return np.mean(mean_dice_per_image_no_empty), std_dice_per_image_no_empty
        else:
            return np.mean(mean_dice_per_image_no_empty)

    else:
        # Calculate Dice per image including empty cases
        dice1_per_image = compute_dice_from_metrics(tp, fp, tn, fn, reduction='none')
        dice0_per_image = compute_dice_from_metrics(tn, fn, tp, fp, reduction='none')
        
        # Compute mean and optionally standard deviation
        combined_dice_scores = np.array([dice0_per_image.cpu().numpy(), dice1_per_image.cpu().numpy()])
        mean_dice_per_image = np.nanmean(combined_dice_scores, axis=0)
        
        if not reduce_mean:
            return mean_dice_per_image
        if return_std:
            std_dice_per_image = np.nanstd(mean_dice_per_image)
            return np.mean(mean_dice_per_image), std_dice_per_image
        else:
            return np.mean(mean_dice_per_image)


def compute_volumetric_iou(
    gt_volume: torch.Tensor,
    pred_volume: torch.Tensor,
    num_classes: int = 2,
    exclude_empty: bool = False
) -> List[float]:
    """
    Compute volumetric IoU for each class.
    
    Args:
        gt_volume: Ground truth volume
        pred_volume: Predicted volume
        num_classes: Number of classes
        exclude_empty: Whether to exclude empty volumes
    
    Returns:
        List of IoU scores for each class
    """
    ious = []
    
    for class_id in range(num_classes):
        gt_class = (gt_volume == class_id).float()
        pred_class = (pred_volume == class_id).float()
        
        intersection = torch.sum(gt_class * pred_class)
        union = torch.sum(gt_class) + torch.sum(pred_class) - intersection
        
        if exclude_empty and union == 0:
            continue
            
        iou = (intersection / (union + 1e-7)).item()
        ious.append(iou)
    
    return ious


def compute_dice_score(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    class_id: int = 1,
    reduction: str = 'micro',
    exclude_empty: bool = False
) -> float:
    """
    Compute Dice score for a specific class.
    
    Args:
        y_true: Ground truth tensor
        y_pred: Prediction tensor
        class_id: Class ID to compute Dice for
        reduction: Reduction method ('micro', 'micro_image_wise')
        exclude_empty: Whether to exclude empty masks
    
    Returns:
        Dice score
    """
    
    def compute_dice_score_single(y_true_single, y_pred_single, class_id_single, exclude_empty=False):
        y_true_class = (y_true_single == class_id_single).float()
        y_pred_class = (y_pred_single == class_id_single).float()

        intersection = torch.sum(y_true_class * y_pred_class)
        union = torch.sum(y_true_class) + torch.sum(y_pred_class)

        if union == 0:
            if exclude_empty:
                dice_score = torch.tensor(float('nan'))
            else:
                dice_score = torch.tensor(1.0)
        else:
            dice_score = (2. * intersection) / (union)
        
        return dice_score

    if reduction == 'micro':
        y_true_flat = y_true.view(-1)
        y_pred_flat = y_pred.view(-1)
        dice_score = torch.tensor(compute_dice_score_single(y_true_flat, y_pred_flat, class_id, exclude_empty)).float()
        return dice_score

    elif reduction == 'micro_image_wise':
        dice_scores = torch.tensor([compute_dice_score_single(y, p, class_id, exclude_empty) for y, p in zip(y_true, y_pred)], dtype=torch.float32)
        return torch.nanmean(dice_scores)

    else:
        raise ValueError("Reduction method should be either 'micro' or 'micro_image_wise'")


def compute_mean_precision(tp, fp, fn, tn):
    """
    Compute the mean precision for binary classification across two classes.

    Args:
        tp (torch.Tensor): True Positives, tensor of shape (B, 1).
        fp (torch.Tensor): False Positives, tensor of shape (B, 1).
        tn (torch.Tensor): True Negatives, tensor of shape (B, 1).
        fn (torch.Tensor): False Negatives, tensor of shape (B, 1).

    Returns:
        torch.Tensor: The mean precision across classes.
    """

    # Precision for class 1
    precision_class_1 = torch.div(tp, tp + fp)
    precision_class_1[torch.isnan(precision_class_1)] = 1

    # Precision for class 0 (inverting perspective)
    precision_class_0 = torch.div(tn, tn + fn)
    precision_class_0[torch.isnan(precision_class_0)] = 1

    # Mean precision across both classes
    mean_precision = (precision_class_1 + precision_class_0) / 2

    # Average across the batch
    mean_precision = torch.mean(mean_precision)

    return mean_precision


def compute_mean_recall(tp, fp, fn, tn):
    """
    Compute the mean recall for binary classification across two classes.

    Args:
        tp (torch.Tensor): True Positives, tensor of shape (B, 1).
        fp (torch.Tensor): False Positives, tensor of shape (B, 1).
        tn (torch.Tensor): True Negatives, tensor of shape (B, 1).
        fn (torch.Tensor): False Negatives, tensor of shape (B, 1).

    Returns:
        torch.Tensor: The mean recall across classes.
    """
    recall_class_1 = torch.div(tp, tp + fn)
    recall_class_1[torch.isnan(recall_class_1)] = 1

    recall_class_0 = torch.div(tn, tn + fp)
    recall_class_0[torch.isnan(recall_class_0)] = 1

    mean_recall = (recall_class_1 + recall_class_0) / 2
    mean_recall = torch.mean(mean_recall)

    return mean_recall


def compute_dice_score_from_cm(tp, fp, fn, tn, reduction='micro', exclude_empty=False):
    # Convert to float for division
    tp = tp.float()
    fp = fp.float()
    fn = fn.float()
    
    if reduction == 'micro':
        # Sum across all classes and samples for micro averaging
        tp_sum = tp.sum()
        fp_sum = fp.sum()
        fn_sum = fn.sum()
        
        # Compute Dice score, handling division by zero
        denominator = 2 * tp_sum + fp_sum + fn_sum
        dice_score = 2 * tp_sum / denominator if denominator != 0 else torch.tensor(1.0)
        
    elif reduction == 'micro-imagewise':
        # Compute Dice Score per sample, then average across samples
        denominator = 2 * tp + fp + fn
        valid = denominator != 0
        dice_scores = torch.zeros_like(tp)
        dice_scores[valid] = 2 * tp[valid] / denominator[valid]

        if exclude_empty:
            dice_scores[~valid] = torch.tensor(float('nan'))
            dice_score = dice_scores.nanmean(dim=0)
        else:
            dice_scores[~valid] = torch.tensor(1.0)
            dice_score = dice_scores.mean(dim=0)
        
    else:
        raise ValueError("Reduction method must be either 'micro' or 'micro-imagewise'")
    
    return dice_score


def class_specific_accuracy_score(preds, targets, class_id=1, eps=1e-7, reduction='mean', averaging='micro'):
    """
    Compute the class-specific accuracy score.

    Parameters
    ----------
    preds : torch.Tensor
        The predicted values from the model. Assumes values are [0, 1].
    targets : torch.Tensor
        The ground truth values. Assumes values are in {0, 1}.
    class_id : int, optional
        The class for which to compute the accuracy. Assumes values are in {0, 1}. Default is 1.
    eps : float, optional
        Small value to prevent division by zero. Default is 1e-7.
    reduction : str, optional
        Specifies the reduction to apply to the output: 'mean', 'sum' or 'none'.
        'none': no reduction will be applied.
        'mean': the sum of the output will be divided by the number of elements in the output.
        'sum': the output will be summed. Default is 'mean'.
    averaging : str, optional
        Specifies the type of averaging to use: 'micro' or 'micro_image_wise'.
        'micro': Calculate metrics globally by counting the total true positives, false negatives, and false positives.
        'micro_image_wise': Calculate metrics for each instance, and find their average.

    Returns
    -------
    score : torch.Tensor
        The computed class-specific accuracy score.
    """

    if class_id == 1:
        # Compute True Positive (TP), False Positive (FP) and False Negative (FN)
        TP = (preds * targets).sum(dim=(1, 2, 3))
        FP = (preds * (1 - targets)).sum(dim=(1, 2, 3))
        FN = ((1 - preds) * targets).sum(dim=(1, 2, 3))
    else:
        # Compute True Negative (TN), False Positive (FP) and False Negative (FN)
        TN = ((1 - preds) * (1 - targets)).sum(dim=(1, 2, 3))
        FP = ((1 - preds) * targets).sum(dim=(1, 2, 3))
        FN = (preds * (1 - targets)).sum(dim=(1, 2, 3))

    if averaging == 'micro':
        if class_id == 1:
            accuracy = TP.sum() / (TP.sum() + FP.sum() + FN.sum() + eps)
        else:
            accuracy = TN.sum() / (TN.sum() + FP.sum() + FN.sum() + eps)
    elif averaging == 'micro_image_wise':
        if class_id == 1:
            accuracy = torch.mean(TP / (TP + FP + FN + eps))
        else:
            accuracy = torch.mean(TN / (TN + FP + FN + eps))
    else:
        raise ValueError("Averaging method must be either 'micro' or 'micro_image_wise'")

    return accuracy
