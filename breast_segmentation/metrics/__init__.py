"""Evaluation metrics and loss functions."""

from .losses import (
    AsymmetricFocalLoss, AsymmetricFocalTverskyLoss, AsymmetricUnifiedFocalLoss,
    SoftDiceLoss, FocalTverskyIOULoss, SurfaceLossBinary,
    CABFL, CrossEntropy2d, compute_class_weight, get_loss_function
)
from .evaluation import (
    compute_metrics_from_confusion_matrix,
    compute_iou, compute_dice,
    compute_iou_from_metrics, compute_dice_from_metrics,
    compute_iou_imagewise_from_cumulator,
    compute_dice_imagewise_from_cumulator,
    compute_mean_iou_imagewise_from_cumulator,
    compute_mean_dice_imagewise_from_cumulator,
    compute_volumetric_iou
)
from .volume_metrics import (
    filter_masses,
    remove_far_masses_based_on_largest_mass,
    compute_iou_npy, compute_dice_score_npy,
    calculate_mass_detection_imagewise_volume,
    compute_accuracy_from_cumulator,
    compute_mean_precision_from_cumulator,
    compute_mean_recall_from_cumulator,
    compute_precision_from_cumulator,
    compute_recall_from_cumulator,
    compute_f1_from_cumulator,
    compute_accuracy_excluding_cases,
    compute_precision_excluding_cases_from_cumulator,
    compute_recall_excluding_cases_from_cumulator,
    compute_f1_excluding_cases_from_cumulator
)

__all__ = [
    # Losses
    'AsymmetricFocalLoss', 'AsymmetricFocalTverskyLoss', 'AsymmetricUnifiedFocalLoss',
    'SoftDiceLoss', 'FocalTverskyIOULoss', 'SurfaceLossBinary',
    'CABFL', 'CrossEntropy2d', 'compute_class_weight', 'get_loss_function',
    # Metrics
    'compute_metrics_from_confusion_matrix',
    'compute_iou', 'compute_dice',
    'compute_iou_from_metrics', 'compute_dice_from_metrics',
    'compute_iou_imagewise_from_cumulator',
    'compute_dice_imagewise_from_cumulator',
    'compute_mean_iou_imagewise_from_cumulator',
    'compute_mean_dice_imagewise_from_cumulator',
    'compute_volumetric_iou',
    # Volume metrics
    'filter_masses',
    'remove_far_masses_based_on_largest_mass',
    'compute_iou_npy', 'compute_dice_score_npy',
    'calculate_mass_detection_imagewise_volume',
    'compute_accuracy_from_cumulator',
    'compute_mean_precision_from_cumulator',
    'compute_mean_recall_from_cumulator',
    'compute_precision_from_cumulator',
    'compute_recall_from_cumulator',
    'compute_f1_from_cumulator',
    'compute_accuracy_excluding_cases',
    'compute_precision_excluding_cases_from_cumulator',
    'compute_recall_excluding_cases_from_cumulator',
    'compute_f1_excluding_cases_from_cumulator'
]