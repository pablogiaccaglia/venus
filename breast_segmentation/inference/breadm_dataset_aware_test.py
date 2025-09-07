"""Dataset-aware test functions using backend metric functions and reference implementation patterns."""

import torch
import numpy as np
import monai
import segmentation_models_pytorch as smp
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any

from ..models import BreastSegmentationModel, BreastFusionModel
from ..data import PairedDataset
from ..utils import reverse_transformations
from ..metrics import (
    compute_mean_iou_imagewise_from_cumulator,
    compute_mean_dice_imagewise_from_cumulator,
    compute_iou_imagewise_from_cumulator,
    compute_dice_imagewise_from_cumulator,
    compute_accuracy_from_cumulator,
    compute_precision_from_cumulator,
    compute_recall_from_cumulator,
    compute_f1_from_cumulator,
    compute_accuracy_excluding_cases,
    compute_precision_excluding_cases_from_cumulator,
    compute_recall_excluding_cases_from_cumulator,
    compute_f1_excluding_cases_from_cumulator,
    calculate_mass_detection_imagewise_volume,
    filter_masses,
)
from ..utils.postprocessing import fuse_segmentations


def test_dataset_aware_no_patches(
    model_path: str,
    patient_ids: List[str],
    datasets: Dict,
    dataset_key: str,
    filter: bool = False,
    get_scores_for_statistics: bool = False,
    get_only_masses: bool = False,
    arch_name: Optional[str] = None,
    strict: bool = False,
    subtracted: bool = True,
) -> Dict:
    if arch_name:
        model = BreastSegmentationModel.load_from_checkpoint(
            model_path, strict=strict, arch=arch_name
        )
    else:
        model = BreastSegmentationModel.load_from_checkpoint(model_path, strict=strict)

    model_class_mean_iou = []
    model_class_mean_dice = []
    model_detection_iou = []

    model_iou_mass_volume = []
    model_iou_mass_volume_no_empty = []

    model_dice_mass_volume = []
    model_dice_mass_volume_no_empty = []

    model_accuracy = []
    model_precision = []
    model_recall = []

    TP = []
    FP = []
    FN = []
    TN = []
    detection_iou = []

    im_path_key = "subtracted_filename_or_obj" if subtracted else "filename_or_obj"
    loadimage = monai.transforms.LoadImage(
        ensure_channel_first=True,
        reader=monai.data.PILReader(converter=lambda image: image.convert("L")),
    )

    for patient_id in patient_ids:

        predicted_label_slices = []
        gt_label_slices = []
        image_slices = []

        print(patient_id)
        dataset = datasets[patient_id][dataset_key]

        for idx, e in tqdm(enumerate(dataset), total=len(dataset)):
            original_image = loadimage(e["image_meta_dict"][im_path_key])
            original_image = monai.transforms.Rotate90()(original_image)

            pred_label = torch.zeros(original_image.shape, dtype=torch.uint8)

            gt_label = loadimage(e["label_meta_dict"][im_path_key])
            gt_label = monai.transforms.Rotate90()(gt_label)
            gt_label = gt_label / 255.0

            if True:  # e['keep_sample']:
                image = torch.unsqueeze(e["image"], 0)

                with torch.no_grad():
                    model = model.to("cuda")
                    model.eval()

                    masks = model(image.to("cuda"))
                    if isinstance(masks, List) or isinstance(masks, Tuple):
                        masks = masks[0]

                    masks = masks.sigmoid()

                pred_label = masks[0]
                pred_label = (pred_label > 0.4).int()
                pred_label = reverse_transformations(dataset[idx], pred_label, mode="whole")

                pred_label = monai.transforms.Resize(
                    spatial_size=(original_image.shape[1], original_image.shape[2]),
                    mode="nearest-exact",
                )(pred_label)

            if not filter:
                tp, fp, fn, tn = smp.metrics.get_stats(
                    torch.tensor(np.expand_dims(pred_label, 0).astype(int)),
                    torch.tensor(np.expand_dims(gt_label, 0).astype(int)),
                    mode="binary",
                )
                TP.append(tp)
                FP.append(fp)
                FN.append(fn)
                TN.append(tn)

            """plt.figure(figsize=(15, 10))
    
            plt.subplot(2, 2, 1)
            plt.imshow(original_image.squeeze(),  cmap='gray')  # convert CHW -> HWC
            plt.title("Image")
            plt.axis("off")
        
            plt.subplot(2, 2, 2)
            plt.imshow(label_whole, cmap='gray') # just squeeze classes dim, because we have only one class
            plt.title("Whole")
            plt.axis("off")
        
            plt.subplot(2, 2, 3)
            plt.imshow(label_patches, cmap='gray') # just squeeze classes dim, because we have only one class
            plt.title("Patch")
            plt.axis("off")
    
            plt.subplot(2, 2, 4)
            plt.imshow(gt_label, cmap='gray') # just squeeze classes dim, because we have only one class
            plt.title("GT")
            plt.axis("off")
        
            plt.show()
    
            plt.imshow(fusion , cmap='gray')
            plt.show()"""

            predicted_label_slices.append(pred_label.squeeze())
            gt_label_slices.append(gt_label.squeeze())
            image_slices.append(original_image.squeeze())

        predicted_label_volume = np.stack(
            predicted_label_slices, axis=-1
        )  # Stack along the first axis to create a 3D volume

        gt_label_volume = np.stack(gt_label_slices, axis=-1)
        images_volume = np.stack(image_slices, axis=-1)

        if filter:
            predicted_label_volume = filter_masses(
                predicted_label_volume, min_slices=3, window_size=3
            )  # H x W x N
            # H x W x N -> N x H x W -> N x 1 x H x W
            predicted_label_volume_for_stats = np.transpose(predicted_label_volume, (2, 0, 1))
            predicted_label_volume_for_stats = np.expand_dims(
                predicted_label_volume_for_stats, 1
            )  # N x 1 x H x W

            gt_label_volume_for_stats = np.transpose(gt_label_volume, (2, 0, 1))
            gt_label_volume_for_stats = np.expand_dims(
                gt_label_volume_for_stats, 1
            )  # N x 1 x H x W

            tp, fp, fn, tn = smp.metrics.get_stats(
                torch.tensor(predicted_label_volume_for_stats.astype(int)),
                torch.tensor(gt_label_volume_for_stats.astype(int)),
                mode="binary",
            )
            TP += [torch.tensor([[elem]]) for elem in tp.squeeze()]
            FP += [torch.tensor([[elem]]) for elem in fp.squeeze()]
            FN += [torch.tensor([[elem]]) for elem in fn.squeeze()]
            TN += [torch.tensor([[elem]]) for elem in tn.squeeze()]

        detection_iou += calculate_mass_detection_imagewise_volume(
            predicted_label_volume.astype(int), gt_label_volume
        )

    model_detection_iou = np.array(detection_iou).mean()
    model_detection_iou_std = np.array(detection_iou).std()

    model_class_mean_iou, model_class_std_iou = compute_mean_iou_imagewise_from_cumulator(
        TP, FP, FN, TN, exclude_empty=True, return_std=True
    )
    model_class_mean_dice, model_class_std_dice = compute_mean_dice_imagewise_from_cumulator(
        TP, FP, FN, TN, exclude_empty=True, return_std=True
    )

    model_iou_mass_volume, model_iou_mass_volume_std = compute_iou_imagewise_from_cumulator(
        TP, FP, FN, TN, exclude_empty=False, return_std=True
    )
    model_iou_mass_volume_no_empty, model_iou_mass_volume_no_empty_std = (
        compute_iou_imagewise_from_cumulator(TP, FP, FN, TN, exclude_empty=True, return_std=True)
    )
    model_iou_mass_volume_no_empty_optimistic, model_iou_mass_volume_no_empty_optimistic_std = (
        compute_iou_imagewise_from_cumulator(
            TP, FP, FN, TN, exclude_empty=True, exclude_empty_only_gt=True, return_std=True
        )
    )

    model_dice_mass_volume, model_dice_mass_volume_std = compute_dice_imagewise_from_cumulator(
        TP, FP, FN, TN, exclude_empty=False, return_std=True
    )
    model_dice_mass_volume_no_empty, model_dice_mass_volume_no_empty_std = (
        compute_dice_imagewise_from_cumulator(TP, FP, FN, TN, exclude_empty=True, return_std=True)
    )
    model_dice_mass_volume_no_empty_optimistic, model_dice_mass_volume_no_empty_optimistic_std = (
        compute_dice_imagewise_from_cumulator(
            TP, FP, FN, TN, exclude_empty=True, exclude_empty_only_gt=True, return_std=True
        )
    )

    model_mean_accuracy_no_empty, model_mean_accuracy_no_empty_std = (
        compute_accuracy_from_cumulator(
            TP, FP, FN, TN, exclude_empty=True, is_mean=True, return_std=True
        )
    )
    model_mean_precision_no_empty, model_mean_precision_no_empty_std = (
        compute_precision_from_cumulator(
            TP, FP, FN, TN, exclude_empty=True, is_mean=True, return_std=True
        )
    )
    model_mean_recall_no_empty, model_mean_recall_no_empty_std = compute_recall_from_cumulator(
        TP, FP, FN, TN, exclude_empty=True, is_mean=True, return_std=True
    )
    model_mean_f1_no_empty, model_mean_f1_no_empty_std = compute_f1_from_cumulator(
        TP, FP, FN, TN, exclude_empty=True, is_mean=True, return_std=True
    )

    model_accuracy_excluding_cases, model_accuracy_excluding_cases_std = (
        compute_accuracy_excluding_cases(TP, FP, FN, TN, return_std=True)
    )
    model_precision_excluding_cases, model_precision_excluding_cases_std = (
        compute_precision_excluding_cases_from_cumulator(TP, FP, FN, TN, return_std=True)
    )
    model_recall_excluding_cases, model_recall_excluding_cases_std = (
        compute_recall_excluding_cases_from_cumulator(TP, FP, FN, TN, return_std=True)
    )

    model_accuracy_no_empty, model_accuracy_no_empty_std = compute_accuracy_from_cumulator(
        TP, FP, FN, TN, exclude_empty=True, is_mean=False, return_std=True
    )
    model_precision_no_empty, model_precision_no_empty_std = (
        compute_precision_excluding_cases_from_cumulator(
            TP, FP, FN, TN, return_std=True, exclude_only_zero_denominator=True
        )
    )
    model_recall_no_empty, model_recall_no_empty_std = (
        compute_recall_excluding_cases_from_cumulator(
            TP, FP, FN, TN, return_std=True, exclude_only_zero_denominator=True
        )
    )

    model_f1_no_empty, model_f1_no_empty_std = compute_f1_excluding_cases_from_cumulator(
        TP, FP, FN, TN, return_std=True, exclude_only_zero_denominator=True
    )

    print("MODEL CLASS MEAN IOU ", model_class_mean_iou)
    print("MODEL CLASS STD IOU ", model_class_std_iou)
    print()
    print("MODEL CLASS MEAN DICE ", model_class_mean_dice)
    print("MODEL CLASS STD DICE ", model_class_std_dice)
    print()
    print("MODEL DIOU", model_detection_iou)
    print("MODEL DIOU STD ", model_detection_iou_std)
    print()
    print("MODEL IOU MASS VOLUME ", model_iou_mass_volume)
    print("MODEL IOU MASS VOLUME STD ", model_iou_mass_volume_std)
    print()
    print("MODEL IOU MASS VOLUME NO EMPTY ", model_iou_mass_volume_no_empty)
    print("MODEL IOU MASS VOLUME NO EMPTY STD ", model_iou_mass_volume_no_empty_std)
    print()
    print("MODEL IOU MASS VOLUME NO EMPTY OPTIMISTIC ", model_iou_mass_volume_no_empty_optimistic)
    print(
        "MODEL IOU MASS VOLUME NO EMPTY OPTIMISTIC STD ",
        model_iou_mass_volume_no_empty_optimistic_std,
    )

    print("MODEL DICE MASS VOLUME ", model_dice_mass_volume)
    print("MODEL DICE MASS VOLUME STD ", model_dice_mass_volume_std)
    print()
    print("MODEL DICE MASS VOLUME NO EMPTY ", model_dice_mass_volume_no_empty)
    print("MODEL DICE MASS VOLUME NO EMPTY STD ", model_dice_mass_volume_no_empty_std)
    print()
    print("MODEL DICE MASS VOLUME NO EMPTY OPTIMISTIC ", model_dice_mass_volume_no_empty_optimistic)
    print(
        "MODEL DICE MASS VOLUME NO EMPTY OPTIMISTIC STD ",
        model_dice_mass_volume_no_empty_optimistic_std,
    )
    print()
    print("MODEL MEAN ACCURACY NO EMPTY", model_mean_accuracy_no_empty)
    print("MODEL MEAN ACCURACY NO EMPTY STD", model_mean_accuracy_no_empty_std)
    print()
    print("MODEL MEAN PRECISION NO EMPTY", model_mean_precision_no_empty)
    print("MODEL MEAN PRECISION NO EMPTY STD", model_mean_precision_no_empty_std)
    print()
    print("MODEL MEAN RECALL NO EMPTY", model_mean_recall_no_empty)
    print("MODEL MEAN RECALL NO EMPTY STD", model_mean_recall_no_empty_std)
    print()
    print("MODEL MEAN F1 NO EMPTY", model_mean_f1_no_empty)
    print("MODEL MEAN F1 NO EMPTY STD", model_mean_f1_no_empty_std)
    print()
    print("MODEL ACCURACY EXCLUDING CASES ", model_accuracy_excluding_cases)
    print("MODEL ACCURACY EXCLUDING CASES STD ", model_accuracy_excluding_cases_std)
    print()
    print("MODEL PRECISION EXCLUDING CASES ", model_precision_excluding_cases)
    print("MODEL PRECISION EXCLUDING CASES STD ", model_precision_excluding_cases_std)
    print()
    print("MODEL RECALL EXCLUDING CASES ", model_recall_excluding_cases)
    print("MODEL RECALL EXCLUDING CASES STD ", model_recall_excluding_cases_std)
    print()
    print("MODEL ACCURACY NO EMPTY ", model_accuracy_no_empty)
    print("MODEL ACCURACY NO EMPTY STD ", model_accuracy_no_empty_std)
    print()
    print("MODEL PRECISION NO EMPTY", model_precision_no_empty)
    print("MODEL PRECISION NO EMPTY STD ", model_precision_no_empty_std)
    print()
    print("MODEL RECALL NO EMPTY ", model_recall_no_empty)
    print("MODEL RECALL NO EMPTY STD ", model_recall_no_empty_std)
    print()
    print("MODEL F1 NO EMPTY ", model_f1_no_empty)
    print("MODEL F1 NO EMPTY STD ", model_f1_no_empty_std)
    print()

    if get_scores_for_statistics:
        tp = torch.cat([tp for tp in TP])
        fp = torch.cat([fp for fp in FP])
        fn = torch.cat([fn for fn in FN])
        tn = torch.cat([tn for tn in TN])

        if get_only_masses:
            # Create a mask where tp + fn is not equal to 0
            mask = (tp + fn) != 0

            # Apply this mask to each tensor to filter out the desired values
            tp = tp[mask]
            fp = fp[mask]
            fn = fn[mask]
            tn = tn[mask]

        miou_scores = compute_mean_iou_imagewise_from_cumulator(
            tp, fp, fn, tn, exclude_empty=False, return_std=False, reduce_mean=False
        )
        mdice_scores = compute_mean_dice_imagewise_from_cumulator(
            tp, fp, fn, tn, exclude_empty=False, return_std=False, reduce_mean=False
        )
        mf1_scores = compute_f1_from_cumulator(
            tp, fp, fn, tn, exclude_empty=False, is_mean=True, return_std=False, reduce_mean=False
        )

        scores_dict = {
            "miou": miou_scores.squeeze().tolist(),
            "mdice": mdice_scores.squeeze().tolist(),
            "mf1": mf1_scores.squeeze().tolist(),
        }
        return scores_dict


def test_dataset_aware_fusion(
    model_path: str,
    patient_ids: List[str],
    datasets: Dict,
    whole_dataset_key: str,
    patches_dataset_key: str,
    use_simple_fusion: bool = False,
    use_decoder_attention: bool = True,
    strict: bool = True,
    filter: bool = False,
    subtracted: bool = True,
    get_scores_for_statistics: bool = False,
    get_only_masses: bool = False,
    base_channels: int = 64,
) -> Dict:
    """
    Test fusion model with dataset-aware metric computation.
    """
    # Load model
    model = BreastFusionModel.load_from_checkpoint(
        model_path,
        strict=strict,
        use_simple_fusion=use_simple_fusion,
        use_decoder_attention=use_decoder_attention,
        base_channels=base_channels,
    )

    model_class_mean_iou = []
    model_class_mean_dice = []
    model_detection_iou = []

    model_iou_mass_volume = []
    model_iou_mass_volume_no_empty = []

    model_dice_mass_volume = []
    model_dice_mass_volume_no_empty = []

    model_accuracy = []
    model_precision = []
    model_recall = []

    TP = []
    FP = []
    FN = []
    TN = []

    detection_iou = []

    loadimage = monai.transforms.LoadImage(
        ensure_channel_first=True,
        reader=monai.data.PILReader(converter=lambda image: image.convert("L")),
    )

    im_path_key = "subtracted_filename_or_obj" if subtracted else "filename_or_obj"

    for patient_id in patient_ids:
        predicted_label_slices = []
        gt_label_slices = []
        image_slices = []
        print(patient_id)
        patches_ds = datasets[patient_id][patches_dataset_key]
        whole_image_ds = datasets[patient_id][whole_dataset_key]

        fusion_dataset = PairedDataset(whole_image_ds, patches_ds, augment=False)

        prev_had_mask = False

        for idx, e in tqdm(enumerate(fusion_dataset), total=len(patches_ds)):
            original_image = loadimage(e[0]["image_meta_dict"][im_path_key])
            original_image = monai.transforms.Rotate90()(original_image)

            pred_label = torch.zeros(original_image.shape, dtype=torch.uint8)

            gt_label = loadimage(e[0]["label_meta_dict"][im_path_key])
            gt_label = monai.transforms.Rotate90()(gt_label)
            gt_label = gt_label / 255.0

            if True:  # fusion_dataset[idx][0]['keep_sample']:

                whole_image = torch.unsqueeze(fusion_dataset[idx][0]["image"], 0)
                patch_image2 = torch.unsqueeze(fusion_dataset[idx][1]["image"], 0)
                patch_image3 = torch.unsqueeze(fusion_dataset[idx][2]["image"], 0)

                with torch.no_grad():
                    masks = []
                    # pass to model
                    model = model.to("cuda")
                    model.eval()

                    masks = model(
                        whole_image.to("cuda"), patch_image2.to("cuda"), patch_image3.to("cuda")
                    )
                    masks = masks.sigmoid()

                pred_label = masks[0]
                pred_label = (pred_label > 0.4).int()
                pred_label = reverse_transformations(
                    fusion_dataset[idx][0], pred_label, mode="whole"
                )

            pred_label = monai.transforms.Resize(
                spatial_size=(original_image.shape[1], original_image.shape[2]),
                mode="nearest-exact",
            )(pred_label)

            if not filter:
                tp, fp, fn, tn = smp.metrics.get_stats(
                    torch.tensor(np.expand_dims(pred_label, 0).astype(int)),
                    torch.tensor(np.expand_dims(gt_label, 0).astype(int)),
                    mode="binary",
                )
                TP.append(tp)
                FP.append(fp)
                FN.append(fn)
                TN.append(tn)

            """plt.figure(figsize=(15, 10))
    
            plt.subplot(2, 2, 1)
            plt.imshow(original_image.squeeze(),  cmap='gray')  # convert CHW -> HWC
            plt.title("Image")
            plt.axis("off")
        
            plt.subplot(2, 2, 2)
            plt.imshow(label_whole, cmap='gray') # just squeeze classes dim, because we have only one class
            plt.title("Whole")
            plt.axis("off")
        
            plt.subplot(2, 2, 3)
            plt.imshow(label_patches, cmap='gray') # just squeeze classes dim, because we have only one class
            plt.title("Patch")
            plt.axis("off")
    
            plt.subplot(2, 2, 4)
            plt.imshow(gt_label, cmap='gray') # just squeeze classes dim, because we have only one class
            plt.title("GT")
            plt.axis("off")
        
            plt.show()
    
            plt.imshow(fusion , cmap='gray')
            plt.show()"""

            predicted_label_slices.append(pred_label.squeeze())
            gt_label_slices.append(gt_label.squeeze())
            image_slices.append(original_image.squeeze())

        predicted_label_volume = np.stack(
            predicted_label_slices, axis=-1
        )  # Stack along the first axis to create a 3D volume

        gt_label_volume = np.stack(gt_label_slices, axis=-1)
        images_volume = np.stack(image_slices, axis=-1)

        if filter:
            predicted_label_volume = filter_masses(
                predicted_label_volume, min_slices=3, window_size=3
            )  # H x W x N
            # H x W x N -> N x H x W -> N x 1 x H x W
            predicted_label_volume_for_stats = np.transpose(predicted_label_volume, (2, 0, 1))
            predicted_label_volume_for_stats = np.expand_dims(
                predicted_label_volume_for_stats, 1
            )  # N x 1 x H x W

            gt_label_volume_for_stats = np.transpose(gt_label_volume, (2, 0, 1))
            gt_label_volume_for_stats = np.expand_dims(
                gt_label_volume_for_stats, 1
            )  # N x 1 x H x W

            tp, fp, fn, tn = smp.metrics.get_stats(
                torch.tensor(predicted_label_volume_for_stats.astype(int)),
                torch.tensor(gt_label_volume_for_stats.astype(int)),
                mode="binary",
            )
            TP += [torch.tensor([[elem]]) for elem in tp.squeeze()]
            FP += [torch.tensor([[elem]]) for elem in fp.squeeze()]
            FN += [torch.tensor([[elem]]) for elem in fn.squeeze()]
            TN += [torch.tensor([[elem]]) for elem in tn.squeeze()]

        detection_iou += calculate_mass_detection_imagewise_volume(
            predicted_label_volume.astype(int), gt_label_volume
        )

    model_detection_iou = np.array(detection_iou).mean()
    model_detection_iou_std = np.array(detection_iou).std()

    model_class_mean_iou, model_class_std_iou = compute_mean_iou_imagewise_from_cumulator(
        TP, FP, FN, TN, exclude_empty=True, return_std=True
    )
    model_class_mean_dice, model_class_std_dice = compute_mean_dice_imagewise_from_cumulator(
        TP, FP, FN, TN, exclude_empty=True, return_std=True
    )

    model_iou_mass_volume, model_iou_mass_volume_std = compute_iou_imagewise_from_cumulator(
        TP, FP, FN, TN, exclude_empty=False, return_std=True
    )
    model_iou_mass_volume_no_empty, model_iou_mass_volume_no_empty_std = (
        compute_iou_imagewise_from_cumulator(TP, FP, FN, TN, exclude_empty=True, return_std=True)
    )
    model_iou_mass_volume_no_empty_optimistic, model_iou_mass_volume_no_empty_optimistic_std = (
        compute_iou_imagewise_from_cumulator(
            TP, FP, FN, TN, exclude_empty=True, exclude_empty_only_gt=True, return_std=True
        )
    )

    model_dice_mass_volume, model_dice_mass_volume_std = compute_dice_imagewise_from_cumulator(
        TP, FP, FN, TN, exclude_empty=False, return_std=True
    )
    model_dice_mass_volume_no_empty, model_dice_mass_volume_no_empty_std = (
        compute_dice_imagewise_from_cumulator(TP, FP, FN, TN, exclude_empty=True, return_std=True)
    )
    model_dice_mass_volume_no_empty_optimistic, model_dice_mass_volume_no_empty_optimistic_std = (
        compute_dice_imagewise_from_cumulator(
            TP, FP, FN, TN, exclude_empty=True, exclude_empty_only_gt=True, return_std=True
        )
    )

    model_mean_accuracy_no_empty, model_mean_accuracy_no_empty_std = (
        compute_accuracy_from_cumulator(
            TP, FP, FN, TN, exclude_empty=True, is_mean=True, return_std=True
        )
    )
    model_mean_precision_no_empty, model_mean_precision_no_empty_std = (
        compute_precision_from_cumulator(
            TP, FP, FN, TN, exclude_empty=True, is_mean=True, return_std=True
        )
    )
    model_mean_recall_no_empty, model_mean_recall_no_empty_std = compute_recall_from_cumulator(
        TP, FP, FN, TN, exclude_empty=True, is_mean=True, return_std=True
    )
    model_mean_f1_no_empty, model_mean_f1_no_empty_std = compute_f1_from_cumulator(
        TP, FP, FN, TN, exclude_empty=True, is_mean=True, return_std=True
    )

    model_accuracy_excluding_cases, model_accuracy_excluding_cases_std = (
        compute_accuracy_excluding_cases(TP, FP, FN, TN, return_std=True)
    )
    model_precision_excluding_cases, model_precision_excluding_cases_std = (
        compute_precision_excluding_cases_from_cumulator(TP, FP, FN, TN, return_std=True)
    )
    model_recall_excluding_cases, model_recall_excluding_cases_std = (
        compute_recall_excluding_cases_from_cumulator(TP, FP, FN, TN, return_std=True)
    )

    model_accuracy_no_empty, model_accuracy_no_empty_std = compute_accuracy_from_cumulator(
        TP, FP, FN, TN, exclude_empty=True, is_mean=False, return_std=True
    )
    model_precision_no_empty, model_precision_no_empty_std = (
        compute_precision_excluding_cases_from_cumulator(
            TP, FP, FN, TN, return_std=True, exclude_only_zero_denominator=True
        )
    )
    model_recall_no_empty, model_recall_no_empty_std = (
        compute_recall_excluding_cases_from_cumulator(
            TP, FP, FN, TN, return_std=True, exclude_only_zero_denominator=True
        )
    )

    model_f1_no_empty, model_f1_no_empty_std = compute_f1_excluding_cases_from_cumulator(
        TP, FP, FN, TN, return_std=True, exclude_only_zero_denominator=True
    )

    print("MODEL CLASS MEAN IOU ", model_class_mean_iou)
    print("MODEL CLASS STD IOU ", model_class_std_iou)
    print()
    print("MODEL CLASS MEAN DICE ", model_class_mean_dice)
    print("MODEL CLASS STD DICE ", model_class_std_dice)
    print()
    print("MODEL DIOU", model_detection_iou)
    print("MODEL DIOU STD ", model_detection_iou_std)
    print()
    print("MODEL IOU MASS VOLUME ", model_iou_mass_volume)
    print("MODEL IOU MASS VOLUME STD ", model_iou_mass_volume_std)
    print()
    print("MODEL IOU MASS VOLUME NO EMPTY ", model_iou_mass_volume_no_empty)
    print("MODEL IOU MASS VOLUME NO EMPTY STD ", model_iou_mass_volume_no_empty_std)
    print()
    print("MODEL IOU MASS VOLUME NO EMPTY OPTIMISTIC ", model_iou_mass_volume_no_empty_optimistic)
    print(
        "MODEL IOU MASS VOLUME NO EMPTY OPTIMISTIC STD ",
        model_iou_mass_volume_no_empty_optimistic_std,
    )

    print("MODEL DICE MASS VOLUME ", model_dice_mass_volume)
    print("MODEL DICE MASS VOLUME STD ", model_dice_mass_volume_std)
    print()
    print("MODEL DICE MASS VOLUME NO EMPTY ", model_dice_mass_volume_no_empty)
    print("MODEL DICE MASS VOLUME NO EMPTY STD ", model_dice_mass_volume_no_empty_std)
    print()
    print("MODEL DICE MASS VOLUME NO EMPTY OPTIMISTIC ", model_dice_mass_volume_no_empty_optimistic)
    print(
        "MODEL DICE MASS VOLUME NO EMPTY OPTIMISTIC STD ",
        model_dice_mass_volume_no_empty_optimistic_std,
    )
    print()
    print("MODEL MEAN ACCURACY NO EMPTY", model_mean_accuracy_no_empty)
    print("MODEL MEAN ACCURACY NO EMPTY STD", model_mean_accuracy_no_empty_std)
    print()
    print("MODEL MEAN PRECISION NO EMPTY", model_mean_precision_no_empty)
    print("MODEL MEAN PRECISION NO EMPTY STD", model_mean_precision_no_empty_std)
    print()
    print("MODEL MEAN RECALL NO EMPTY", model_mean_recall_no_empty)
    print("MODEL MEAN RECALL NO EMPTY STD", model_mean_recall_no_empty_std)
    print()
    print("MODEL MEAN F1 NO EMPTY", model_mean_f1_no_empty)
    print("MODEL MEAN F1 NO EMPTY STD", model_mean_f1_no_empty_std)
    print()
    print("MODEL ACCURACY EXCLUDING CASES ", model_accuracy_excluding_cases)
    print("MODEL ACCURACY EXCLUDING CASES STD ", model_accuracy_excluding_cases_std)
    print()
    print("MODEL PRECISION EXCLUDING CASES ", model_precision_excluding_cases)
    print("MODEL PRECISION EXCLUDING CASES STD ", model_precision_excluding_cases_std)
    print()
    print("MODEL RECALL EXCLUDING CASES ", model_recall_excluding_cases)
    print("MODEL RECALL EXCLUDING CASES STD ", model_recall_excluding_cases_std)
    print()
    print("MODEL ACCURACY NO EMPTY ", model_accuracy_no_empty)
    print("MODEL ACCURACY NO EMPTY STD ", model_accuracy_no_empty_std)
    print()
    print("MODEL PRECISION NO EMPTY", model_precision_no_empty)
    print("MODEL PRECISION NO EMPTY STD ", model_precision_no_empty_std)
    print()
    print("MODEL RECALL NO EMPTY ", model_recall_no_empty)
    print("MODEL RECALL NO EMPTY STD ", model_recall_no_empty_std)
    print()
    print("MODEL F1 NO EMPTY ", model_f1_no_empty)
    print("MODEL F1 NO EMPTY STD ", model_f1_no_empty_std)
    print()

    if get_scores_for_statistics:
        tp = torch.cat([tp for tp in TP])
        fp = torch.cat([fp for fp in FP])
        fn = torch.cat([fn for fn in FN])
        tn = torch.cat([tn for tn in TN])

        if get_only_masses:
            # Create a mask where tp + fn is not equal to 0
            mask = (tp + fn) != 0

            # Apply this mask to each tensor to filter out the desired values
            tp = tp[mask]
            fp = fp[mask]
            fn = fn[mask]
            tn = tn[mask]

        miou_scores = compute_mean_iou_imagewise_from_cumulator(
            tp, fp, fn, tn, exclude_empty=False, return_std=False, reduce_mean=False
        )
        mdice_scores = compute_mean_dice_imagewise_from_cumulator(
            tp, fp, fn, tn, exclude_empty=False, return_std=False, reduce_mean=False
        )
        mf1_scores = compute_f1_from_cumulator(
            tp, fp, fn, tn, exclude_empty=False, is_mean=True, return_std=False, reduce_mean=False
        )

        scores_dict = {
            "miou": miou_scores.squeeze().tolist(),
            "mdice": mdice_scores.squeeze().tolist(),
            "mf1": mf1_scores.squeeze().tolist(),
        }
        return scores_dict


def test_dataset_aware_ensemble(
    model_whole_path: str,
    model_patches_path: str,
    patient_ids: List[str],
    datasets: Dict,
    whole_dataset_key: str,
    patches_dataset_key: str,
    filter: bool = False,
    get_scores_for_statistics: bool = False,
    get_only_masses: bool = False,
    subtracted: bool = True,
    base_channels: int = 64,
    strict: bool = True,
) -> Dict:
    """
    Test ensemble model with dataset-aware metric computation.
    """
    # Load models
    model_whole = BreastFusionModel.load_from_checkpoint(
        model_whole_path, strict=strict, base_channels=base_channels
    )
    model_patches = BreastSegmentationModel.load_from_checkpoint(model_patches_path, strict=strict)

    model_whole.eval()
    model_patches.eval()
    model_class_mean_iou = []
    model_class_mean_dice = []
    model_detection_iou = []

    model_iou_mass_volume = []
    model_iou_mass_volume_no_empty = []

    model_dice_mass_volume = []
    model_dice_mass_volume_no_empty = []

    model_accuracy = []
    model_precision = []
    model_recall = []

    TP = []
    FP = []
    FN = []
    TN = []

    detection_iou = []

    im_path_key = "subtracted_filename_or_obj" if subtracted else "filename_or_obj"
    loadimage = monai.transforms.LoadImage(
        ensure_channel_first=True,
        reader=monai.data.PILReader(converter=lambda image: image.convert("L")),
    )

    for patient_id in patient_ids:

        predicted_label_slices = []
        gt_label_slices = []
        image_slices = []

        print(patient_id)
        patches_ds = datasets[patient_id][patches_dataset_key]
        whole_image_ds = datasets[patient_id][whole_dataset_key]

        fusion_dataset = PairedDataset(whole_image_ds, patches_ds, augment=False)

        prev_had_mask = False

        for idx, e in tqdm(enumerate(patches_ds), total=len(patches_ds)):
            original_image = loadimage(e[0]["image_meta_dict"][im_path_key])
            original_image = monai.transforms.Rotate90()(original_image)

            merged_label_for_fusion = torch.zeros(original_image.shape)

            gt_label = loadimage(e[0]["label_meta_dict"][im_path_key])
            gt_label = monai.transforms.Rotate90()(gt_label)
            gt_label = gt_label / 255.0

            ## FIRST MODEL
            for elem in e:
                if True:  # elem['keep_sample']:
                    image = torch.unsqueeze(elem["image"], 0)
                    with torch.no_grad():
                        model_patches = model_patches.to("cuda")
                        model_patches.eval()
                        logits = model_patches(image.to("cuda"))[0]

                    pr_mask = logits.sigmoid()

                    if pr_mask.ndim > 3:  # THIS IS THE CASE FOR MODELS RETURNING (BS,1,H,W)
                        pr_mask = pr_mask[0]

                    if pr_mask.sum() > 0:
                        # label = pr_mask
                        label = reverse_transformations(elem, pr_mask, mode="patches")
                        merged_label_for_fusion += label

            original_image = np.transpose(original_image, (1, 2, 0))

            label_patches_for_fusion = merged_label_for_fusion[0]

            # SECOND MODEL

            if (
                True
            ):  # fusion_dataset[idx][0]['keep_sample'] or fusion_dataset[idx][1]['keep_sample'] or fusion_dataset[idx][2]['keep_sample']:

                whole_image = torch.unsqueeze(fusion_dataset[idx][0]["image"], 0)
                patch_image2 = torch.unsqueeze(fusion_dataset[idx][1]["image"], 0)
                patch_image3 = torch.unsqueeze(fusion_dataset[idx][2]["image"], 0)

                with torch.no_grad():
                    masks = []
                    # pass to model
                    model_whole = model_whole.to("cuda")
                    model_whole.eval()

                    masks = model_whole(
                        whole_image.to("cuda"), patch_image2.to("cuda"), patch_image3.to("cuda")
                    )
                    masks = masks.sigmoid()

                label_whole = masks[0]
                label_whole = (label_whole > 0.4).int()
                label_whole = reverse_transformations(
                    whole_image_ds[idx], label_whole, mode="whole"
                )
                label_whole = label_whole.squeeze()

                label_whole_for_fusion = masks[0]
                label_whole_for_fusion = reverse_transformations(
                    whole_image_ds[idx], label_whole_for_fusion, mode="whole"
                )

                # Plot the first image
            else:
                label_whole_for_fusion = torch.zeros(original_image.shape)

            original_image_squeeze = loadimage(e[0]["image_meta_dict"][im_path_key])
            original_image_squeeze = monai.transforms.Rotate90()(original_image)

            fusion = fuse_segmentations(
                label_whole_for_fusion.numpy(),
                label_patches_for_fusion.numpy(),
                prob_threshold=0.4,
                boost_factor=3,
                penalty_factor=0.5,
                kernel_size=50,
            )

            fusion = (fusion > 0.4).astype(int)

            fusion = np.expand_dims(fusion, 0)
            pred_label = fusion

            if not filter:
                tp, fp, fn, tn = smp.metrics.get_stats(
                    torch.tensor(np.expand_dims(fusion, 0).astype(int)),
                    torch.tensor(np.expand_dims(gt_label, 0).astype(int)),
                    mode="binary",
                )
                TP.append(tp)
                FP.append(fp)
                FN.append(fn)
                TN.append(tn)

            """plt.figure(figsize=(15, 10))
    
            plt.subplot(2, 2, 1)
            plt.imshow(original_image.squeeze(),  cmap='gray')  # convert CHW -> HWC
            plt.title("Image")
            plt.axis("off")
        
            plt.subplot(2, 2, 2)
            plt.imshow(((label_whole_for_fusion > 0.4).numpy().astype(int)).squeeze(), cmap='gray') # just squeeze classes dim, because we have only one class
            plt.title("Whole")
            plt.axis("off")
        
            plt.subplot(2, 2, 3)
            plt.imshow(((label_patches_for_fusion > 0.4).numpy().astype(int)).squeeze(), cmap='gray') # just squeeze classes dim, because we have only one class
            plt.title("Patch")
            plt.axis("off")
    
            plt.subplot(2, 2, 4)
            plt.imshow(gt_label.squeeze(), cmap='gray') # just squeeze classes dim, because we have only one class
            plt.title("GT")
            plt.axis("off")
        
            plt.show()
    
            plt.imshow(fusion.squeeze() , cmap='gray')
            plt.show()"""

            predicted_label_slices.append(pred_label.squeeze())
            gt_label_slices.append(gt_label.squeeze())
            image_slices.append(original_image.squeeze())

        predicted_label_volume = np.stack(
            predicted_label_slices, axis=-1
        )  # Stack along the first axis to create a 3D volume

        gt_label_volume = np.stack(gt_label_slices, axis=-1)
        images_volume = np.stack(image_slices, axis=-1)

        if filter:
            print("filtering")
            predicted_label_volume = filter_masses(
                predicted_label_volume, min_slices=3, window_size=5
            )  # H x W x N
            # H x W x N -> N x H x W -> N x 1 x H x W
            predicted_label_volume_for_stats = np.transpose(predicted_label_volume, (2, 0, 1))
            predicted_label_volume_for_stats = np.expand_dims(
                predicted_label_volume_for_stats, 1
            )  # N x 1 x H x W

            gt_label_volume_for_stats = np.transpose(gt_label_volume, (2, 0, 1))
            gt_label_volume_for_stats = np.expand_dims(
                gt_label_volume_for_stats, 1
            )  # N x 1 x H x W

            tp, fp, fn, tn = smp.metrics.get_stats(
                torch.tensor(predicted_label_volume_for_stats.astype(int)),
                torch.tensor(gt_label_volume_for_stats.astype(int)),
                mode="binary",
            )
            TP += [torch.tensor([[elem]]) for elem in tp.squeeze()]
            FP += [torch.tensor([[elem]]) for elem in fp.squeeze()]
            FN += [torch.tensor([[elem]]) for elem in fn.squeeze()]
            TN += [torch.tensor([[elem]]) for elem in tn.squeeze()]

        detection_iou += calculate_mass_detection_imagewise_volume(
            predicted_label_volume.astype(int), gt_label_volume
        )

    model_detection_iou = np.array(detection_iou).mean()
    model_detection_iou_std = np.array(detection_iou).std()

    model_class_mean_iou, model_class_std_iou = compute_mean_iou_imagewise_from_cumulator(
        TP, FP, FN, TN, exclude_empty=True, return_std=True
    )
    model_class_mean_dice, model_class_std_dice = compute_mean_dice_imagewise_from_cumulator(
        TP, FP, FN, TN, exclude_empty=True, return_std=True
    )

    model_iou_mass_volume, model_iou_mass_volume_std = compute_iou_imagewise_from_cumulator(
        TP, FP, FN, TN, exclude_empty=False, return_std=True
    )
    model_iou_mass_volume_no_empty, model_iou_mass_volume_no_empty_std = (
        compute_iou_imagewise_from_cumulator(TP, FP, FN, TN, exclude_empty=True, return_std=True)
    )
    model_iou_mass_volume_no_empty_optimistic, model_iou_mass_volume_no_empty_optimistic_std = (
        compute_iou_imagewise_from_cumulator(
            TP, FP, FN, TN, exclude_empty=True, exclude_empty_only_gt=True, return_std=True
        )
    )

    model_dice_mass_volume, model_dice_mass_volume_std = compute_dice_imagewise_from_cumulator(
        TP, FP, FN, TN, exclude_empty=False, return_std=True
    )
    model_dice_mass_volume_no_empty, model_dice_mass_volume_no_empty_std = (
        compute_dice_imagewise_from_cumulator(TP, FP, FN, TN, exclude_empty=True, return_std=True)
    )
    model_dice_mass_volume_no_empty_optimistic, model_dice_mass_volume_no_empty_optimistic_std = (
        compute_dice_imagewise_from_cumulator(
            TP, FP, FN, TN, exclude_empty=True, exclude_empty_only_gt=True, return_std=True
        )
    )

    model_mean_accuracy_no_empty, model_mean_accuracy_no_empty_std = (
        compute_accuracy_from_cumulator(
            TP, FP, FN, TN, exclude_empty=True, is_mean=True, return_std=True
        )
    )
    model_mean_precision_no_empty, model_mean_precision_no_empty_std = (
        compute_precision_from_cumulator(
            TP, FP, FN, TN, exclude_empty=True, is_mean=True, return_std=True
        )
    )
    model_mean_recall_no_empty, model_mean_recall_no_empty_std = compute_recall_from_cumulator(
        TP, FP, FN, TN, exclude_empty=True, is_mean=True, return_std=True
    )
    model_mean_f1_no_empty, model_mean_f1_no_empty_std = compute_f1_from_cumulator(
        TP, FP, FN, TN, exclude_empty=True, is_mean=True, return_std=True
    )

    model_accuracy_excluding_cases, model_accuracy_excluding_cases_std = (
        compute_accuracy_excluding_cases(TP, FP, FN, TN, return_std=True)
    )
    model_precision_excluding_cases, model_precision_excluding_cases_std = (
        compute_precision_excluding_cases_from_cumulator(TP, FP, FN, TN, return_std=True)
    )
    model_recall_excluding_cases, model_recall_excluding_cases_std = (
        compute_recall_excluding_cases_from_cumulator(TP, FP, FN, TN, return_std=True)
    )

    model_accuracy_no_empty, model_accuracy_no_empty_std = compute_accuracy_from_cumulator(
        TP, FP, FN, TN, exclude_empty=True, is_mean=False, return_std=True
    )
    model_precision_no_empty, model_precision_no_empty_std = (
        compute_precision_excluding_cases_from_cumulator(
            TP, FP, FN, TN, return_std=True, exclude_only_zero_denominator=True
        )
    )
    model_recall_no_empty, model_recall_no_empty_std = (
        compute_recall_excluding_cases_from_cumulator(
            TP, FP, FN, TN, return_std=True, exclude_only_zero_denominator=True
        )
    )

    model_f1_no_empty, model_f1_no_empty_std = compute_f1_excluding_cases_from_cumulator(
        TP, FP, FN, TN, return_std=True, exclude_only_zero_denominator=True
    )

    print("MODEL CLASS MEAN IOU ", model_class_mean_iou)
    print("MODEL CLASS STD IOU ", model_class_std_iou)
    print()
    print("MODEL CLASS MEAN DICE ", model_class_mean_dice)
    print("MODEL CLASS STD DICE ", model_class_std_dice)
    print()
    print("MODEL DIOU", model_detection_iou)
    print("MODEL DIOU STD ", model_detection_iou_std)
    print()
    print("MODEL IOU MASS VOLUME ", model_iou_mass_volume)
    print("MODEL IOU MASS VOLUME STD ", model_iou_mass_volume_std)
    print()
    print("MODEL IOU MASS VOLUME NO EMPTY ", model_iou_mass_volume_no_empty)
    print("MODEL IOU MASS VOLUME NO EMPTY STD ", model_iou_mass_volume_no_empty_std)
    print()
    print("MODEL IOU MASS VOLUME NO EMPTY OPTIMISTIC ", model_iou_mass_volume_no_empty_optimistic)
    print(
        "MODEL IOU MASS VOLUME NO EMPTY OPTIMISTIC STD ",
        model_iou_mass_volume_no_empty_optimistic_std,
    )

    print("MODEL DICE MASS VOLUME ", model_dice_mass_volume)
    print("MODEL DICE MASS VOLUME STD ", model_dice_mass_volume_std)
    print()
    print("MODEL DICE MASS VOLUME NO EMPTY ", model_dice_mass_volume_no_empty)
    print("MODEL DICE MASS VOLUME NO EMPTY STD ", model_dice_mass_volume_no_empty_std)
    print()
    print("MODEL DICE MASS VOLUME NO EMPTY OPTIMISTIC ", model_dice_mass_volume_no_empty_optimistic)
    print(
        "MODEL DICE MASS VOLUME NO EMPTY OPTIMISTIC STD ",
        model_dice_mass_volume_no_empty_optimistic_std,
    )
    print()
    print("MODEL MEAN ACCURACY NO EMPTY", model_mean_accuracy_no_empty)
    print("MODEL MEAN ACCURACY NO EMPTY STD", model_mean_accuracy_no_empty_std)
    print()
    print("MODEL MEAN PRECISION NO EMPTY", model_mean_precision_no_empty)
    print("MODEL MEAN PRECISION NO EMPTY STD", model_mean_precision_no_empty_std)
    print()
    print("MODEL MEAN RECALL NO EMPTY", model_mean_recall_no_empty)
    print("MODEL MEAN RECALL NO EMPTY STD", model_mean_recall_no_empty_std)
    print()
    print("MODEL MEAN F1 NO EMPTY", model_mean_f1_no_empty)
    print("MODEL MEAN F1 NO EMPTY STD", model_mean_f1_no_empty_std)
    print()
    print("MODEL ACCURACY EXCLUDING CASES ", model_accuracy_excluding_cases)
    print("MODEL ACCURACY EXCLUDING CASES STD ", model_accuracy_excluding_cases_std)
    print()
    print("MODEL PRECISION EXCLUDING CASES ", model_precision_excluding_cases)
    print("MODEL PRECISION EXCLUDING CASES STD ", model_precision_excluding_cases_std)
    print()
    print("MODEL RECALL EXCLUDING CASES ", model_recall_excluding_cases)
    print("MODEL RECALL EXCLUDING CASES STD ", model_recall_excluding_cases_std)
    print()
    print("MODEL ACCURACY NO EMPTY ", model_accuracy_no_empty)
    print("MODEL ACCURACY NO EMPTY STD ", model_accuracy_no_empty_std)
    print()
    print("MODEL PRECISION NO EMPTY", model_precision_no_empty)
    print("MODEL PRECISION NO EMPTY STD ", model_precision_no_empty_std)
    print()
    print("MODEL RECALL NO EMPTY ", model_recall_no_empty)
    print("MODEL RECALL NO EMPTY STD ", model_recall_no_empty_std)
    print()
    print("MODEL F1 NO EMPTY ", model_f1_no_empty)
    print("MODEL F1 NO EMPTY STD ", model_f1_no_empty_std)
    print()

    if get_scores_for_statistics:
        tp = torch.cat([tp for tp in TP])
        fp = torch.cat([fp for fp in FP])
        fn = torch.cat([fn for fn in FN])
        tn = torch.cat([tn for tn in TN])

        if get_only_masses:
            # Create a mask where tp + fn is not equal to 0
            mask = (tp + fn) != 0

            # Apply this mask to each tensor to filter out the desired values
            tp = tp[mask]
            fp = fp[mask]
            fn = fn[mask]
            tn = tn[mask]

        miou_scores = compute_mean_iou_imagewise_from_cumulator(
            tp, fp, fn, tn, exclude_empty=False, return_std=False, reduce_mean=False
        )
        mdice_scores = compute_mean_dice_imagewise_from_cumulator(
            tp, fp, fn, tn, exclude_empty=False, return_std=False, reduce_mean=False
        )
        mf1_scores = compute_f1_from_cumulator(
            tp, fp, fn, tn, exclude_empty=False, is_mean=True, return_std=False, reduce_mean=False
        )

        scores_dict = {
            "miou": miou_scores.squeeze().tolist(),
            "mdice": mdice_scores.squeeze().tolist(),
            "mf1": mf1_scores.squeeze().tolist(),
        }
        return scores_dict
