"""
Private dataset inference functions using the exact approach from the reference notebook.
"""

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
    filter_masses
)
from ..utils.postprocessing import fuse_segmentations
import time

import torch

def print_model_params_and_memory(model):
    # Function to calculate memory size in MB
    def get_model_memory(model):
        total_params = sum(p.numel() for p in model.parameters())
        param_size = next(model.parameters()).element_size()  # Size of one parameter (in bytes)
        total_memory = total_params * param_size / (1024 ** 2)  # Memory in MB
        return total_memory
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Calculate trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate non-trainable parameters
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    # Get model memory size
    model_memory = get_model_memory(model)
    
    # Print the results in millions
    print(f"Total parameters: {total_params / 1e6:.2f} million")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f} million")
    print(f"Non-trainable parameters: {non_trainable_params / 1e6:.2f} million")
    print(f"Memory required (in MB): {model_memory:.2f} MB")


def create_patient_datasets(patient_ids: List[str], dataset_base_path: str , mean_patches_sub, std_patches_sub, mean_no_thorax_third_sub, std_no_thorax_third_sub, num_workers: int) -> Dict[str, Dict[str, Any]]:
    """
    Create datasets for each patient following the exact reference notebook approach.
    
    Args:
        patient_ids: List of patient IDs
        dataset_base_path: Base path to the dataset
        num_workers: Number of workers for data loading
        
    Returns:
        Dictionary with patient datasets
    """
    
    sub_third_images_path_prefixes = ("Dataset-arrays-4-FINAL", "Dataset-arrays-FINAL")
    image_only = False
    has_mask = True
    get_boundaryloss = True
    
    test_transforms_patches_sub = Compose([
        LoadImaged(keys=["image", "label"], image_only=image_only, reader=monai.data.NumpyReader()),
        EnsureChannelFirstd(keys=["image", "label"]),
        Preprocess(
            has_mask=has_mask,
            keys=None, 
            mode='test', 
            get_boundaryloss=get_boundaryloss, 
            subtracted_images_path_prefixes=sub_third_images_path_prefixes, 
            subtrahend=mean_patches_sub, 
            divisor=std_patches_sub,
            get_patches=True
        )
    ])
    
    test_transforms_no_thorax_third_sub = Compose([
        LoadImaged(keys=["image", "label"], image_only=image_only, reader=monai.data.NumpyReader()),
        EnsureChannelFirstd(keys=["image", "label"]),
        Preprocess(
            has_mask=has_mask,
            keys=None, 
            mode='test', 
            get_boundaryloss=get_boundaryloss, 
            subtracted_images_path_prefixes=sub_third_images_path_prefixes, 
            subtrahend=mean_no_thorax_third_sub, 
            divisor=std_no_thorax_third_sub, 
            get_patches=False
        )
    ])
    
    datasets = {}
    
    for patient_id in patient_ids:
        print(f"Creating datasets for patient: {patient_id}")
        
        # Get patient files
        images_fnames, _ = get_filenames(
            suffix="images", 
            base_path=dataset_base_path,
            patient_ids=[patient_id], 
            remove_black_samples=False,
            get_top_bottom_and_remove_black_samples=True,
            random_samples_indexes_list=None, 
            remove_picked_samples=True
        )
        labels_fnames, _ = get_filenames(
            suffix="masks", 
            base_path=dataset_base_path,
            patient_ids=[patient_id], 
            remove_black_samples=False,
            get_top_bottom_and_remove_black_samples=True,
            random_samples_indexes_list=None, 
            remove_picked_samples=True
        )
        
        if not images_fnames or not labels_fnames:
            continue
            
        test_dicts = [{"image": image_name, "label": label_name} 
                     for image_name, label_name in zip(images_fnames, labels_fnames)]
        
        no_thorax_sub_test_ds = CacheDataset(
            data=test_dicts, 
            transform=test_transforms_no_thorax_third_sub,
            num_workers=num_workers
        )
        patches_sub_test_ds = CacheDataset(
            data=test_dicts, 
            transform=test_transforms_patches_sub,
            num_workers=num_workers
        )
        
        datasets[patient_id] = {
            "no_thorax_sub_test_ds": no_thorax_sub_test_ds,
            "patches_sub_test_ds": patches_sub_test_ds
        }
    
    return datasets


def test_dataset_aware_fusion(model_path: str, patient_ids: List[str], datasets: Dict,
                             whole_dataset_key: str, patches_dataset_key: str, 
                             use_simple_fusion: bool = False, use_decoder_attention: bool = True,
                             strict: bool = True, filter: bool = False, subtracted: bool = True,
                             get_scores_for_statistics: bool = False, get_only_masses: bool = False, 
                             base_channels: int = 64) -> Dict:
    """
    Test fusion model with dataset-aware metric computation.
    """
    # Load model
    model = BreastFusionModel.load_from_checkpoint(
        model_path, 
        strict=strict, 
        use_simple_fusion=use_simple_fusion, 
        use_decoder_attention=use_decoder_attention, 
        base_channels=base_channels
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


    detection_iou=[]


    for patient_id in patient_ids:
         predicted_label_slices = []
         gt_label_slices=[]
         image_slices=[]
         print(patient_id)
         patches_ds = datasets[patient_id][patches_dataset_key]
         whole_image_ds = datasets[patient_id][whole_dataset_key]

         fusion_dataset = PairedDataset(whole_image_ds, patches_ds, augment=False)
        
         prev_had_mask=False

         for idx, e in tqdm(enumerate(fusion_dataset), total = len(patches_ds)):
            original_image = np.load(e[0]['image_meta_dict']['filename_or_obj'])
            original_image = np.expand_dims(original_image,0)
    
            
            pred_label = torch.zeros(original_image.shape, dtype=torch.uint8)
    
            gt_label = np.load(e[0]['label_meta_dict']['filename_or_obj'])
            gt_label= np.expand_dims(gt_label,0)

            if fusion_dataset[idx][0]['keep_sample']:
    
                whole_image = torch.unsqueeze(fusion_dataset[idx][0]['image'], 0)
                patch_image2 = torch.unsqueeze(fusion_dataset[idx][1]['image'], 0)
                patch_image3 = torch.unsqueeze(fusion_dataset[idx][2]['image'], 0)
                    
        
                with torch.no_grad():
                    masks = []
                    # pass to model
                    model = model.to("cuda")
                    model.eval()
                    
                    masks = model(whole_image.to("cuda"),patch_image2.to("cuda"),patch_image3.to("cuda"))
                    masks = masks.sigmoid()
                    

                pred_label = (pred_label > 0.4).int()
                pred_label = reverse_transformations(fusion_dataset[idx][0], pred_label, mode='whole')
                

                
            pred_label = monai.transforms.Resize(spatial_size=(original_image.shape[1], original_image.shape[2]), mode='nearest-exact')(pred_label)
             
            if not filter:
                tp, fp, fn, tn = smp.metrics.get_stats(torch.tensor(np.expand_dims(pred_label,0).astype(int)), torch.tensor(np.expand_dims(gt_label,0).astype(int)), mode = "binary")
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

         predicted_label_volume = np.stack(predicted_label_slices, axis=-1)  # Stack along the first axis to create a 3D volume
                    
         gt_label_volume = np.stack(gt_label_slices, axis=-1)
         images_volume = np.stack(image_slices, axis=-1)
    
         if filter:
             predicted_label_volume = filter_masses(predicted_label_volume, min_slices=3, window_size=3) # H x W x N
             # H x W x N -> N x H x W -> N x 1 x H x W
             predicted_label_volume_for_stats = np.transpose(predicted_label_volume, (2, 0, 1))
             predicted_label_volume_for_stats = np.expand_dims(predicted_label_volume_for_stats, 1)  # N x 1 x H x W

             gt_label_volume_for_stats = np.transpose(gt_label_volume, (2, 0, 1))
             gt_label_volume_for_stats = np.expand_dims(gt_label_volume_for_stats, 1)  # N x 1 x H x W             
            
             
             tp, fp, fn, tn = smp.metrics.get_stats(torch.tensor(predicted_label_volume_for_stats.astype(int)), torch.tensor(gt_label_volume_for_stats.astype(int)), mode = "binary")
             TP +=  [torch.tensor([[elem]]) for elem in tp.squeeze()]
             FP +=  [torch.tensor([[elem]]) for elem in fp.squeeze()]
             FN +=  [torch.tensor([[elem]]) for elem in fn.squeeze()]
             TN +=  [torch.tensor([[elem]]) for elem in tn.squeeze()]



         detection_iou+=calculate_mass_detection_imagewise_volume(predicted_label_volume.astype(int), gt_label_volume)
    
    model_detection_iou = np.array(detection_iou).mean()
    model_detection_iou_std = np.array(detection_iou).std()
    
    model_class_mean_iou, model_class_std_iou = compute_mean_iou_imagewise_from_cumulator(TP, FP, FN, TN, exclude_empty=True, return_std=True)
    model_class_mean_dice, model_class_std_dice = compute_mean_dice_imagewise_from_cumulator(TP, FP, FN, TN, exclude_empty=True, return_std=True)

    model_iou_mass_volume , model_iou_mass_volume_std = compute_iou_imagewise_from_cumulator(TP, FP, FN, TN, exclude_empty=False, return_std=True)
    model_iou_mass_volume_no_empty, model_iou_mass_volume_no_empty_std =compute_iou_imagewise_from_cumulator(TP, FP, FN, TN, exclude_empty=True, return_std=True)
    model_iou_mass_volume_no_empty_optimistic, model_iou_mass_volume_no_empty_optimistic_std =compute_iou_imagewise_from_cumulator(TP, FP, FN, TN, exclude_empty=True, exclude_empty_only_gt=True, return_std=True)
    
    
    model_dice_mass_volume, model_dice_mass_volume_std = compute_dice_imagewise_from_cumulator(TP, FP, FN, TN,exclude_empty=False, return_std=True)
    model_dice_mass_volume_no_empty, model_dice_mass_volume_no_empty_std = compute_dice_imagewise_from_cumulator(TP, FP, FN, TN,exclude_empty=True, return_std=True)
    model_dice_mass_volume_no_empty_optimistic, model_dice_mass_volume_no_empty_optimistic_std = compute_dice_imagewise_from_cumulator(TP, FP, FN, TN,exclude_empty=True, exclude_empty_only_gt=True,return_std=True)
    
    model_mean_accuracy_no_empty, model_mean_accuracy_no_empty_std = compute_accuracy_from_cumulator(TP, FP, FN, TN, exclude_empty=True, is_mean=True, return_std=True)
    model_mean_precision_no_empty,model_mean_precision_no_empty_std = compute_precision_from_cumulator(TP, FP, FN, TN, exclude_empty=True, is_mean=True, return_std=True)
    model_mean_recall_no_empty, model_mean_recall_no_empty_std = compute_recall_from_cumulator(TP, FP, FN, TN, exclude_empty=True, is_mean=True, return_std=True)
    model_mean_f1_no_empty, model_mean_f1_no_empty_std = compute_f1_from_cumulator(TP, FP, FN, TN, exclude_empty=True, is_mean=True, return_std=True)
        
    model_accuracy_excluding_cases, model_accuracy_excluding_cases_std = compute_accuracy_excluding_cases(TP, FP, FN, TN, return_std=True)
    model_precision_excluding_cases,model_precision_excluding_cases_std =compute_precision_excluding_cases_from_cumulator(TP, FP, FN, TN, return_std=True)
    model_recall_excluding_cases,model_recall_excluding_cases_std  =compute_recall_excluding_cases_from_cumulator(TP, FP, FN, TN, return_std=True)

    model_accuracy_no_empty, model_accuracy_no_empty_std = compute_accuracy_from_cumulator(TP, FP, FN, TN, exclude_empty=True, is_mean=False, return_std=True)
    model_precision_no_empty,model_precision_no_empty_std =compute_precision_excluding_cases_from_cumulator(TP, FP, FN, TN, return_std=True,exclude_only_zero_denominator=True)
    model_recall_no_empty,model_recall_no_empty_std  = compute_recall_excluding_cases_from_cumulator(TP, FP, FN, TN, return_std=True,exclude_only_zero_denominator=True)

    model_f1_no_empty,model_f1_no_empty_std = compute_f1_excluding_cases_from_cumulator(TP, FP, FN, TN, return_std=True,exclude_only_zero_denominator=True)
    
    
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
    print("MODEL IOU MASS VOLUME NO EMPTY OPTIMISTIC STD ", model_iou_mass_volume_no_empty_optimistic_std)
    
    print("MODEL DICE MASS VOLUME ", model_dice_mass_volume)
    print("MODEL DICE MASS VOLUME STD ", model_dice_mass_volume_std)
    print()
    print("MODEL DICE MASS VOLUME NO EMPTY ", model_dice_mass_volume_no_empty)
    print("MODEL DICE MASS VOLUME NO EMPTY STD ", model_dice_mass_volume_no_empty_std)
    print()
    print("MODEL DICE MASS VOLUME NO EMPTY OPTIMISTIC ", model_dice_mass_volume_no_empty_optimistic)
    print("MODEL DICE MASS VOLUME NO EMPTY OPTIMISTIC STD ", model_dice_mass_volume_no_empty_optimistic_std)
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
    print("MODEL ACCURACY EXCLUDING CASES ",  model_accuracy_excluding_cases)
    print("MODEL ACCURACY EXCLUDING CASES STD ",  model_accuracy_excluding_cases_std)
    print()
    print("MODEL PRECISION EXCLUDING CASES ",  model_precision_excluding_cases)
    print("MODEL PRECISION EXCLUDING CASES STD ",  model_precision_excluding_cases_std)
    print()
    print("MODEL RECALL EXCLUDING CASES ", model_recall_excluding_cases)
    print("MODEL RECALL EXCLUDING CASES STD ", model_recall_excluding_cases_std)
    print()
    print("MODEL ACCURACY NO EMPTY ",  model_accuracy_no_empty)
    print("MODEL ACCURACY NO EMPTY STD ",  model_accuracy_no_empty_std)
    print()
    print("MODEL PRECISION NO EMPTY",  model_precision_no_empty)
    print("MODEL PRECISION NO EMPTY STD ",  model_precision_no_empty_std)
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

            miou_scores = compute_mean_iou_imagewise_from_cumulator(tp, fp, fn, tn, exclude_empty=False, return_std=False,reduce_mean=False)
            mdice_scores = compute_mean_dice_imagewise_from_cumulator(tp, fp, fn, tn, exclude_empty=False, return_std=False,reduce_mean=False)
            mf1_scores = compute_f1_from_cumulator(tp, fp, fn, tn, exclude_empty=False, is_mean=True, return_std=False,reduce_mean=False)


            scores_dict = {
                 'miou': miou_scores.squeeze().tolist(),
                 'mdice': mdice_scores.squeeze().tolist(),
                 "mf1": mf1_scores.squeeze().tolist(),
            }
            return scores_dict


def test_dataset_aware_no_patches(model_path: str, patient_ids: List[str], datasets: Dict,
                                 dataset_key: str, filter: bool = False, get_scores_for_statistics: bool = False,
                                 get_only_masses: bool = False, arch_name: Optional[str] = None, 
                                 strict: bool = False, subtracted: bool = True) -> Dict:
    if arch_name:
        model = BreastSegmentationModel.load_from_checkpoint(model_path, strict=strict, arch=arch_name)
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
    detection_iou=[]
    
    for patient_id in patient_ids:

         predicted_label_slices = []
         gt_label_slices=[]
         image_slices=[]
        
         print(patient_id)
         dataset = datasets[patient_id][dataset_key]
        
         for idx, e in tqdm(enumerate(dataset), total = len(dataset)):
            original_image = np.load(e['image_meta_dict']['filename_or_obj'])
            original_image = np.expand_dims(original_image,0)

            gt_label = np.load(e['label_meta_dict']['filename_or_obj'])
            gt_label= np.expand_dims(gt_label,0)

            if e['keep_sample']:
                image = torch.unsqueeze(e['image'], 0)
                    
                with torch.no_grad():
                    model = model.to("cuda")
                    model.eval()
                    if arch_name:
                            masks = model(image.to("cuda"))[0]
                    else:
                            masks = model(image.to("cuda"))[0]
                    masks = masks.sigmoid()
                    
                pred_label = masks[0]
                pred_label = (pred_label > 0.4).int()
                pred_label = torch.squeeze(pred_label)
                pred_label = torch.unsqueeze(pred_label,0)
                pred_label = reverse_transformations(dataset[idx], pred_label, mode='whole')
                
                pred_label = monai.transforms.Resize(spatial_size=(original_image.shape[1], original_image.shape[2]), mode='nearest-exact')(pred_label)
            else:
                pred_label = torch.zeros(original_image.shape, dtype=torch.uint8)

            if not filter:
                tp, fp, fn, tn = smp.metrics.get_stats(torch.tensor(np.expand_dims(pred_label,0).astype(int)), torch.tensor(np.expand_dims(gt_label,0).astype(int)), mode = "binary")
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

         predicted_label_volume = np.stack(predicted_label_slices, axis=-1)  # Stack along the first axis to create a 3D volume
                    
         gt_label_volume = np.stack(gt_label_slices, axis=-1)
         images_volume = np.stack(image_slices, axis=-1)
    
         if filter:
             predicted_label_volume = filter_masses(predicted_label_volume, min_slices=3, window_size=3) # H x W x N
             # H x W x N -> N x H x W -> N x 1 x H x W
             predicted_label_volume_for_stats = np.transpose(predicted_label_volume, (2, 0, 1))
             predicted_label_volume_for_stats = np.expand_dims(predicted_label_volume_for_stats, 1)  # N x 1 x H x W

             gt_label_volume_for_stats = np.transpose(gt_label_volume, (2, 0, 1))
             gt_label_volume_for_stats = np.expand_dims(gt_label_volume_for_stats, 1)  # N x 1 x H x W             
            
             
             tp, fp, fn, tn = smp.metrics.get_stats(torch.tensor(predicted_label_volume_for_stats.astype(int)), torch.tensor(gt_label_volume_for_stats.astype(int)), mode = "binary")
             TP +=  [torch.tensor([[elem]]) for elem in tp.squeeze()]
             FP +=  [torch.tensor([[elem]]) for elem in fp.squeeze()]
             FN +=  [torch.tensor([[elem]]) for elem in fn.squeeze()]
             TN +=  [torch.tensor([[elem]]) for elem in tn.squeeze()]

         detection_iou+=calculate_mass_detection_imagewise_volume(predicted_label_volume.astype(int), gt_label_volume)

    model_detection_iou = np.array(detection_iou).mean()
    model_detection_iou_std = np.array(detection_iou).std()
    
    model_class_mean_iou, model_class_std_iou = compute_mean_iou_imagewise_from_cumulator(TP, FP, FN, TN, exclude_empty=True, return_std=True)
    model_class_mean_dice, model_class_std_dice = compute_mean_dice_imagewise_from_cumulator(TP, FP, FN, TN, exclude_empty=True, return_std=True)

    model_iou_mass_volume , model_iou_mass_volume_std = compute_iou_imagewise_from_cumulator(TP, FP, FN, TN, exclude_empty=False, return_std=True)
    model_iou_mass_volume_no_empty, model_iou_mass_volume_no_empty_std =compute_iou_imagewise_from_cumulator(TP, FP, FN, TN, exclude_empty=True, return_std=True)
    model_iou_mass_volume_no_empty_optimistic, model_iou_mass_volume_no_empty_optimistic_std =compute_iou_imagewise_from_cumulator(TP, FP, FN, TN, exclude_empty=True, exclude_empty_only_gt=True, return_std=True)
    
    
    model_dice_mass_volume, model_dice_mass_volume_std = compute_dice_imagewise_from_cumulator(TP, FP, FN, TN,exclude_empty=False, return_std=True)
    model_dice_mass_volume_no_empty, model_dice_mass_volume_no_empty_std = compute_dice_imagewise_from_cumulator(TP, FP, FN, TN,exclude_empty=True, return_std=True)
    model_dice_mass_volume_no_empty_optimistic, model_dice_mass_volume_no_empty_optimistic_std = compute_dice_imagewise_from_cumulator(TP, FP, FN, TN,exclude_empty=True, exclude_empty_only_gt=True,return_std=True)
    
    model_mean_accuracy_no_empty, model_mean_accuracy_no_empty_std = compute_accuracy_from_cumulator(TP, FP, FN, TN, exclude_empty=True, is_mean=True, return_std=True)
    model_mean_precision_no_empty,model_mean_precision_no_empty_std = compute_precision_from_cumulator(TP, FP, FN, TN, exclude_empty=True, is_mean=True, return_std=True)
    model_mean_recall_no_empty, model_mean_recall_no_empty_std = compute_recall_from_cumulator(TP, FP, FN, TN, exclude_empty=True, is_mean=True, return_std=True)
    model_mean_f1_no_empty, model_mean_f1_no_empty_std = compute_f1_from_cumulator(TP, FP, FN, TN, exclude_empty=True, is_mean=True, return_std=True)
        
    model_accuracy_excluding_cases, model_accuracy_excluding_cases_std = compute_accuracy_excluding_cases(TP, FP, FN, TN, return_std=True)
    model_precision_excluding_cases,model_precision_excluding_cases_std =compute_precision_excluding_cases_from_cumulator(TP, FP, FN, TN, return_std=True)
    model_recall_excluding_cases,model_recall_excluding_cases_std  =compute_recall_excluding_cases_from_cumulator(TP, FP, FN, TN, return_std=True)

    model_accuracy_no_empty, model_accuracy_no_empty_std = compute_accuracy_from_cumulator(TP, FP, FN, TN, exclude_empty=True, is_mean=False, return_std=True)
    model_precision_no_empty,model_precision_no_empty_std =compute_precision_excluding_cases_from_cumulator(TP, FP, FN, TN, return_std=True,exclude_only_zero_denominator=True)
    model_recall_no_empty,model_recall_no_empty_std  = compute_recall_excluding_cases_from_cumulator(TP, FP, FN, TN, return_std=True,exclude_only_zero_denominator=True)

    model_f1_no_empty,model_f1_no_empty_std = compute_f1_excluding_cases_from_cumulator(TP, FP, FN, TN, return_std=True,exclude_only_zero_denominator=True)
    
    
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
    print("MODEL IOU MASS VOLUME NO EMPTY OPTIMISTIC STD ", model_iou_mass_volume_no_empty_optimistic_std)
    
    print("MODEL DICE MASS VOLUME ", model_dice_mass_volume)
    print("MODEL DICE MASS VOLUME STD ", model_dice_mass_volume_std)
    print()
    print("MODEL DICE MASS VOLUME NO EMPTY ", model_dice_mass_volume_no_empty)
    print("MODEL DICE MASS VOLUME NO EMPTY STD ", model_dice_mass_volume_no_empty_std)
    print()
    print("MODEL DICE MASS VOLUME NO EMPTY OPTIMISTIC ", model_dice_mass_volume_no_empty_optimistic)
    print("MODEL DICE MASS VOLUME NO EMPTY OPTIMISTIC STD ", model_dice_mass_volume_no_empty_optimistic_std)
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
    print("MODEL ACCURACY EXCLUDING CASES ",  model_accuracy_excluding_cases)
    print("MODEL ACCURACY EXCLUDING CASES STD ",  model_accuracy_excluding_cases_std)
    print()
    print("MODEL PRECISION EXCLUDING CASES ",  model_precision_excluding_cases)
    print("MODEL PRECISION EXCLUDING CASES STD ",  model_precision_excluding_cases_std)
    print()
    print("MODEL RECALL EXCLUDING CASES ", model_recall_excluding_cases)
    print("MODEL RECALL EXCLUDING CASES STD ", model_recall_excluding_cases_std)
    print()
    print("MODEL ACCURACY NO EMPTY ",  model_accuracy_no_empty)
    print("MODEL ACCURACY NO EMPTY STD ",  model_accuracy_no_empty_std)
    print()
    print("MODEL PRECISION NO EMPTY",  model_precision_no_empty)
    print("MODEL PRECISION NO EMPTY STD ",  model_precision_no_empty_std)
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

            miou_scores = compute_mean_iou_imagewise_from_cumulator(tp, fp, fn, tn, exclude_empty=False, return_std=False,reduce_mean=False)
            mdice_scores = compute_mean_dice_imagewise_from_cumulator(tp, fp, fn, tn, exclude_empty=False, return_std=False,reduce_mean=False)
            mf1_scores = compute_f1_from_cumulator(tp, fp, fn, tn, exclude_empty=False, is_mean=True, return_std=False,reduce_mean=False)


            scores_dict = {
                 'miou': miou_scores.squeeze().tolist(),
                 'mdice': mdice_scores.squeeze().tolist(),
                 "mf1": mf1_scores.squeeze().tolist(),
            }
            return scores_dict


def test_dataset_aware_ensemble(model_whole_path: str, model_patches_path: str, patient_ids: List[str], 
                               datasets: Dict, whole_dataset_key: str, patches_dataset_key: str,
                               filter: bool = False, get_scores_for_statistics: bool = False,
                               get_only_masses: bool = False, subtracted: bool = True, 
                               base_channels: int = 64) -> Dict:
    """
    Test ensemble model with dataset-aware metric computation.
    """
    # Load models
    model_whole = BreastFusionModel.load_from_checkpoint(model_whole_path, strict=False, base_channels=base_channels)
    model_patches = BreastSegmentationModel.load_from_checkpoint(model_patches_path, strict=False)
    
    #print_model_params_and_memory(model_whole)
    #print_model_params_and_memory(model_patches)

    
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

    detection_iou =  []

    # Initialize performance metrics
    inference_times = []  # To store the time for each volume inference
    inference_times_slice = []  # To store the time for each volume inference
    memory_usage = []  # To store the memory usage for each volume

    for patient_id in patient_ids:

         predicted_label_slices = []
         gt_label_slices=[]
         image_slices=[]

         print(patient_id)
         patches_ds = datasets[patient_id][patches_dataset_key]
         whole_image_ds = datasets[patient_id][whole_dataset_key]

         fusion_dataset = PairedDataset(whole_image_ds, patches_ds, augment=False)
        
         prev_had_mask=False

         # Measure inference time per slice
         start_time = time.time()

         for idx, e in tqdm(enumerate(patches_ds), total = len(patches_ds)):

            start_time_slice = time.time()


             
            original_image = np.load(e[0]['image_meta_dict']['filename_or_obj'])
            original_image = np.expand_dims(original_image,0)
    
            merged_label_for_fusion = torch.zeros(original_image.shape)
    
            gt_label = np.load(e[0]['label_meta_dict']['filename_or_obj'])
            gt_label= np.expand_dims(gt_label,0)
            
            ## FIRST MODEL
            for elem in e:
                if elem['keep_sample']:
                    image = torch.unsqueeze(elem['image'], 0)
                    with torch.no_grad():
                        model_patches = model_patches.to("cuda")
                        model_patches.eval()
                        logits = model_patches(image.to("cuda"))[0]
    
                    pr_mask = logits.sigmoid()
                    
                    if pr_mask.ndim > 3: # THIS IS THE CASE FOR MODELS RETURNING (BS,1,H,W)
                        pr_mask = pr_mask[0]
                    #pr_mask_to_viz = (pr_mask.cpu().numpy() > 0.4).astype(int)
    
                    if pr_mask.sum()>0:
                        #label = pr_mask
                        label = reverse_transformations(elem, pr_mask, mode='patches')
                        merged_label_for_fusion += label
    
    
            original_image = np.transpose(original_image, (1,2,0))
    
            label_patches_for_fusion = merged_label_for_fusion[0]
    
            # SECOND MODEL
            if fusion_dataset[idx][0]['keep_sample'] or fusion_dataset[idx][1]['keep_sample'] or fusion_dataset[idx][2]['keep_sample']:
    
                whole_image = torch.unsqueeze(fusion_dataset[idx][0]['image'], 0)
                patch_image2 = torch.unsqueeze(fusion_dataset[idx][1]['image'], 0)
                patch_image3 = torch.unsqueeze(fusion_dataset[idx][2]['image'], 0)
                    
        
                with torch.no_grad():
                    masks = []
                    # pass to model
                    model_whole = model_whole.to("cuda")
                    model_whole.eval()
                    
                    masks = model_whole(whole_image.to("cuda"),patch_image2.to("cuda"),patch_image3.to("cuda"))
                    masks = masks.sigmoid()
                    
                    
    
                label_whole = masks[0]
                label_whole = (label_whole > 0.4).int()
                label_whole = reverse_transformations(whole_image_ds[idx], label_whole, mode='whole')
                label_whole = label_whole.squeeze()
        
                label_whole_for_fusion= masks[0]
                label_whole_for_fusion = reverse_transformations(whole_image_ds[idx], label_whole_for_fusion, mode='whole')
                
                # Plot the first image
            else:
                label_whole_for_fusion = torch.zeros(original_image.shape)
                
            original_image_squeeze = np.load(e[0]['image_meta_dict']['filename_or_obj'])

            fusion = fuse_segmentations(label_whole_for_fusion.numpy(), label_patches_for_fusion.numpy(), prob_threshold=0.4, boost_factor=3, penalty_factor=0.5, kernel_size=150)
            
            fusion = (fusion > 0.4).astype(int)
    
            fusion = np.expand_dims(fusion, 0)
            pred_label=fusion

            if not filter:
                tp, fp, fn, tn = smp.metrics.get_stats(torch.tensor(np.expand_dims(fusion,0).astype(int)), torch.tensor(np.expand_dims(gt_label,0).astype(int)), mode = "binary")
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

            # Measure inference time after processing  the voluyme
            end_time_slice = time.time()
            inference_times_slice.append(end_time_slice - start_time_slice)

         predicted_label_volume = np.stack(predicted_label_slices, axis=-1)  # Stack along the first axis to create a 3D volume
                    
         gt_label_volume = np.stack(gt_label_slices, axis=-1)
         images_volume = np.stack(image_slices, axis=-1)

        
         # Measure inference time after processing  the volume
         end_time = time.time()
         inference_times.append(end_time - start_time)
         memory_allocated = torch.cuda.memory_allocated()
         memory_usage.append(memory_allocated)
    
         if filter:
             print("filtering")
             predicted_label_volume = filter_masses(predicted_label_volume, min_slices=3, window_size=3) # H x W x N
             # H x W x N -> N x H x W -> N x 1 x H x W
             predicted_label_volume_for_stats = np.transpose(predicted_label_volume, (2, 0, 1))
             predicted_label_volume_for_stats = np.expand_dims(predicted_label_volume_for_stats, 1)  # N x 1 x H x W

             gt_label_volume_for_stats = np.transpose(gt_label_volume, (2, 0, 1))
             gt_label_volume_for_stats = np.expand_dims(gt_label_volume_for_stats, 1)  # N x 1 x H x W             
            
             
             tp, fp, fn, tn = smp.metrics.get_stats(torch.tensor(predicted_label_volume_for_stats.astype(int)), torch.tensor(gt_label_volume_for_stats.astype(int)), mode = "binary")
             TP +=  [torch.tensor([[elem]]) for elem in tp.squeeze()]
             FP +=  [torch.tensor([[elem]]) for elem in fp.squeeze()]
             FN +=  [torch.tensor([[elem]]) for elem in fn.squeeze()]
             TN +=  [torch.tensor([[elem]]) for elem in tn.squeeze()]

        


         detection_iou+=calculate_mass_detection_imagewise_volume(predicted_label_volume.astype(int), gt_label_volume)




    # Calculate mean and standard deviation for inference time and memory usage
    mean_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)

    mean_inference_time_slice = np.mean(inference_times_slice)
    std_inference_time_slice = np.std(inference_times_slice)
    
    mean_memory_usage = np.mean(memory_usage)
    std_memory_usage = np.std(memory_usage)
        
    # Frames per second (inference speed)
    fps = 1 / mean_inference_time_slice
        
    # Final outputs
    print(f"Mean Inference Time per Volume: {mean_inference_time:.4f} seconds")
    print(f"Standard Deviation of Inference Time per Volume: {std_inference_time:.4f} seconds")

    print(f"Mean Inference Time per Slice: {mean_inference_time_slice:.4f} seconds")
    print(f"Standard Deviation of Inference Time per Slice: {std_inference_time_slice:.4f} seconds")

    print(f"Frames per second (FPS): {fps:.2f}")
    print(f"Mean Memory Usage per Volume: {mean_memory_usage / (1024**2):.2f} MB")  # Convert to MB
    print(f"Standard Deviation of Memory Usage: {std_memory_usage / (1024**2):.2f} MB")  # Convert to MB

    model_detection_iou = np.array(detection_iou).mean()
    model_detection_iou_std = np.array(detection_iou).std()
    
    model_class_mean_iou, model_class_std_iou = compute_mean_iou_imagewise_from_cumulator(TP, FP, FN, TN, exclude_empty=True, return_std=True)
    model_class_mean_dice, model_class_std_dice = compute_mean_dice_imagewise_from_cumulator(TP, FP, FN, TN, exclude_empty=True, return_std=True)

    model_iou_mass_volume , model_iou_mass_volume_std = compute_iou_imagewise_from_cumulator(TP, FP, FN, TN, exclude_empty=False, return_std=True)
    model_iou_mass_volume_no_empty, model_iou_mass_volume_no_empty_std =compute_iou_imagewise_from_cumulator(TP, FP, FN, TN, exclude_empty=True, return_std=True)
    model_iou_mass_volume_no_empty_optimistic, model_iou_mass_volume_no_empty_optimistic_std =compute_iou_imagewise_from_cumulator(TP, FP, FN, TN, exclude_empty=True, exclude_empty_only_gt=True, return_std=True)
    
    
    model_dice_mass_volume, model_dice_mass_volume_std = compute_dice_imagewise_from_cumulator(TP, FP, FN, TN,exclude_empty=False, return_std=True)
    model_dice_mass_volume_no_empty, model_dice_mass_volume_no_empty_std = compute_dice_imagewise_from_cumulator(TP, FP, FN, TN,exclude_empty=True, return_std=True)
    model_dice_mass_volume_no_empty_optimistic, model_dice_mass_volume_no_empty_optimistic_std = compute_dice_imagewise_from_cumulator(TP, FP, FN, TN,exclude_empty=True, exclude_empty_only_gt=True,return_std=True)
    
    model_mean_accuracy_no_empty, model_mean_accuracy_no_empty_std = compute_accuracy_from_cumulator(TP, FP, FN, TN, exclude_empty=True, is_mean=True, return_std=True)
    model_mean_precision_no_empty,model_mean_precision_no_empty_std = compute_precision_from_cumulator(TP, FP, FN, TN, exclude_empty=True, is_mean=True, return_std=True)
    model_mean_recall_no_empty, model_mean_recall_no_empty_std = compute_recall_from_cumulator(TP, FP, FN, TN, exclude_empty=True, is_mean=True, return_std=True)
    model_mean_f1_no_empty, model_mean_f1_no_empty_std = compute_f1_from_cumulator(TP, FP, FN, TN, exclude_empty=True, is_mean=True, return_std=True)
        
    model_accuracy_excluding_cases, model_accuracy_excluding_cases_std = compute_accuracy_excluding_cases(TP, FP, FN, TN, return_std=True)
    model_precision_excluding_cases,model_precision_excluding_cases_std =compute_precision_excluding_cases_from_cumulator(TP, FP, FN, TN, return_std=True)
    model_recall_excluding_cases,model_recall_excluding_cases_std  =compute_recall_excluding_cases_from_cumulator(TP, FP, FN, TN, return_std=True)

    model_accuracy_no_empty, model_accuracy_no_empty_std = compute_accuracy_from_cumulator(TP, FP, FN, TN, exclude_empty=True, is_mean=False, return_std=True)
    model_precision_no_empty,model_precision_no_empty_std =compute_precision_excluding_cases_from_cumulator(TP, FP, FN, TN, return_std=True,exclude_only_zero_denominator=True)
    model_recall_no_empty,model_recall_no_empty_std  = compute_recall_excluding_cases_from_cumulator(TP, FP, FN, TN, return_std=True,exclude_only_zero_denominator=True)

    model_f1_no_empty,model_f1_no_empty_std = compute_f1_excluding_cases_from_cumulator(TP, FP, FN, TN, return_std=True,exclude_only_zero_denominator=True)
    
    
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
    print("MODEL IOU MASS VOLUME NO EMPTY OPTIMISTIC STD ", model_iou_mass_volume_no_empty_optimistic_std)
    
    print("MODEL DICE MASS VOLUME ", model_dice_mass_volume)
    print("MODEL DICE MASS VOLUME STD ", model_dice_mass_volume_std)
    print()
    print("MODEL DICE MASS VOLUME NO EMPTY ", model_dice_mass_volume_no_empty)
    print("MODEL DICE MASS VOLUME NO EMPTY STD ", model_dice_mass_volume_no_empty_std)
    print()
    print("MODEL DICE MASS VOLUME NO EMPTY OPTIMISTIC ", model_dice_mass_volume_no_empty_optimistic)
    print("MODEL DICE MASS VOLUME NO EMPTY OPTIMISTIC STD ", model_dice_mass_volume_no_empty_optimistic_std)
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
    print("MODEL ACCURACY EXCLUDING CASES ",  model_accuracy_excluding_cases)
    print("MODEL ACCURACY EXCLUDING CASES STD ",  model_accuracy_excluding_cases_std)
    print()
    print("MODEL PRECISION EXCLUDING CASES ",  model_precision_excluding_cases)
    print("MODEL PRECISION EXCLUDING CASES STD ",  model_precision_excluding_cases_std)
    print()
    print("MODEL RECALL EXCLUDING CASES ", model_recall_excluding_cases)
    print("MODEL RECALL EXCLUDING CASES STD ", model_recall_excluding_cases_std)
    print()
    print("MODEL ACCURACY NO EMPTY ",  model_accuracy_no_empty)
    print("MODEL ACCURACY NO EMPTY STD ",  model_accuracy_no_empty_std)
    print()
    print("MODEL PRECISION NO EMPTY",  model_precision_no_empty)
    print("MODEL PRECISION NO EMPTY STD ",  model_precision_no_empty_std)
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


            miou_scores = compute_mean_iou_imagewise_from_cumulator(tp, fp, fn, tn, exclude_empty=False, return_std=False,reduce_mean=False)
            mdice_scores = compute_mean_dice_imagewise_from_cumulator(tp, fp, fn, tn, exclude_empty=False, return_std=False,reduce_mean=False)
            mf1_scores = compute_f1_from_cumulator(tp, fp, fn, tn, exclude_empty=False, is_mean=True, return_std=False,reduce_mean=False)


            scores_dict = {
                 'miou': miou_scores.squeeze().tolist(),
                 'mdice': mdice_scores.squeeze().tolist(),
                 "mf1": mf1_scores.squeeze().tolist(),
            }
            return scores_dict