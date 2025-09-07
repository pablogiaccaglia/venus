import os
import sys
import pprint
import numpy as np

import torch

import monai
from monai.config import print_config
from monai.data import CacheDataset
from monai.transforms import LoadImaged, EnsureChannelFirstd, Compose

from breast_segmentation.config.settings import config
from breast_segmentation.transforms.compose import Preprocess
from breast_segmentation.inference.breadm_dataset_aware_test import (
    test_dataset_aware_ensemble,
    test_dataset_aware_no_patches,
    test_dataset_aware_fusion,
)
from breast_segmentation.metrics.losses import (
    CABFL, SurfaceLossBinary, AsymmetricUnifiedFocalLoss,
    AsymmetricFocalLoss, AsymmetricFocalTverskyLoss, SoftDiceLoss,
)
from breast_segmentation.data.dataset import get_image_label_files, create_data_dicts


pp = pprint.PrettyPrinter(indent=4)


def _section(title: str) -> None:
    print(f"\n[ {title} ]\n")


def _info(message: str) -> None:
    print(f"- {message}")


def main():
    torch.set_float32_matmul_precision('medium')
    _section("Environment & MONAI Config")
    print_config()

    # Register loss classes for checkpoint loading
    sys.modules['__main__'].CABFL = CABFL
    sys.modules['__main__'].SurfaceLossBinary = SurfaceLossBinary
    sys.modules['__main__'].AsymmetricUnifiedFocalLoss = AsymmetricUnifiedFocalLoss
    sys.modules['__main__'].AsymmetricFocalLoss = AsymmetricFocalLoss
    sys.modules['__main__'].AsymmetricFocalTverskyLoss = AsymmetricFocalTverskyLoss
    sys.modules['__main__'].SoftDiceLoss = SoftDiceLoss

    _section("Data Setup - BREADM Dataset")
    NUM_WORKERS = config.NUM_WORKERS
    dataset_base_path = config.DATASET_BASE_PATH_BREADM
    CHECKPOINTS_DIR = config.checkpoints_dir_breadm
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    _info(f"Checkpoint directory verified: {CHECKPOINTS_DIR}")

    image_type = "VIBRANT+C2"
    train_images, train_labels = get_image_label_files(dataset_base_path, "train", image_type)
    val_images, val_labels = get_image_label_files(dataset_base_path, "val", image_type)
    test_images, test_labels = get_image_label_files(dataset_base_path, "test", image_type)

    test_dicts = create_data_dicts(test_images, test_labels)
    _info(f"Test samples: {len(test_dicts)}")

    _section("Create Test Datasets")
    sub_third_images_path_prefixes = ("VIBRANT+C2", "SUB2")

    mean_no_thorax_third_sub = 10.217766761779785
    std_no_thorax_third_sub = 26.677101135253906
    mean_patches_sub = 20.63081550598144
    std_patches_sub = 35.328887939453125

    test_transforms_no_thorax_third_sub = Compose([
        LoadImaged(keys=["image", "label"], image_only=False, reader=monai.data.PILReader(converter=lambda im: im.convert("L"))),
        EnsureChannelFirstd(keys=["image", "label"]),
        monai.transforms.Rotate90d(keys=["image", "label"]),
        Preprocess(
            keys=None,
            mode='test',
            dataset="BREADM",
            subtracted_images_path_prefixes=sub_third_images_path_prefixes,
            subtrahend=mean_no_thorax_third_sub,
            divisor=std_no_thorax_third_sub,
            get_patches=False,
            get_boundaryloss=True,
        ),
    ])

    test_transforms_patches_sub = Compose([
        LoadImaged(keys=["image", "label"], image_only=False, reader=monai.data.PILReader(converter=lambda im: im.convert("L"))),
        EnsureChannelFirstd(keys=["image", "label"]),
        monai.transforms.Rotate90d(keys=["image", "label"]),
        Preprocess(
            keys=None,
            mode='test',
            dataset="BREADM",
            subtracted_images_path_prefixes=sub_third_images_path_prefixes,
            subtrahend=mean_patches_sub,
            divisor=std_patches_sub,
            get_patches=True,
            get_boundaryloss=True,
        ),
    ])
    print("Test transforms created")

    # Build per-patient datasets dict similar to private script but using test split
    datasets = {}
    # in BREADM, patients correspond to test image stems up to first underscore
    # we'll group test_dicts by patient id extracted from path
    def _extract_pid(path: str) -> str:
        base = os.path.basename(path)
        return base.split("_")[0]

    pid_to_items = {}
    for item in test_dicts:
        pid = _extract_pid(item["image"])
        pid_to_items.setdefault(pid, []).append(item)

    for pid, items in pid_to_items.items():
        no_thorax_sub_test_ds = CacheDataset(data=items, transform=test_transforms_no_thorax_third_sub, num_workers=NUM_WORKERS)
        patches_sub_test_ds = CacheDataset(data=items, transform=test_transforms_patches_sub, num_workers=NUM_WORKERS)
        datasets[pid] = {
            "no_thorax_sub_test_ds": no_thorax_sub_test_ds,
            "patches_sub_test_ds": patches_sub_test_ds,
        }

    _section("Model Checkpoint Paths")
    model_paths = {
        'venus_tiny': f'{CHECKPOINTS_DIR}/venus-tiny-best.ckpt',
        'unetplusplus': f'{CHECKPOINTS_DIR}/unetplusplus_model.ckpt',
        'skinny': f'{CHECKPOINTS_DIR}/skinny_model.ckpt',
        'resnet50': f'{CHECKPOINTS_DIR}/resnet50-model.ckpt',
        'fcn': f'{CHECKPOINTS_DIR}/unetplusplus_model.ckpt',
        'segnet': f'{CHECKPOINTS_DIR}/segnet_model_large.ckpt',
        'swin': f'{CHECKPOINTS_DIR}/swin_model.ckpt',
        'resnet50_patches': f'{CHECKPOINTS_DIR}/resnet50-patches.ckpt',
        'resnet18_patches': f'{CHECKPOINTS_DIR}/resnet18-patches.ckpt',
        'unetplusplus_patches': f'{CHECKPOINTS_DIR}/unetplusplus-patches-cabfl.ckpt',
    }
    
    print("Available models:")
    for name, path in model_paths.items():
        if os.path.exists(path):
            print(f"  ✓ {name}: {path}")
        else:
            print(f"  ✗ {name}: {path} (not found)")

    _section("Test Baseline Models")
    baseline_tests = [
        ('unetplusplus', 'UNet++', 'unetplusplus'),
        ('skinny', 'SkinnyNet', 'skinny'),
        ('fcn', 'FCN', 'unetplusplus'),
        ('segnet', 'SegNet', 'segnet'),
        ('swin', 'Swin-UNETR', 'swin_unetr'),
        ('resnet50', 'ResNet50', 'resnet50'),
    ]
    baseline_results = {}
    for model_key, model_name, arch_name in baseline_tests:
        if os.path.exists(model_paths[model_key]):
            print(f"Testing {model_name} model...")
            # use all patient ids from datasets dict
            patient_ids = list(datasets.keys())
            result = test_dataset_aware_no_patches(
                model_path=model_paths[model_key],
                patient_ids=patient_ids,
                datasets=datasets,
                dataset_key="no_thorax_sub_test_ds",
                filter=False,
                get_scores_for_statistics=False,
                get_only_masses=False,
                arch_name=arch_name,
                strict=True,
                subtracted=True,
            )
            baseline_results[model_key] = result
            print(f"\n{model_name} Results:")
            pp.pprint(result)
        else:
            print(f"{model_paths[model_key]} not found.")
            baseline_results[model_key] = None
    print(f"\nCompleted {len([r for r in baseline_results.values() if r is not None])} baseline model tests.")

    _section("Test VENUS Model")
    if os.path.exists(os.path.join(CHECKPOINTS_DIR, "venus-large-best.ckpt")):
        print("Testing VENUS Large ...")
        patient_ids = list(datasets.keys())
        scores_for_statistics_fusion_large2 = test_dataset_aware_fusion(
            model_path=model_paths['venus_large'],
            patient_ids=patient_ids,
            datasets=datasets,
            whole_dataset_key="no_thorax_sub_test_ds",
            patches_dataset_key="patches_sub_test_ds",
            use_simple_fusion=True,
            use_decoder_attention=True,
            strict=True,
            filter=False,
            subtracted=True,
            get_scores_for_statistics=False,
            get_only_masses=False,
            base_channels=64,
        )
        print("\nVENUS Large:")
        pp.pprint(scores_for_statistics_fusion_large2)
    else:
        print("venus-large-best.ckpt not found.")

    _section("Test Ensemble Model")
    ensemble_tests = [
        ('venus_large', 'resnet18-patches', False, 64, 'VENUS Large + ResNet18 patches'),
        ('venus_large', 'resnet18-patches', True, 64, 'VENUS Large + ResNet18 patches (filtered)'),
    ]
    ensemble_results = {}
    for whole_key, patches_key, use_filter, base_channels, description in ensemble_tests:
        if os.path.exists(model_paths.get(whole_key, "")) and os.path.exists(model_paths.get(patches_key, "")):
            print(f"Testing Ensemble: {description}...")
            patient_ids = list(datasets.keys())
            result = test_dataset_aware_ensemble(
                model_whole_path=model_paths[whole_key],
                model_patches_path=model_paths[patches_key],
                patient_ids=patient_ids,
                datasets=datasets,
                whole_dataset_key="no_thorax_sub_test_ds",
                patches_dataset_key="patches_sub_test_ds",
                filter=use_filter,
                get_scores_for_statistics=False,
                get_only_masses=False,
                subtracted=True,
                base_channels=base_channels,
                use_decoder_attention = True,
                use_simple_fusion = True
            )
            ensemble_key = f"{whole_key}+{patches_key}{'_filtered' if use_filter else ''}"
            ensemble_results[ensemble_key] = result
            print(f"\n{description} Results:")
            pp.pprint(result)
        else:
            print(f"Required models not found for ensemble: {description}")
            ensemble_key = f"{whole_key}+{patches_key}{'_filtered' if use_filter else ''}"
            ensemble_results[ensemble_key] = None
    print(f"\nCompleted {len([r for r in ensemble_results.values() if r is not None])} ensemble model tests.")


if __name__ == "__main__":
    main()


