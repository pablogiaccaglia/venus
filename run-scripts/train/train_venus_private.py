import os
import pprint

import torch
import numpy as np
import lightning.pytorch as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

from sklearn.model_selection import train_test_split

from monai.config import print_config
from monai.data import CacheDataset, DataLoader
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged
import monai

from breast_segmentation.config.settings import config
from breast_segmentation.utils.seed import set_deterministic_mode, seed_worker, reseed
from breast_segmentation.data.private_dataset import (
    PATIENT_INFO,
    get_train_val_test_dicts,
)
from breast_segmentation.data.dataset import PairedDataset, PairedDataLoader
from breast_segmentation.data import custom_collate_no_patches, custom_collate
from breast_segmentation.transforms.compose import Preprocess
from breast_segmentation.models.fusion_module import BreastFusionModel
from breast_segmentation.models.lightning_module import BreastSegmentationModel


pp = pprint.PrettyPrinter(indent=4)


def _section(title: str) -> None:
    print(f"\n[ {title} ]\n")


def _info(message: str) -> None:
    print(f"- {message}")


def _build_statistics_loader_global(batch_size: int, num_workers: int, train_dicts):
    statistics_transforms = Compose([
        LoadImaged(
            keys=["image", "label"],
            image_only=False,
            reader=monai.data.NumpyReader(),
        ),
        EnsureChannelFirstd(keys=["image", "label"]),
        Preprocess(
            keys=None,
            mode='statistics',
            dataset="private",
            subtracted_images_path_prefixes=("Dataset-arrays-4-FINAL", "Dataset-arrays-FINAL"),
            get_patches=False,
            get_boundaryloss=False,
        ),
    ])

    ds = CacheDataset(data=train_dicts, transform=statistics_transforms, num_workers=num_workers)
    g = set_deterministic_mode(config.SEED)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=False,
        drop_last=False,
        collate_fn=custom_collate_no_patches,
    )
    return loader


def _build_statistics_loader_patches(batch_size: int, num_workers: int, train_dicts):
    statistics_transforms = Compose([
        LoadImaged(
            keys=["image", "label"],
            image_only=False,
            reader=monai.data.NumpyReader(),
        ),
        EnsureChannelFirstd(keys=["image", "label"]),
        Preprocess(
            keys=None,
            mode='statistics',
            dataset="private",
            subtracted_images_path_prefixes=("Dataset-arrays-4-FINAL", "Dataset-arrays-FINAL"),
            get_patches=True,
            get_boundaryloss=False,
        ),
    ])

    ds = CacheDataset(data=train_dicts, transform=statistics_transforms, num_workers=num_workers)
    g = set_deterministic_mode(config.SEED)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=False,
        drop_last=False,
        collate_fn=custom_collate,
    )
    return loader


def _get_mean_std_dataloader(dataloader) -> tuple[float, float]:
    sum_of_images = 0.0
    sum_of_squares = 0.0
    num_pixels = 0
    for batch in dataloader:
        if batch is None:
            continue
        image = batch["image"]
        sum_of_images += image.sum()
        sum_of_squares += (image ** 2).sum()
        num_pixels += image.numel()
    mean = (sum_of_images / num_pixels).item()
    std_dev = ((sum_of_squares / num_pixels - (sum_of_images / num_pixels) ** 2) ** 0.5).item()
    return mean, std_dev


def train_venus_private():
    torch.set_float32_matmul_precision("medium")
    _section("Environment & MONAI Config")
    print_config()

    NUM_WORKERS = config.NUM_WORKERS
    SEED = config.SEED
    batch_size = config.BATCH_SIZE

    dataset_base_path = config.DATASET_BASE_PATH_PRIVATE
    CHECKPOINTS_DIR = config.checkpoints_dir_private

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _info(f"Using device: {device}")
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    _info(f"Checkpoint directory: {CHECKPOINTS_DIR}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    _section("Data preparation: patient split and dicts")
    patient_ids = list(PATIENT_INFO.keys())
    x_train_val, x_test = train_test_split(patient_ids, test_size=0.2, random_state=SEED)
    x_train, x_val = train_test_split(x_train_val, test_size=0.25, random_state=SEED)
    train_dicts, val_dicts, test_dicts = get_train_val_test_dicts(dataset_base_path, x_train, x_val, x_test)
    _info(f"Train dicts: {len(train_dicts)} | Val dicts: {len(val_dicts)} | Test dicts: {len(test_dicts)}")

    _section("Compute normalization statistics (global and patches)")
    _info("Global stats: building loader")
    stats_loader_global = _build_statistics_loader_global(batch_size, NUM_WORKERS, train_dicts)
    _info("Global stats: computing")
    mean_global, std_global = _get_mean_std_dataloader(stats_loader_global)
    _info(f"Global mean/std: {mean_global:.6f} / {std_global:.6f}")

    _info("Patches stats: building loader")
    stats_loader_patches = _build_statistics_loader_patches(batch_size, NUM_WORKERS, train_dicts)
    _info("Patches stats: computing")
    mean_patches, std_patches = _get_mean_std_dataloader(stats_loader_patches)
    _info(f"Patches mean/std: {mean_patches:.6f} / {std_patches:.6f}")

    _section("Build transforms and CacheDatasets (global and patches)")
    sub_third_images_path_prefixes = ("Dataset-arrays-4-FINAL", "Dataset-arrays-FINAL")

    test_transforms_no_thorax_third_sub = Compose([
        LoadImaged(keys=["image", "label"], image_only=False, reader=monai.data.NumpyReader()),
        EnsureChannelFirstd(keys=["image", "label"]),
        Preprocess(
            keys=None,
            mode='test',
            dataset="private",
            subtracted_images_path_prefixes=sub_third_images_path_prefixes,
            subtrahend=mean_global,
            divisor=std_global,
            get_patches=False,
            get_boundaryloss=True,
        ),
    ])
    train_transforms_no_thorax_third_sub = Compose([
        LoadImaged(keys=["image", "label"], image_only=False, reader=monai.data.NumpyReader()),
        EnsureChannelFirstd(keys=["image", "label"]),
        Preprocess(
            keys=None,
            mode='train',
            dataset="private",
            subtracted_images_path_prefixes=sub_third_images_path_prefixes,
            subtrahend=mean_global,
            divisor=std_global,
            get_patches=False,
            get_boundaryloss=True,
        ),
    ])
    train_transforms_patches_sub = Compose([
        LoadImaged(keys=["image", "label"], image_only=False, reader=monai.data.NumpyReader()),
        EnsureChannelFirstd(keys=["image", "label"]),
        Preprocess(
            keys=None,
            mode='train',
            dataset="private",
            subtracted_images_path_prefixes=sub_third_images_path_prefixes,
            subtrahend=mean_patches,
            divisor=std_patches,
            get_patches=True,
            get_boundaryloss=True,
        ),
    ])
    test_transforms_patches_sub = Compose([
        LoadImaged(keys=["image", "label"], image_only=False, reader=monai.data.NumpyReader()),
        EnsureChannelFirstd(keys=["image", "label"]),
        Preprocess(
            keys=None,
            mode='test',
            dataset="private",
            subtracted_images_path_prefixes=sub_third_images_path_prefixes,
            subtrahend=mean_patches,
            divisor=std_patches,
            get_patches=True,
            get_boundaryloss=True,
        ),
    ])

    train_ds_no_thorax_third_sub = CacheDataset(data=train_dicts, transform=train_transforms_no_thorax_third_sub, num_workers=NUM_WORKERS)
    val_ds_no_thorax_third_sub = CacheDataset(data=val_dicts, transform=test_transforms_no_thorax_third_sub, num_workers=NUM_WORKERS)
    test_ds_no_thorax_third_sub = CacheDataset(data=test_dicts, transform=test_transforms_no_thorax_third_sub, num_workers=NUM_WORKERS)
    train_ds_patches_sub = CacheDataset(data=train_dicts, transform=train_transforms_patches_sub, num_workers=NUM_WORKERS)
    val_ds_patches_sub = CacheDataset(data=val_dicts, transform=test_transforms_patches_sub, num_workers=NUM_WORKERS)
    test_ds_patches_sub = CacheDataset(data=test_dicts, transform=test_transforms_patches_sub, num_workers=NUM_WORKERS)

    _section("Build fusion and patches DataLoaders")
    g = reseed()
    train_loader_fusion_sub = PairedDataLoader(
        train_ds_no_thorax_third_sub,
        train_ds_patches_sub,
        batch_size=batch_size,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=True,
        drop_last=False,
        num_workers=NUM_WORKERS,
        augment=False,
    )
    val_loader_fusion_sub = PairedDataLoader(
        val_ds_no_thorax_third_sub,
        val_ds_patches_sub,
        batch_size=batch_size,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        augment=False,
    )
    test_loader_fusion_sub = PairedDataLoader(
        test_ds_no_thorax_third_sub,
        test_ds_patches_sub,
        batch_size=batch_size,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        augment=False,
    )

    train_loader_patches_sub = DataLoader(
        train_ds_patches_sub,
        batch_size=batch_size,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=True,
        drop_last=False,
    )
    val_loader_patches_sub = DataLoader(
        val_ds_patches_sub,
        batch_size=batch_size,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=False,
        drop_last=False,
    )
    test_loader_patches_sub = DataLoader(
        test_ds_patches_sub,
        batch_size=batch_size,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=False,
        drop_last=False,
    )

    _section("Training VENUS Fusion (CABFL)")
    import gc
    gc.collect()
    with torch.no_grad():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    g = reseed()
    model_fusion_sub_cabl = BreastFusionModel(
        arch="venus",
        encoder_name=None,
        use_boundary_loss=True,
        use_simple_fusion=True,
        loss_function="cabfl",
        loss_kwargs={"idc": [1], "weight_aufl": 0.5, "delta": 0.7, "gamma": 0.4},
        in_channels=1,
        out_classes=1,
        batch_size=batch_size,
        len_train_loader=len(train_ds_no_thorax_third_sub)//batch_size,
    )
    es = EarlyStopping(monitor="valid_loss", mode="min", patience=config.EARLY_STOPPING_PATIENCE)
    cc_fusion_sub_cabl = ModelCheckpoint(
        monitor="valid_loss",
        save_top_k=1,
        mode="min",
        filename='venus-fusion-cabl-{epoch:02d}-{valid_loss:.2f}',
        dirpath=CHECKPOINTS_DIR,
        auto_insert_metric_name=False,
    )
    trainer_fusion_sub_cabl = L.Trainer(
        devices=1,
        accelerator='auto',
        max_epochs=config.MAX_EPOCHS,
        callbacks=[es, cc_fusion_sub_cabl],
        log_every_n_steps=10,
        gradient_clip_val=1,
        num_sanity_val_steps=1,
        deterministic=False,
    )
    trainer_fusion_sub_cabl.fit(
        model_fusion_sub_cabl,
        train_dataloaders=train_loader_fusion_sub,
        val_dataloaders=val_loader_fusion_sub,
    )
    _info("Fusion training complete. Loading best checkpoint and testing")
    model_fusion_sub_cabl = BreastFusionModel.load_from_checkpoint(
        cc_fusion_sub_cabl.best_model_path,
        use_boundary_loss=True,
        use_simple_fusion=True,
        loss_function="cabfl",
        loss_kwargs={"idc": [1], "weight_aufl": 0.5, "delta": 0.4, "gamma": 0.1},
    )
    test_metrics = trainer_fusion_sub_cabl.test(
        model_fusion_sub_cabl,
        dataloaders=test_loader_fusion_sub,
        verbose=False,
    )
    pp.pprint(test_metrics[0])

    _section("Training ResNet18 baseline on patches (CABFL)")
    g = reseed()
    model_resnet_cabl = BreastSegmentationModel(
        arch="UNet",
        encoder_name="resnet18",
        use_boundary_loss=True,
        loss_function="cabfl",
        loss_kwargs={"idc": [1], "weight_aufl": 0.5, "delta": 0.7, "gamma": 0.4},
        in_channels=1,
        out_classes=1,
        batch_size=batch_size,
        len_train_loader=len(train_ds_patches_sub)//batch_size,
    )
    es_resnet = EarlyStopping(monitor="valid_loss", mode="min", patience=config.EARLY_STOPPING_PATIENCE)
    cc_resnet_cabl = ModelCheckpoint(
        monitor="valid_loss",
        save_top_k=1,
        mode="min",
        filename='resnet18-patches-cabl-{epoch:02d}-{valid_loss:.2f}',
        dirpath=CHECKPOINTS_DIR,
        auto_insert_metric_name=False,
    )
    trainer_resnet_cabl = L.Trainer(
        devices=1,
        accelerator='auto',
        max_epochs=config.MAX_EPOCHS,
        callbacks=[es_resnet, cc_resnet_cabl],
        log_every_n_steps=10,
        gradient_clip_val=1,
        num_sanity_val_steps=1,
        deterministic=False,
    )
    trainer_resnet_cabl.fit(
        model_resnet_cabl,
        train_dataloaders=train_loader_patches_sub,
        val_dataloaders=val_loader_patches_sub,
    )
    _info("ResNet18 training complete. Loading best checkpoint and testing")
    model_resnet_cabl = BreastSegmentationModel.load_from_checkpoint(
        cc_resnet_cabl.best_model_path,
        strict=True,
        use_boundary_loss=True,
        loss_function="cabfl",
        loss_kwargs={"idc": [1], "weight_aufl": 0.5, "delta": 0.4, "gamma": 0.1},
    )
    test_metrics_resnet = trainer_resnet_cabl.test(
        model_resnet_cabl,
        dataloaders=test_loader_patches_sub,
        verbose=False,
    )
    pp.pprint(test_metrics_resnet[0])


def main():
    train_venus_private()


if __name__ == "__main__":
    main()


