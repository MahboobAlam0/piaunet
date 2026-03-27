from __future__ import annotations
import os
import glob
from typing import List, Tuple

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import torchvision.transforms.functional as TF
import cv2
import random


# CLAHE ENHANCEMENT

def apply_clahe(img: Image.Image) -> Image.Image:
    img_np = np.array(img)

    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    lab = cv2.merge((l_channel, a_channel, b_channel))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return Image.fromarray(enhanced)


# MASK PROCESSING

def process_mask(mask_path: str):
    mask = np.array(Image.open(mask_path))

    # handle RGB masks
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    # normalize → binary
    if mask.max() > 1:
        mask = (mask > 127).astype(np.uint8)

    unique_vals = np.unique(mask)

    # remove completely broken masks
    if len(unique_vals) < 2:
        return None

    return mask


# ============================================================
# VALIDATION FILTER (MINIMAL)
# ============================================================
def is_valid_mask(mask_path: str):
    mask = process_mask(mask_path)

    if mask is None:
        return False, 0.0

    fg_ratio = mask.sum() / mask.size

    # only remove extreme garbage
    if fg_ratio < 0.001:
        return False, fg_ratio

    if fg_ratio > 0.98:
        return False, fg_ratio

    return True, fg_ratio


# ============================================================
# DATASET
# ============================================================
class GenericSegmentationDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        image_size: Tuple[int, int] = (256, 256),
        augment: bool = False,
        enhance: bool = False,
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        self.augment = augment
        self.enhance = enhance

        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")

        mask_np = process_mask(self.mask_paths[idx])
        if mask_np is None:
            raise RuntimeError("Invalid mask passed filtering.")

        mask = Image.fromarray(mask_np)

        # ----------------------------
        # Enhancement
        # ----------------------------
        if self.enhance:
            img = apply_clahe(img)

        # ----------------------------
        # Augmentation (PIL SAFE)
        # ----------------------------
        if self.augment:
            if random.random() < 0.5:
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

            if random.random() < 0.5:
                img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        # ----------------------------
        # Resize
        # ----------------------------
        img = img.resize(self.image_size, Image.Resampling.BILINEAR)
        mask = mask.resize(self.image_size, Image.Resampling.NEAREST)

        # ----------------------------
        # Tensor
        # ----------------------------
        img = TF.to_tensor(img)
        img = TF.normalize(img, self.mean, self.std)

        mask = torch.from_numpy(np.array(mask)).long()

        return {"image": img, "mask": mask}


# ============================================================
# SAMPLE WEIGHTS (CLASS BALANCE)
# ============================================================
def compute_sample_weights(mask_paths):
    weights = []

    for m in mask_paths:
        mask = process_mask(m)

        if mask is None:
            weights.append(0.0)
            continue

        fg_ratio = mask.sum() / mask.size

        # inverse weighting
        weight = 1.0 - fg_ratio

        # boost rare objects
        if fg_ratio < 0.1:
            weight *= 2.0

        weights.append(weight)

    weights = np.array(weights)
    weights = weights / (weights.sum() + 1e-8)

    return weights


# ============================================================
# SCAN DATASET
# ============================================================
def scan_aqua(root_dir: str):
    images, masks = [], []

    img_dir = os.path.join(root_dir, "images")
    mask_dir = os.path.join(root_dir, "masks")

    removed = 0
    fg_stats = []

    for img_path in glob.glob(os.path.join(img_dir, "*.jpg")):
        filename = os.path.basename(img_path)
        mask_path = os.path.join(mask_dir, filename.replace(".jpg", ".png"))

        if os.path.exists(mask_path):
            valid, fg_ratio = is_valid_mask(mask_path)

            if valid:
                images.append(img_path)
                masks.append(mask_path)
                fg_stats.append(fg_ratio)
            else:
                removed += 1

    print("\nDataset cleaning:")
    print(f"Removed bad masks: {removed}")
    print(f"Valid samples: {len(images)}")

    if len(fg_stats) > 0:
        print("\nForeground ratio stats:")
        print(f"Min: {np.min(fg_stats):.4f}")
        print(f"Max: {np.max(fg_stats):.4f}")
        print(f"Mean: {np.mean(fg_stats):.4f}")

    return images, masks


# ============================================================
# DATA LOADER
# ============================================================
def get_data_loaders(
    root_dir: str,
    image_size=(256, 256),
    batch_size=4,
    num_workers=0,
):
    images, masks = scan_aqua(root_dir)

    if len(images) == 0:
        raise RuntimeError("No valid data found.")

    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        images, masks, test_size=0.2, random_state=42
    )

    train_dataset = GenericSegmentationDataset(
        train_imgs,
        train_masks,
        image_size=image_size,
        augment=True,
        enhance=True
    )

    val_dataset = GenericSegmentationDataset(
        val_imgs,
        val_masks,
        image_size=image_size,
        augment=False,
        enhance=False
    )

    # ----------------------------
    # BALANCED SAMPLING
    # ----------------------------
    weights = compute_sample_weights(train_masks)

    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader