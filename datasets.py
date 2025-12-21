# datasets.py

from __future__ import annotations
import os
import glob
from typing import Dict, Tuple, Any, List
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from sklearn.model_selection import train_test_split


# Fish Segmentation Dataset Class (Takes pre-split paths)
class FishSegmentationDataset(Dataset):
   
    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        image_size: Tuple[int, int] = (128, 128),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        augment: bool = True,
    ) -> None:
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.augment = augment
        self.image_paths = image_paths
        self.mask_paths = mask_paths

        if not self.image_paths:
            raise ValueError("No image_paths were provided to the dataset.")
        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError(f"Image/mask count mismatch: {len(self.image_paths)} vs {len(self.mask_paths)}")

        self.interp_bilinear = InterpolationMode.BILINEAR
        self.interp_nearest = InterpolationMode.NEAREST

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_pair(self, idx: int) -> Tuple[Image.Image, Image.Image]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        return img, mask

    def _apply_augmentations(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if torch.rand(1).item() < 0.5:
            img = TF.hflip(img)  # type: ignore[arg-type]
            mask = TF.hflip(mask)  # type: ignore[arg-type]
        if torch.rand(1).item() < 0.5:
            img = TF.vflip(img)  # type: ignore[arg-type]
            mask = TF.vflip(mask)  # type: ignore[arg-type]
        if torch.rand(1).item() < 0.3:
            angle = float(torch.randint(-15, 15, (1,)).item())
            img = TF.rotate(img, angle, interpolation=self.interp_bilinear)  # type: ignore[arg-type]
            mask = TF.rotate(mask, angle, interpolation=self.interp_nearest)  # type: ignore[arg-type]
        return img, mask

    def _preprocess(self, img: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        img = TF.resize(img, list(self.image_size), interpolation=self.interp_bilinear)  # type: ignore[arg-type]
        mask = TF.resize(mask, list(self.image_size), interpolation=self.interp_nearest)  # type: ignore[arg-type]
        img_t = TF.to_tensor(img)
        img_t = TF.normalize(img_t, mean=list(self.mean), std=list(self.std))
        mask_arr = np.array(mask, dtype=np.uint8)
        mask_bin = (mask_arr > 127).astype(np.int64) 
        mask_t = torch.as_tensor(mask_bin, dtype=torch.long)
        return img_t, mask_t

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img, mask = self._load_pair(idx)
        if self.augment:
            img, mask = self._apply_augmentations(img, mask)
        img_t, mask_t = self._preprocess(img, mask)
        return {
            "image": img_t,
            "mask": mask_t,
            "mean": torch.tensor(self.mean, dtype=torch.float32),
            "std": torch.tensor(self.std, dtype=torch.float32),
        }

# Fish Dataset Loader Function

def get_fish_data_loaders(
    root_dir: str,
    image_size: Tuple[int, int],
    batch_size: int,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
        
    # --- 1. Find all file paths ---
    print(f"Scanning for Fish Dataset in: {root_dir}")
    image_paths, mask_paths = [], []
    for species in sorted(os.listdir(root_dir)):
        species_root = os.path.join(root_dir, species)
        
        if not os.path.isdir(species_root):
            continue
        img_dir = os.path.join(species_root, species)
        mask_dir = os.path.join(species_root, f"{species} GT")
       
        if not (os.path.isdir(img_dir) and os.path.isdir(mask_dir)):
            continue

        for img_file in glob.glob(os.path.join(img_dir, "*.png")):
            fname = os.path.basename(img_file)
            mask_file = os.path.join(mask_dir, os.path.splitext(fname)[0] + ".png")
            if os.path.exists(mask_file):
                image_paths.append(img_file)
                mask_paths.append(mask_file)
    
    if not image_paths:
        raise FileNotFoundError(f"No matched image/mask pairs found in {root_dir}. Check paths.")
    print(f"Found {len(image_paths)} total image/mask pairs.")

    print("Splitting dataset...")
    train_val_imgs, test_imgs, train_val_masks, test_masks = train_test_split(
        image_paths, mask_paths, test_size=test_size, random_state=random_state
    )
    
    relative_val_size = val_size / (1.0 - test_size)
    
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        train_val_imgs, train_val_masks, test_size=relative_val_size, random_state=random_state
    )

    print("\nDataset splits:")
    print(f"  Train:      {len(train_imgs)} images")
    print(f"  Validation: {len(val_imgs)} images")
    print(f"  Test:       {len(test_imgs)} images")
    print(f"  Total:      {len(image_paths)} images")
    print("-" * 30)

    # --- 3. Create Dataset objects ---
    train_dataset = FishSegmentationDataset(
        image_paths=train_imgs, 
        mask_paths=train_masks, 
        image_size=image_size, 
        augment=True
    )
    val_dataset = FishSegmentationDataset(
        image_paths=val_imgs, 
        mask_paths=val_masks, 
        image_size=image_size, 
        augment=False
    )
    test_dataset = FishSegmentationDataset(
        image_paths=test_imgs, 
        mask_paths=test_masks, 
        image_size=image_size, 
        augment=False
    )

    # --- 4. Create DataLoader objects ---
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
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
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader