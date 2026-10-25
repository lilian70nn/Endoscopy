from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


class SegmentationDataset(Dataset):
    def __init__(self, samples, image_size=224, train=False):
        self.samples = samples
        self.image_size = image_size
        self.train = train

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        with Image.open(mask_path) as m:
            mask = m.convert("L")

        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
        )

        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )

        if self.train:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if random.random() < 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

        image = TF.to_tensor(image)
        image = self.normalize(image)
        mask = torch.from_numpy((np.array(mask) > 127).astype(np.int64))

        return image, mask


def initialize_segmentation_dataloaders(
    data_dir="./data/segmented-images",
    batch_size=16,
    image_size=224,
    num_workers=2,
    seed=42,
):
    data_dir = Path(data_dir)

    image_dir = data_dir / "images"
    mask_dir = data_dir / "masks"

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    image_extensions = {".jpg", ".jpeg", ".png"}
    image_paths = sorted(
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in image_extensions
    )

    mask_paths = {
        p.stem: p
        for p in mask_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in image_extensions
    }

    samples = []
    missing_masks = []

    for image_path in image_paths:
        mask_path = mask_paths.get(image_path.stem)

        if mask_path is None:
            missing_masks.append(image_path.name)
            continue

        samples.append((image_path, mask_path))

    if missing_masks:
        print(f"[Segmentation] Images without masks: {len(missing_masks)}")

    if len(samples) < 2:
        raise ValueError(f"Not enough image-mask pairs found in {data_dir}")

    train_samples, val_samples = train_test_split(
        samples,
        test_size=0.15,
        random_state=seed,
        shuffle=True,
    )

    train_loader = DataLoader(
        SegmentationDataset(
            train_samples,
            image_size=image_size,
            train=True,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        SegmentationDataset(
            val_samples,
            image_size=image_size,
            train=False,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"[Segmentation] Total image-mask pairs: {len(samples)}")
    print(f"[Segmentation] Training images: {len(train_samples)}")
    print(f"[Segmentation] Validation images: {len(val_samples)}")

    return train_loader, val_loader