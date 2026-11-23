from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ColorJitter, InterpolationMode
from torchvision.transforms import functional as TF


ID_PATTERN = re.compile(r"(ISIC_\d+)", re.IGNORECASE)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def isic_id(path: Path) -> str:
    match = ID_PATTERN.search(path.stem)
    if not match:
        raise ValueError(f"无法从文件名提取 ISIC ID: {path.name}")
    return match.group(1).upper()


def _index_files(folder: str | Path) -> dict[str, Path]:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"数据目录不存在: {folder}")
    indexed: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            key = isic_id(path)
            if key in indexed:
                raise ValueError(f"发现重复 ID {key}: {indexed[key]} 和 {path}")
            indexed[key] = path
    return indexed


def discover_pairs(image_dir: str | Path, mask_dir: str | Path) -> tuple[list[dict[str, Path | str]], dict[str, list[str]]]:
    images = _index_files(image_dir)
    masks = _index_files(mask_dir)
    shared = sorted(images.keys() & masks.keys())
    pairs = [{"id": key, "image": images[key], "mask": masks[key]} for key in shared]
    issues = {
        "images_without_masks": sorted(images.keys() - masks.keys()),
        "masks_without_images": sorted(masks.keys() - images.keys()),
    }
    return pairs, issues


def load_split_ids(path: str | Path) -> list[str]:
    split_path = Path(path)
    if not split_path.exists():
        raise FileNotFoundError(f"划分文件不存在，请先运行 02_prepare_data.py: {split_path}")
    return [line.strip().upper() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]


class ISICDataset(Dataset):
    """ISIC2018二值分割数据集。

    图像和Mask按ISIC编号配对；Mask始终使用最近邻插值；所有几何增强同步
    应用于图像和Mask；ImageNet归一化用于匹配预训练VGG16编码器。
    """
    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        ids: Iterable[str] | None = None,
        image_size: int = 256,
        augment: bool = False,
        normalize: str = "imagenet",
    ) -> None:
        pairs, issues = discover_pairs(image_dir, mask_dir)
        if issues["images_without_masks"] or issues["masks_without_images"]:
            raise ValueError(f"数据存在未配对文件: {issues}")
        if ids is not None:
            wanted = {value.upper() for value in ids}
            pairs = [pair for pair in pairs if pair["id"] in wanted]
            found = {str(pair["id"]) for pair in pairs}
            missing = sorted(wanted - found)
            if missing:
                raise ValueError(f"划分文件中有 {len(missing)} 个 ID 无法配对，例如: {missing[:5]}")
        self.pairs = pairs
        self.image_size = int(image_size)
        self.augment = augment
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.pairs)

    def _augment(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        """同步执行几何增强，并只对RGB图像执行颜色扰动。"""
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        angle = random.uniform(-30.0, 30.0)
        translate = [int(random.uniform(-0.05, 0.05) * self.image_size) for _ in range(2)]
        scale = random.uniform(0.90, 1.10)
        image = TF.affine(image, angle, translate, scale, shear=[0.0, 0.0], interpolation=InterpolationMode.BILINEAR, fill=0)
        mask = TF.affine(mask, angle, translate, scale, shear=[0.0, 0.0], interpolation=InterpolationMode.NEAREST, fill=0)
        if random.random() < 0.5:
            image = ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02)(image)
        return image, mask

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        pair = self.pairs[index]
        with Image.open(pair["image"]) as src:
            image = src.convert("RGB")
        with Image.open(pair["mask"]) as src:
            mask = src.convert("L")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        if self.augment:
            image, mask = self._augment(image, mask)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        if self.normalize == "imagenet":
            image_array = (image_array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
                [0.229, 0.224, 0.225], dtype=np.float32
            )
        mask_array = (np.asarray(mask, dtype=np.uint8) >= 128).astype(np.float32)
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1).copy())
        mask_tensor = torch.from_numpy(mask_array[None, ...].copy())
        return {"image": image_tensor, "mask": mask_tensor, "id": str(pair["id"])}


def create_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

