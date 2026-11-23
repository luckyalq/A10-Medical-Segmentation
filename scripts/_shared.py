from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from datasets.isic_dataset import ISICDataset, create_loader, load_split_ids
from models.vgg_unet import build_segmentation_model
from utils.common import data_paths, load_checkpoint, project_path


def make_dataset(config: dict[str, Any], split: str, augment: bool = False) -> ISICDataset:
    """使用统一配置创建指定划分的数据集。"""
    ids = load_split_ids(project_path(config, "splits", f"{split}.txt"))
    _, image_dir, mask_dir = data_paths(config)
    return ISICDataset(
        image_dir,
        mask_dir,
        ids=ids,
        image_size=config["data"]["image_size"],
        augment=augment,
        normalize=config["data"]["normalize"],
    )


def make_loader(config: dict[str, Any], split: str, augment: bool = False, shuffle: bool = False):
    dataset = make_dataset(config, split, augment)
    return create_loader(dataset, config["training"]["batch_size"], shuffle, config["data"]["num_workers"])


def build_unet(config: dict[str, Any], dropout: float = 0.0, pretrained: bool | None = None) -> torch.nn.Module:
    """按配置构建模型；读取检查点时不重复下载预训练权重。"""
    model_cfg = config["model"]
    use_pretrained = model_cfg.get("encoder_weights") == "imagenet" if pretrained is None else pretrained
    return build_segmentation_model(model_cfg["architecture"], model_cfg["in_channels"], model_cfg["out_channels"], model_cfg["base_channels"], dropout, use_pretrained)


def load_model(config: dict[str, Any], checkpoint: Path, device: torch.device, dropout: float = 0.0) -> torch.nn.Module:
    model = build_unet(config, dropout=dropout, pretrained=False).to(device)
    load_checkpoint(model, checkpoint, device)
    model.eval()
    return model


def denormalize(image: torch.Tensor, mode: str) -> np.ndarray:
    array = image.detach().cpu().numpy().transpose(1, 2, 0)
    if mode == "imagenet":
        array = array * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    return np.clip(array, 0.0, 1.0)

