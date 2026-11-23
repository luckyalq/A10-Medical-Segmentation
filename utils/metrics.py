from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


EPS = 1e-7


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    dims = tuple(range(1, targets.ndim))
    intersection = (probabilities * targets).sum(dim=dims)
    denominator = probabilities.sum(dim=dims) + targets.sum(dim=dims)
    return 1.0 - ((2.0 * intersection + EPS) / (denominator + EPS)).mean()


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """BCE监督逐像素分类，Dice缓解病变前景/背景面积不平衡。"""
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    return bce + soft_dice_loss(logits, targets)


@dataclass
class Confusion:
    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0
    tn: float = 0.0

    def update(self, probabilities: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> None:
        predictions = probabilities >= threshold
        truth = targets >= 0.5
        self.tp += torch.logical_and(predictions, truth).sum().item()
        self.fp += torch.logical_and(predictions, ~truth).sum().item()
        self.fn += torch.logical_and(~predictions, truth).sum().item()
        self.tn += torch.logical_and(~predictions, ~truth).sum().item()

    def compute(self) -> dict[str, float]:
        return metrics_from_counts(self.tp, self.fp, self.fn, self.tn)


def metrics_from_counts(tp: float, fp: float, fn: float, tn: float) -> dict[str, float]:
    """由全数据集混淆计数计算Micro指标，FNR是本课题核心漏检指标。"""
    return {
        "dice": (2 * tp + EPS) / (2 * tp + fp + fn + EPS),
        "iou": (tp + EPS) / (tp + fp + fn + EPS),
        "precision": (tp + EPS) / (tp + fp + EPS),
        "recall": (tp + EPS) / (tp + fn + EPS),
        "fnr": fn / (tp + fn + EPS),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def numpy_counts(probability: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    prediction = probability >= threshold
    truth = target >= 0.5
    tp = np.logical_and(prediction, truth).sum()
    fp = np.logical_and(prediction, ~truth).sum()
    fn = np.logical_and(~prediction, truth).sum()
    tn = np.logical_and(~prediction, ~truth).sum()
    return metrics_from_counts(float(tp), float(fp), float(fn), float(tn))

