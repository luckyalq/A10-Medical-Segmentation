from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_segmentation_panel(image: np.ndarray, target: np.ndarray, probability: np.ndarray, path: str | Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    axes[0].imshow(np.clip(image, 0, 1))
    axes[0].set_title("Image")
    axes[1].imshow(target, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")
    axes[2].imshow(probability >= 0.5, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Prediction")
    for axis in axes:
        axis.axis("off")
    _save(fig, path)


def save_uncertainty_panel(image: np.ndarray, target: np.ndarray, probability: np.ndarray, maps: dict[str, np.ndarray], path: str | Path) -> None:
    fig, axes = plt.subplots(1, 6, figsize=(20, 4))
    items = [
        (image, "Image", None),
        (target, "Ground Truth", "gray"),
        (probability >= 0.5, "Prediction", "gray"),
        (maps["entropy"], "Entropy", "magma"),
        (maps["variance"], "Variance", "magma"),
        (maps["mutual_information"], "Mutual Information", "magma"),
    ]
    for axis, (value, title, cmap) in zip(axes, items):
        axis.imshow(np.clip(value, 0, 1) if title == "Image" else value, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
    _save(fig, path)


def save_curve(x: np.ndarray, ys: dict[str, np.ndarray], xlabel: str, ylabel: str, path: str | Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 5))
    for label, y in ys.items():
        axis.plot(x, y, label=label, linewidth=2)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.3)
    axis.legend()
    _save(fig, path)


def save_reliability_diagram(details: list[dict[str, float]], path: str | Path) -> None:
    fig, axis = plt.subplots(figsize=(6, 6))
    confidence = [row["confidence"] for row in details]
    accuracy = [row["accuracy"] for row in details]
    axis.plot([0.5, 1.0], [0.5, 1.0], "--", color="gray", label="Perfect calibration")
    axis.plot(confidence, accuracy, marker="o", linewidth=2, label="Model")
    axis.set_xlim(0.5, 1.0)
    axis.set_ylim(0.5, 1.0)
    axis.set_xlabel("Confidence")
    axis.set_ylabel("Accuracy")
    axis.grid(alpha=0.3)
    axis.legend()
    _save(fig, path)

