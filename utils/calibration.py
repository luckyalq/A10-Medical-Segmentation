from __future__ import annotations

import numpy as np


class StreamingECE:
    def __init__(self, bins: int = 15) -> None:
        self.bins = bins
        self.count = np.zeros(bins, dtype=np.float64)
        self.confidence_sum = np.zeros(bins, dtype=np.float64)
        self.correct_sum = np.zeros(bins, dtype=np.float64)

    def update(self, probability: np.ndarray, target: np.ndarray) -> None:
        p = probability.reshape(-1).astype(np.float64)
        y = target.reshape(-1) >= 0.5
        confidence = np.maximum(p, 1.0 - p)
        correct = (p >= 0.5) == y
        indices = np.minimum(((confidence - 0.5) * 2.0 * self.bins).astype(int), self.bins - 1)
        for index in range(self.bins):
            selected = indices == index
            if selected.any():
                self.count[index] += selected.sum()
                self.confidence_sum[index] += confidence[selected].sum()
                self.correct_sum[index] += correct[selected].sum()

    def compute(self) -> tuple[float, list[dict[str, float]]]:
        total = max(self.count.sum(), 1.0)
        details = []
        ece = 0.0
        for index in range(self.bins):
            if self.count[index] == 0:
                continue
            confidence = self.confidence_sum[index] / self.count[index]
            accuracy = self.correct_sum[index] / self.count[index]
            fraction = self.count[index] / total
            ece += fraction * abs(confidence - accuracy)
            details.append({"bin": index, "confidence": float(confidence), "accuracy": float(accuracy), "fraction": float(fraction)})
        return float(ece), details


def expected_calibration_error(probability: np.ndarray, target: np.ndarray, bins: int = 15) -> tuple[float, list[dict[str, float]]]:
    p = probability.reshape(-1).astype(np.float64)
    y = (target.reshape(-1) >= 0.5).astype(np.float64)
    confidence = np.maximum(p, 1.0 - p)
    correctness = ((p >= 0.5) == (y >= 0.5)).astype(np.float64)
    edges = np.linspace(0.5, 1.0, bins + 1)
    details: list[dict[str, float]] = []
    ece = 0.0
    for index in range(bins):
        left, right = edges[index], edges[index + 1]
        selected = (confidence >= left) & (confidence < right if index < bins - 1 else confidence <= right)
        if not selected.any():
            continue
        weight = selected.mean()
        avg_confidence = confidence[selected].mean()
        accuracy = correctness[selected].mean()
        ece += weight * abs(avg_confidence - accuracy)
        details.append({"bin_left": left, "bin_right": right, "confidence": avg_confidence, "accuracy": accuracy, "fraction": weight})
    return float(ece), details

