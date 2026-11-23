from __future__ import annotations

import numpy as np


EPS = 1e-7


def binary_entropy(probability: np.ndarray) -> np.ndarray:
    """二分类预测熵：概率越接近0.5，不确定性越高。"""
    p = np.clip(probability.astype(np.float64), EPS, 1.0 - EPS)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)).astype(np.float32)


def uncertainty_maps(samples: np.ndarray) -> dict[str, np.ndarray]:
    """由多次预测计算平均概率、预测熵、方差和互信息。"""
    if samples.ndim != 3:
        raise ValueError(f"samples 应为 [T,H,W]，实际为 {samples.shape}")
    mean_probability = samples.mean(axis=0).astype(np.float32)
    entropy = binary_entropy(mean_probability)
    variance = samples.var(axis=0).astype(np.float32)
    expected_entropy = binary_entropy(samples).mean(axis=0).astype(np.float32)
    mutual_information = np.maximum(entropy - expected_entropy, 0.0).astype(np.float32)
    return {
        "mean_probability": mean_probability,
        "entropy": entropy,
        "variance": variance,
        "mutual_information": mutual_information,
    }


def image_risk_scores(values: np.ndarray, high_quantile: float = 0.90) -> dict[str, float]:
    """把像素级热力图压缩为供Abstain使用的图像级风险分数。"""
    flat = values.reshape(-1).astype(np.float64)
    threshold = float(flat.max()) * high_quantile
    top_count = max(1, int(np.ceil(flat.size * 0.10)))
    return {
        "mean": float(flat.mean()),
        "high_fraction": float((flat >= threshold).mean()),
        "top10_mean": float(np.partition(flat, -top_count)[-top_count:].mean()),
        "max": float(flat.max()),
    }

