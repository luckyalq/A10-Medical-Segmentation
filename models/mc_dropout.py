from __future__ import annotations

import torch
from torch import nn


def enable_dropout_only(model: nn.Module) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


class MCDropoutPredictor:
    def __init__(self, model: nn.Module, passes: int = 20) -> None:
        if passes < 2:
            raise ValueError("MC-Dropout 至少需要 2 次随机前向传播")
        self.model = model
        self.passes = passes

    @torch.inference_mode()
    def predict_samples(self, x: torch.Tensor) -> torch.Tensor:
        enable_dropout_only(self.model)
        return torch.stack([torch.sigmoid(self.model(x)) for _ in range(self.passes)], dim=0)

