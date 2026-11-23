from __future__ import annotations

import torch
from torch import nn


class EnsemblePredictor(nn.Module):
    def __init__(self, members: list[nn.Module]) -> None:
        super().__init__()
        if not members:
            raise ValueError("Ensemble 至少需要一个成员")
        self.members = nn.ModuleList(members)

    @torch.inference_mode()
    def predict_samples(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.stack([torch.sigmoid(model(x)) for model in self.members], dim=0)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict_samples(x).mean(dim=0)

