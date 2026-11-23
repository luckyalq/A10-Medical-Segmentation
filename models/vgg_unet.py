"""基于预训练VGG16-BN编码器的U-Net。

参考项目的关键做法不是单纯把U-Net加深，而是用ImageNet预训练分类网络
作为编码器，再通过U-Net解码器和跳跃连接恢复像素级预测。本文件使用现代
PyTorch实现同一思路，并在高层解码器加入Dropout，供MC-Dropout采样使用。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import VGG16_BN_Weights, vgg16_bn


class ConvBlock(nn.Module):
    """两层卷积、BatchNorm与ReLU组成的特征提取块。"""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """上采样后与同尺度编码器特征拼接，再进行卷积融合。"""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.fuse = ConvBlock(out_channels + skip_channels, out_channels, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([skip, x], dim=1))


class VGG16UNet(nn.Module):
    """VGG16-BN编码器 + 五级U-Net解码器。"""

    def __init__(self, out_channels: int = 1, pretrained: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        weights = VGG16_BN_Weights.DEFAULT if pretrained else None
        features = vgg16_bn(weights=weights).features
        self.enc1, self.pool1 = features[:6], features[6]
        self.enc2, self.pool2 = features[7:13], features[13]
        self.enc3, self.pool3 = features[14:23], features[23]
        self.enc4, self.pool4 = features[24:33], features[33]
        self.enc5, self.pool5 = features[34:43], features[43]
        self.bottleneck = ConvBlock(512, 512, dropout)
        self.dec5 = DecoderBlock(512, 512, 512, dropout)
        self.dec4 = DecoderBlock(512, 512, 256, dropout)
        self.dec3 = DecoderBlock(256, 256, 128, dropout)
        self.dec2 = DecoderBlock(128, 128, 64)
        self.dec1 = DecoderBlock(64, 64, 32)
        self.output = nn.Conv2d(32, out_channels, 1)

        new_modules = [self.bottleneck, self.dec5, self.dec4, self.dec3, self.dec2, self.dec1, self.output]
        for module in new_modules:
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        e5 = self.enc5(self.pool4(e4))
        b = self.bottleneck(self.pool5(e5))
        return self.output(self.dec1(self.dec2(self.dec3(self.dec4(self.dec5(b, e5), e4), e3), e2), e1))

    def set_encoder_trainable(self, trainable: bool) -> None:
        """冻结编码器时也固定BatchNorm统计，解冻后恢复训练状态。"""
        for module in (self.enc1, self.enc2, self.enc3, self.enc4, self.enc5):
            module.train(trainable)
            for parameter in module.parameters():
                parameter.requires_grad = trainable


def build_segmentation_model(architecture: str, in_channels: int, out_channels: int, base_channels: int, dropout: float, pretrained: bool) -> nn.Module:
    """统一模型工厂，保证训练、预测和不确定性实验使用同一结构。"""
    if architecture == "vgg16_unet":
        if in_channels != 3:
            raise ValueError("VGG16预训练编码器只支持3通道RGB输入")
        return VGG16UNet(out_channels, pretrained, dropout)
    if architecture == "classic_unet":
        from models.unet import UNet
        return UNet(in_channels, out_channels, base_channels, dropout)
    raise ValueError(f"不支持的模型结构: {architecture}")

