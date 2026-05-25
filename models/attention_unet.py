from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import DoubleConv


class AttentionGate(nn.Module):
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.gate_projection = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.skip_projection = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        attention = F.relu(self.gate_projection(gate) + self.skip_projection(skip))
        return skip * self.psi(attention)


class AttentionUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: list[int] | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        features = features or [64, 128, 256, 512]
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        current_channels = in_channels
        for feature in features:
            self.downs.append(DoubleConv(current_channels, feature, dropout=dropout))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            current_channels = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, dropout=dropout)
        reversed_features = list(reversed(features))
        self.up_transpose = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        gate_channels = features[-1] * 2
        for feature in reversed_features:
            self.up_transpose.append(nn.ConvTranspose2d(gate_channels, feature, kernel_size=2, stride=2))
            self.attention.append(AttentionGate(feature, feature, max(feature // 2, 1)))
            self.up_convs.append(DoubleConv(feature * 2, feature, dropout=dropout))
            gate_channels = feature

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for up, gate, conv, skip in zip(self.up_transpose, self.attention, self.up_convs, reversed(skips)):
            x = up(x)
            skip = gate(x, skip)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = conv(torch.cat([skip, x], dim=1))
        return torch.sigmoid(self.final_conv(x))
