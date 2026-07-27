"""UNet++ backbone used by the paper."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.block(inputs)


class UNetPlusPlus(nn.Module):
    """Canonical five-level UNet++ with optional deep supervision."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        base_channels: int = 32,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()
        channels = [base_channels * (2**index) for index in range(5)]
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2)

        self.conv0_0 = ConvBlock(in_channels, channels[0])
        self.conv1_0 = ConvBlock(channels[0], channels[1])
        self.conv2_0 = ConvBlock(channels[1], channels[2])
        self.conv3_0 = ConvBlock(channels[2], channels[3])
        self.conv4_0 = ConvBlock(channels[3], channels[4])

        self.conv0_1 = ConvBlock(channels[0] + channels[1], channels[0])
        self.conv1_1 = ConvBlock(channels[1] + channels[2], channels[1])
        self.conv2_1 = ConvBlock(channels[2] + channels[3], channels[2])
        self.conv3_1 = ConvBlock(channels[3] + channels[4], channels[3])

        self.conv0_2 = ConvBlock(channels[0] * 2 + channels[1], channels[0])
        self.conv1_2 = ConvBlock(channels[1] * 2 + channels[2], channels[1])
        self.conv2_2 = ConvBlock(channels[2] * 2 + channels[3], channels[2])

        self.conv0_3 = ConvBlock(channels[0] * 3 + channels[1], channels[0])
        self.conv1_3 = ConvBlock(channels[1] * 3 + channels[2], channels[1])

        self.conv0_4 = ConvBlock(channels[0] * 4 + channels[1], channels[0])

        heads = 4 if deep_supervision else 1
        self.heads = nn.ModuleList(
            nn.Conv2d(channels[0], num_classes, kernel_size=1) for _ in range(heads)
        )

    @staticmethod
    def _up(inputs: Tensor, reference: Tensor) -> Tensor:
        return F.interpolate(
            inputs,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, inputs: Tensor) -> Tensor | list[Tensor]:
        x0_0 = self.conv0_0(inputs)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self._up(x1_0, x0_0)], dim=1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._up(x2_0, x1_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._up(x1_1, x0_0)], dim=1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self._up(x3_0, x2_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._up(x2_1, x1_0)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self._up(x1_2, x0_0)], dim=1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self._up(x4_0, x3_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self._up(x3_1, x2_0)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self._up(x2_2, x1_0)], dim=1))
        x0_4 = self.conv0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, self._up(x1_3, x0_0)], dim=1)
        )

        if self.deep_supervision:
            return [
                head(features)
                for head, features in zip(self.heads, [x0_1, x0_2, x0_3, x0_4])
            ]
        return self.heads[0](x0_4)
