# unet.py
# -*- coding: utf-8 -*-
"""
一个简洁、易扩展的 U-Net 实现（仅包含网络结构）。
你可以在此基础上自由修改通道数、网络深度等结构超参数。
"""

from __future__ import annotations  # 允许使用未来的类型注解特性（Python 3.10+ 更友好）

from typing import List  # 从 typing 导入 List，用于类型标注

import torch  # 导入 PyTorch 主包
from torch import nn  # 从 torch 中导入神经网络模块 nn
import torch.nn.functional as F  # 导入函数式 API，常用于上采样等操作


class DoubleConv(nn.Module):
    """两次 Conv(3x3) + BN + ReLU 的基础卷积块。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        参数:
            in_channels: 输入特征图的通道数
            out_channels: 输出特征图的通道数
        """
        super().__init__()  # 调用父类 nn.Module 的构造函数

        # 使用 nn.Sequential 将多层顺序串联，结构一目了然，便于复用
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,          # 输入通道数
                out_channels,         # 输出通道数
                kernel_size=3,        # 卷积核大小 3x3
                padding=1,            # 填充 1，保证 H/W 尺寸不变
                bias=False            # 去掉偏置，通常与 BN 搭配使用
            ),
            nn.BatchNorm2d(out_channels),  # 对卷积输出做 BatchNorm
            nn.ReLU(inplace=True),         # 非线性激活，inplace 节省显存

            nn.Conv2d(
                out_channels,         # 第二次卷积的输入通道数
                out_channels,         # 输出通道数不变
                kernel_size=3,        # 卷积核大小 3x3
                padding=1,            # 同样保持 H/W 不变
                bias=False            # 仍然不需要偏置
            ),
            nn.BatchNorm2d(out_channels),  # 第二次卷积后的 BatchNorm
            nn.ReLU(inplace=True),         # 第二次 ReLU 激活
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        参数:
            x: 输入特征图，形状通常为 [B, C, H, W]
        返回:
            经过两次卷积块后的特征图，形状为 [B, out_channels, H, W]
        """
        x = self.block(x)  # 将输入依次通过顺序结构 block
        return x           # 返回结果


class DownBlock(nn.Module):
    """下采样块：MaxPool2d + DoubleConv。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        参数:
            in_channels: 输入特征图通道数
            out_channels: 下采样后输出特征图通道数
        """
        super().__init__()  # 调用父类构造函数

        # 2x2 最大池化，将 H/W 缩小一半
        self.pool = nn.MaxPool2d(
            kernel_size=2,  # 池化窗口大小为 2x2
            stride=2        # 步长为 2，使空间尺寸减半
        )

        # 池化后接一个 DoubleConv，提取更深层次的特征
        self.conv = DoubleConv(
            in_channels=in_channels,   # 输入通道
            out_channels=out_channels  # 输出通道
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        步骤: 先下采样 -> 再做两次卷积。
        """
        x = self.pool(x)  # 先通过最大池化，下采样空间尺寸
        x = self.conv(x)  # 再通过 DoubleConv 提取特征
        return x          # 返回结果
