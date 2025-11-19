# model_package/__init__.py
"""
model_package

集中放各种模型。当前提供:
- UNet: 经典 U-Net 语义分割网络
"""

from .unet import UNet

__all__ = ["UNet"]
