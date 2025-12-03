# transforms.py
from typing import Sequence, Tuple
import random

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def _gaussian_smooth_2d(field: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    对 2D 位移场做高斯平滑。
    field: (1, 1, H, W)
    """
    # 根据 sigma 动态生成核大小
    radius = int(3 * sigma)
    kernel_size = 2 * radius + 1

    device = field.device
    x = torch.arange(kernel_size, device=device) - radius
    gauss_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    gauss_1d = gauss_1d / gauss_1d.sum()

    # 先按 H 方向卷积，再按 W 方向卷积，相当于 2D 高斯
    kernel_h = gauss_1d.view(1, 1, kernel_size, 1)  # (out_ch, in_ch, kH, 1)
    kernel_w = gauss_1d.view(1, 1, 1, kernel_size)  # (out_ch, in_ch, 1, kW)

    # padding='same'
    field = F.conv2d(field, kernel_h, padding=(radius, 0))
    field = F.conv2d(field, kernel_w, padding=(0, radius))
    return field


def elastic_deformation(
    image: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 10.0,
    sigma: float = 4.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对 (image, mask) 做弹性形变（Elastic Deformation）。

    image: (C, H, W)
    mask:  (1, H, W)
    alpha: 位移强度（越大形变越明显）
    sigma: 平滑程度（越大越平滑）
    """
    assert image.shape[-2:] == mask.shape[-2:], "image 和 mask 的 H,W 必须一样"
    device = image.device
    _, H, W = image.shape

    # 随机位移场 (dx, dy)，先随机，再高斯平滑，再乘以 alpha 控制幅度
    dx = torch.randn(1, 1, H, W, device=device)
    dy = torch.randn(1, 1, H, W, device=device)

    dx = _gaussian_smooth_2d(dx, sigma) * alpha
    dy = _gaussian_smooth_2d(dy, sigma) * alpha

    # 构造基础网格
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )

    # 加上位移
    xx = xx + dx.squeeze(0).squeeze(0)
    yy = yy + dy.squeeze(0).squeeze(0)

    # 归一化到 [-1, 1]，grid_sample 需要
    grid_x = 2.0 * xx / (W - 1) - 1.0
    grid_y = 2.0 * yy / (H - 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)  # (H, W, 2)
    grid = grid.unsqueeze(0)  # (1, H, W, 2)

    # 对 image 用双线性插值，对 mask 用最近邻插值，保持离散标签
    img_in = image.unsqueeze(0)   # (1, C, H, W)
    mask_in = mask.unsqueeze(0)   # (1, 1, H, W)

    img_def = F.grid_sample(
        img_in,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    mask_def = F.grid_sample(
        mask_in,
        grid,
        mode="nearest",
        padding_mode="border",
        align_corners=True,
    )

    return img_def.squeeze(0), mask_def.squeeze(0)



def get_train_transform(
    mean: Sequence[float],
    std: Sequence[float],
    flip_prob: float = 0.5,
    rotate_prob: float = 0.5,
    elastic_prob: float = 0.3,
    elastic_alpha: float = 10.0,
    elastic_sigma: float = 4.0,
):
    """
    训练时对 (image, mask) 做常见增强 + 归一化。
    image: (C,H,W), mask: (1,H,W)
    """

    def _transform(image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 随机水平翻转
        if random.random() < flip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # 随机垂直翻转
        if random.random() < flip_prob:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # 随机 0/90/180/270 度旋转
        if random.random() < rotate_prob:
            angles = [0, 90, 180, 270]
            angle = random.choice(angles)
            if angle != 0:
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

        # 简单亮度缩放（只对 image）
        if random.random() < 0.3:
            factor = 0.8 + 0.4 * random.random()  # [0.8, 1.2]
            image = torch.clamp(image * factor, 0.0, 1.0)

        # 归一化
        image = TF.normalize(image, mean=mean, std=std)

        return image, mask

    return _transform


def get_val_transform(
    mean: Sequence[float],
    std: Sequence[float],
):
    """
    验证 / 测试时只做归一化。
    """

    def _transform(image: torch.Tensor, mask: torch.Tensor):
        image = TF.normalize(image, mean=mean, std=std)
        return image, mask

    return _transform
