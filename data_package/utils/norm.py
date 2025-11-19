# data_package/utils/norm.py
from typing import Tuple
import torch
from torch.utils.data import DataLoader

def compute_channel_mean_std(
    dataset,
    batch_size: int = 16,
    num_workers: int = 0,
    max_batches: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从一个 Dataset 估计图像通道的 mean 和 std。
    要求 dataset.__getitem__ 返回的 sample 里，image 已经是 [C, H, W] 的 float32 Tensor，
    但还没有做 Normalize。
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers)

    n_batches = 0
    mean = 0.0
    sq_mean = 0.0
    n_pixels = 0

    for batch in loader:
        imgs = batch["image"]  # [B, C, H, W]
        b, c, h, w = imgs.shape
        imgs = imgs.view(b, c, -1)  # [B, C, H*W]

        # 这一批每个通道的均值：先对像素平均，再对 batch 求和
        batch_mean = imgs.mean(dim=2).sum(dim=0)        # [C]
        batch_sq_mean = (imgs ** 2).mean(dim=2).sum(dim=0)

        mean += batch_mean
        sq_mean += batch_sq_mean
        n_pixels += b  # 这里相当于按“图像”为单位平均

        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break

    mean /= n_pixels           # [C]
    sq_mean /= n_pixels
    var = sq_mean - mean ** 2
    std = torch.sqrt(var.clamp(min=1e-6))

    return mean, std
