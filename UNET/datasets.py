# datasets.py
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


class SegmentationDataset(Dataset):
    """
    读取整张图像和 mask，后面由 SegPatchDataset 再划分 patch。
    默认只做 ToTensor (0~1)，不做增强和归一化。
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        num_image_channels: int = 1,  # 1=灰度，3=RGB
    ) -> None:
        self.pairs = pairs
        self.num_image_channels = num_image_channels

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path)
        if self.num_image_channels == 1:
            img = img.convert("L")
        else:
            img = img.convert("RGB")
        img_tensor = TF.to_tensor(img)  # [0,1] float32, CxHxW
        return img_tensor

    def _load_mask(self, path: str) -> torch.Tensor:
        """
        假定单通道二值 mask，非零为前景，0 为背景 -> {0,1} float。
        若多类别，可在这里改成 long 类型类别 ID。
        """
        mask = Image.open(path).convert("L")
        mask_np = np.array(mask, dtype=np.uint8)
        mask_tensor = torch.from_numpy(mask_np)
        mask_tensor = (mask_tensor > 0).float().unsqueeze(0)  # 1xHxW, {0,1}
        return mask_tensor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.pairs[idx]
        image = self._load_image(img_path)
        mask = self._load_mask(mask_path)
        return image, mask


def extract_patches(
    img: torch.Tensor,
    patch_size: int = 128,
    padding_mode: str = "reflect",
) -> torch.Tensor:
    """
    将单张图 (C, H, W) 分成若干个 (patch_size, patch_size) 的 patch。
    为了不丢边缘，对 H/W 做镜像 padding。
    返回形状: (N_patches, C, patch_size, patch_size)
    """
    if img.dim() == 2:
        img = img.unsqueeze(0)  # (1, H, W)
    assert img.dim() == 3, "img 必须是 (C,H,W) 或 (H,W)"

    c, h, w = img.shape
    # 计算需要填充多少才能被 patch_size 整除
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    img_padded = F.pad(
        img.unsqueeze(0),  # (1, C, H, W)
        (pad_left, pad_right, pad_top, pad_bottom),
        mode=padding_mode,
    ).squeeze(0)

    _, H, W = img_padded.shape

    patches = img_padded.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    # (C, n_h, n_w, patch_size, patch_size)
    patches = patches.contiguous().view(c, -1, patch_size, patch_size)  # (C, N, p, p)
    patches = patches.permute(1, 0, 2, 3)  # (N, C, p, p)

    return patches


class SegPatchDataset(Dataset):
    """
    给定一个整图 Dataset，离线将所有图像和 mask 分成 patch 存成列表。
    在 __getitem__ 中再做增强和归一化（可选）。
    """

    def __init__(
        self,
        full_image_dataset: Dataset,
        patch_size: int = 128,
        transform = None,  # (img, mask) -> (img, mask)
    ) -> None:
        self.patch_size = patch_size
        self.transform = transform

        self.image_patches: List[torch.Tensor] = []
        self.mask_patches: List[torch.Tensor] = []

        for i in range(len(full_image_dataset)):
            image, mask = full_image_dataset[i]  # CxHxW, 1xHxW
            img_p = extract_patches(image, patch_size=patch_size)
            mask_p = extract_patches(mask, patch_size=patch_size)

            assert img_p.shape[0] == mask_p.shape[0], "图像与 mask patch 数目不一致"

            self.image_patches.extend(list(img_p))   # 每个元素 (C,p,p)
            self.mask_patches.extend(list(mask_p))   # 每个元素 (1,p,p)

    def __len__(self) -> int:
        return len(self.image_patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.image_patches[idx]
        mask = self.mask_patches[idx]

        if self.transform is not None:
            img, mask = self.transform(img, mask)

        return img, mask
