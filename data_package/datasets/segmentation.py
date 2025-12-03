# data_package/datasets/segmentation.py

from torch.utils.data import Dataset
from typing import List, Optional, Dict, Any, Callable
from PIL import Image
import numpy as np

from data_package.utils import UNetWeightMapGenerator

Sample = Dict[str, Any]


def _load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def _load_mask(path: str) -> np.ndarray:
    # 分割 mask 一般是灰度图，每个像素是类别 id
    mask = Image.open(path)
    return np.array(mask)


class ImageSegmentationDataset(Dataset):
    """
    通用图像分割 Dataset：一张图 + 一张 mask。
    返回的 sample 结构：
        {
            "image": Tensor [C, H, W],
            "mask":  Tensor [H, W] 或 [1, H, W],
            "weight_map": Tensor [H, W] 或 None,
            "meta": {...}
        }
    """

    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        transform: Optional[Callable[[Sample], Sample]] = None,
        weight_gen: UNetWeightMapGenerator | None = None,
        num_classes: int = 1,
    ):
        assert len(image_paths) == len(mask_paths), "图像和 mask 数量不一致"

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

        # ✅ 别忘了保存
        self.weight_gen = weight_gen
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Sample:
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = _load_image(img_path)  # [H, W, C] (np.ndarray)
        mask = _load_mask(mask_path)   # [H, W] (np.ndarray)

        sample: Sample = {
            "image": image,
            "mask": mask,
            "meta": {
                "image_path": img_path,
                "mask_path": mask_path,
                "index": idx,
            },
        }

        # 1) 先做变换（通常会包含 ToTensor + Normalize）
        if self.transform is not None:
            out = self.transform(sample)
            image = out["image"]  # 期望: Tensor [C, H, W]
            mask = out["mask"]    # 期望: Tensor [H, W] 或 [1, H, W]

        # ✅ 写回 sample，确保 DataLoader 看到的是变换后的张量
        sample["image"] = image
        sample["mask"] = mask

        # 2) 生成 weight_map（如果提供了 weight_gen）
        if self.weight_gen is not None:
            mask_for_wm = mask

            # 兼容 [1, H, W]
            if mask_for_wm.dim() == 3 and mask_for_wm.size(0) == 1:
                mask_for_wm = mask_for_wm.squeeze(0)  # [H, W]

            # UNetWeightMapGenerator 期望 [B, H, W]
            mask_for_wm_b = mask_for_wm.unsqueeze(0).long()  # [1, H, W]

            weight_map = self.weight_gen(
                target=mask_for_wm_b,
                num_classes=self.num_classes,
            )  # [1, H, W]

            weight_map = weight_map.squeeze(0)  # [H, W]
            sample["weight_map"] = weight_map   # ⭐ 关键：拼进 sample

        return sample
