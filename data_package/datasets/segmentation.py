# data_package/datasets/segmentation.py

from torch.utils.data import Dataset
from typing import List, Optional, Dict, Any, Callable
from PIL import Image
import numpy as np

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
    """
    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        transform: Optional[Callable[[Sample], Sample]] = None,
    ):
        assert len(image_paths) == len(mask_paths), "图像和 mask 数量不一致"
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Sample:
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = _load_image(img_path)   # H, W, C
        mask = _load_mask(mask_path)    # H, W

        sample: Sample = {
            "image": image,
            "mask": mask,
            "meta": {
                "image_path": img_path,
                "mask_path": mask_path,
                "index": idx,
            },
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
