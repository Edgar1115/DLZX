# data_package/datasets/classification.py

from torch.utils.data import Dataset
from typing import List, Optional, Dict, Any, Callable
from PIL import Image
import numpy as np
import os

Sample = Dict[str, Any]


def _load_image(path: str) -> np.ndarray:
    """读取1张 RGB 图片，返回 numpy 数组 (H, W, C)."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


class ImageClassificationDataset(Dataset):
    """
    一个最简单的图像分类 Dataset。

    参数：
    - image_paths: 每张图的路径列表
    - labels: 对应的标签列表（int）
    - transform: 预处理函数（可选），输入/输出都是 sample 字典
    """
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[Callable[[Sample], Sample]] = None,
    ):
        assert len(image_paths) == len(labels), "图片数量和标签数量不一致"
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Sample:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = _load_image(img_path)

        sample: Sample = {
            "image": image,     # numpy 数组，HWC
            "label": label,     # int
            "meta": {           # 一些辅助信息，方便 debug
                "path": img_path,
                "index": idx,
            },
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
