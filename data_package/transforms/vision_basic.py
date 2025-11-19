# data_package/transforms/vision_basic.py

from typing import Tuple
from PIL import Image
import numpy as np
import torch
import cv2

from .base import BaseTransform, Sample


class Resize(BaseTransform):
    """把 image（以及可能存在的 mask）缩放到指定大小。"""

    def __init__(self, size: Tuple[int, int]):
        """
        size: (H, W)
        """
        self.size = size

    def _resize_pil(self, arr: np.ndarray, mode=Image.BILINEAR) -> np.ndarray:
        pil = Image.fromarray(arr)
        # PIL 的 size 是 (W, H)，所以要倒过来
        pil = pil.resize((self.size[1], self.size[0]), mode)
        return np.array(pil)

    def __call__(self, sample: Sample) -> Sample:
        img = sample["image"]

        if img.ndim == 2:          # 灰度
            img = self._resize_pil(img, Image.BILINEAR)
        elif img.ndim == 3:        # H, W, C
            img = self._resize_pil(img, Image.BILINEAR)
        else:
            raise ValueError(f"不支持的 image 维度: {img.shape}")

        sample["image"] = img

        # 以后做分割时可以一并处理 mask，这里先保留逻辑
        if "mask" in sample and sample["mask"] is not None:
            mask = sample["mask"]
            mask = self._resize_pil(mask, Image.NEAREST)  # mask 用最近邻，不插值
            sample["mask"] = mask

        return sample


class RandomHorizontalFlip(BaseTransform):
    """随机左右翻转，image / mask 一起翻。"""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if np.random.rand() < self.p:
            img = sample["image"]
            sample["image"] = np.flip(img, axis=1).copy()

            if "mask" in sample and sample["mask"] is not None:
                mask = sample["mask"]
                sample["mask"] = np.flip(mask, axis=1).copy()

        return sample

class RandomVerticalFlip(BaseTransform):
    """随机上下翻转，image / mask 一起翻。"""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if np.random.rand() < self.p:
            img = sample["image"]
            sample["image"] = np.flip(img, axis=0).copy()

            if "mask" in sample and sample["mask"] is not None:
                mask = sample["mask"]
                sample["mask"] = np.flip(mask, axis=0).copy()

        return sample



class RandomRotate90(BaseTransform):
    """
    随机旋转 0, 90, 180, 270 度。
    适用于 image 和 mask，一起转。
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if np.random.rand() < self.p:
            k = np.random.randint(0, 4)  # 旋转次数
            if k > 0:
                img = sample["image"]
                # numpy.rot90 默认是 (m, n) 或 (m, n, k) 的最后两个维度旋转
                sample["image"] = np.rot90(img, k, axes=(0, 1)).copy()

                if "mask" in sample and sample["mask"] is not None:
                    mask = sample["mask"]
                    sample["mask"] = np.rot90(mask, k, axes=(0, 1)).copy()

        return sample


class SelectGreenChannel(BaseTransform):
    """
    从 RGB 图像中取出 G 通道，得到单通道灰度图。
    输入：sample["image"] 是 PIL.Image 或 numpy 数组 (H, W, 3)/(H, W)
    输出：sample["image"] 是 numpy 单通道 (H, W)，mask 不变
    """
    def __call__(self, sample: Sample) -> Sample:
        image = sample["image"]

        # 统一转成 numpy 数组
        if isinstance(image, Image.Image):
            arr = np.array(image)
        else:
            arr = np.array(image)

        # RGB -> 只取 G 通道
        if arr.ndim == 3 and arr.shape[2] == 3:
            g = arr[..., 1]      # H x W
        else:
            # 已经是单通道就直接用
            g = arr

        # 后续的 RandomFlip / Resize 都是基于 numpy 的
        sample["image"] = g
        return sample


class ApplyCLAHE(BaseTransform):
    """
    对单通道图像做 CLAHE 自适应直方图均衡，增强对比度。
    默认 clip_limit=2.0, tile_grid_size=(8,8) 是 DRIVE 上常用配置。
    输入：sample["image"] 为单通道或三通道 numpy / PIL
    输出：sample["image"] 为 uint8 单通道 numpy
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, sample: Sample) -> Sample:
        image = sample["image"]

        # 转 numpy
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = np.array(image)

        # 如果还是 3 通道，就先转灰度
        if img.ndim == 3 and img.shape[2] == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img

        # CLAHE 需要 uint8
        if img_gray.dtype != np.uint8:
            img_gray = cv2.normalize(
                img_gray, None, 0, 255, cv2.NORM_MINMAX
            ).astype("uint8")

        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size,
        )
        cl = clahe.apply(img_gray)  # H x W, uint8

        sample["image"] = cl
        return sample
class Normalize(BaseTransform):
    """对 Tensor 形式的 image 做标准化，要求 image 是 [C, H, W]。"""

    def __init__(self, mean, std):
        # mean/std 是长度为 C 的列表，比如 3 通道 RGB
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, sample: Sample) -> Sample:
        x = sample["image"]   # Tensor, [C, H, W]
        sample["image"] = (x - self.mean) / self.std
        return sample
