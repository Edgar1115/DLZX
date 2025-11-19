# data_package/transforms/base.py

import torch
import numpy as np
from typing import Dict, Any, List, Callable

Sample = Dict[str, Any]


class BaseTransform:
    """所有 Transform 的基类（以后可以不继承也行，但这样更清晰）"""
    def __call__(self, sample: Sample) -> Sample:
        raise NotImplementedError


class Compose(BaseTransform):
    """
    把多个 Transform 串起来依次执行：
    transforms = [t1, t2, t3]
    Compose(transforms)(sample) 相当于 t3(t2(t1(sample)))
    """
    def __init__(self, transforms: List[Callable[[Sample], Sample]]):
        self.transforms = transforms

    def __call__(self, sample: Sample) -> Sample:
        for t in self.transforms:
            sample = t(sample)
        return sample


class Identity(BaseTransform):
    """一个啥也不干的 Transform，用来测试流程是否正常。"""
    def __call__(self, sample: Sample) -> Sample:
        return sample

class ToTensor(BaseTransform):
    """
    把 numpy 图像转成 PyTorch Tensor，并把维度从 (H, W, C) 变成 (C, H, W)。
    同时把 label 也变成 tensor。
    """

    def __init__(self, mask_mode: str = "binary"):
        """
        mask_mode:
            - "none": 不处理 mask（分类任务用）
            - "binary": 把 >0 的地方变成 1（适合二类分割）
            - "multiclass": 直接把原始像素值当作类别 id（0 ~ K-1）
        """
        assert mask_mode in ("none", "binary", "multiclass")
        self.mask_mode = mask_mode

    def __call__(self, sample: Sample) -> Sample:
        img = sample["image"]  # numpy, 形状 (H, W, C) 或 (H, W)

        # 1. 先处理维度
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                # 灰度图: H, W -> 1, H, W
                img = img[None, :, :]
            elif img.ndim == 3:
                # 彩色图: H, W, C -> C, H, W
                img = img.transpose(2, 0, 1)
            else:
                raise ValueError(f"不支持的 image 维度: {img.shape}")

            # 2. numpy -> tensor，并转成 float，再缩放到 0~1
            img = torch.from_numpy(img).float() / 255.0

        sample["image"] = img

        # 把 label 也变成 tensor（如果存在）
        if "label" in sample and not isinstance(sample["label"], torch.Tensor):
            sample["label"] = torch.tensor(sample["label"]).long()

        # ---- 处理 mask（分割任务用）----
        if "mask" in sample and sample["mask"] is not None:
            mask = sample["mask"]   # numpy, H x W, 值是 0 或 255

            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)

            # 把 0/255 变成 0/1，然后转成 long，适合 CrossEntropyLoss

            if self.mask_mode == "binary":     # 二类分割
                # 0/255 或各种非 0 都归为 1
                mask = (mask > 0).long()

            elif self.mask_mode == "multiclass":     # 多类分割
                # 假设 mask 像素已经是 0 ~ K-1 的类别 id
                mask = mask.long()
            
            elif self.mask_mode == "none":
                # 不动它
                pass

            sample["mask"] = mask

        return sample
