# data_package/utils/split.py

import random
from pathlib import Path
from typing import Sequence, Tuple, List
import numpy as np
from typing import Dict


def train_val_test_split(
    *arrays: Sequence,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[List, ...]:
    """
    同时对多个数组做 train/val/test 划分（长度必须相同），返回拆分后的列表。

    用法举例：
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
            X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )
    """
    if not arrays:
        raise ValueError("至少传入一个数组")

    n_samples = len(arrays[0])
    for arr in arrays[1:]:
        if len(arr) != n_samples:
            raise ValueError("所有数组长度必须相同")

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test 比例之和必须为 1")

    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    def _split_one(arr):
        arr = np.array(arr)
        return (
            arr[train_idx].tolist(),
            arr[val_idx].tolist(),
            arr[test_idx].tolist(),
        )

    # 对每个数组做同样的拆分，然后把结果拍扁返回
    splits = []
    for arr in arrays:
        splits.extend(_split_one(arr))
    return tuple(splits)


def split_image_mask_dataset(
    image_dir: str,
    mask_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    shuffle: bool = True,
    seed: int = 42,
) -> Dict[str, List[Tuple[str, str]]]:
    """
    从 image_dir 和 mask_dir 中读取文件，按顺序一一对应，然后划分为
    train / val / test 三个子集（test_ratio 可以为 0 表示不划分测试集）。

    返回:
        {
            "train": [(img_path, mask_path), ...],
            "val":   [(img_path, mask_path), ...],
            "test":  [(img_path, mask_path), ...],  # 若 test_ratio == 0，则为空列表
        }
    """

    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    # 支持的图像后缀（你可以按需再加）
    img_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}

    image_paths = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in img_exts]
    )
    mask_paths = sorted(
        [p for p in mask_dir.iterdir() if p.suffix.lower() in img_exts]
    )

    assert len(image_paths) == len(mask_paths), \
        f"图像数量({len(image_paths)})和mask数量({len(mask_paths)})不一致！"

    # 按排序后一一对应配对 —— 对于 DRIVE 这种官方数据集通常是可行的
    pairs = list(zip(image_paths, mask_paths))

    # 转成字符串路径，方便后续使用
    pairs = [(str(img), str(mask)) for img, mask in pairs]

    # 是否打乱
    if shuffle:
        random.seed(seed)
        random.shuffle(pairs)

    n = len(pairs)

    # 规范化比例，使得 train:val:test 比例按输入比值划分
    total_ratio = train_ratio + val_ratio + test_ratio
    train_ratio_n = train_ratio / total_ratio
    val_ratio_n = val_ratio / total_ratio
    test_ratio_n = test_ratio / total_ratio

    n_train = int(n * train_ratio_n)
    n_val = int(n * val_ratio_n)
    # 剩下的全部给 test（包括由于取整造成的 “尾数”）
    n_test = n - n_train - n_val

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:] if test_ratio > 0 else []

    return {
        "train": train_pairs,
        "val": val_pairs,
        "test": test_pairs,
    }