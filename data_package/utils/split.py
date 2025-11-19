# data_package/utils/split.py

from typing import Sequence, Tuple, List
import numpy as np


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
