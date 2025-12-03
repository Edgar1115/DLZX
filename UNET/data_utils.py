# data_utils.py
from pathlib import Path
from typing import List, Tuple, Sequence
import random


def load_image_mask_pairs(
    root_dir: str | Path,
    image_subdir: str = "images",
    mask_subdir: str = "masks",
    image_exts: Sequence[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"),
) -> List[Tuple[str, str]]:
    """
    支持不同命名方式的图像和 mask，例如：
    image: 21_training.tif
    mask : 21_manual.gif

    自动按前缀匹配：21_****
    """
    root_dir = Path(root_dir)
    img_dir = root_dir / image_subdir
    mask_dir = root_dir / mask_subdir

    if not img_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"找不到 images 或 masks 目录: {img_dir}, {mask_dir}")

    # 允许更多 mask 格式
    mask_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif")

    image_paths: List[Path] = []
    for ext in image_exts:
        image_paths.extend(img_dir.glob(f"*{ext}"))

    image_paths = sorted(image_paths)
    pairs: List[Tuple[str, str]] = []
    for img_path in image_paths:
        # 提取编号（遇到第一个下划线前的部分，例如 21_training → 21）
        prefix = img_path.stem.split("_")[0]
        # 在 mask_dir 中寻找以相同 prefix 开头的文件
        candidates = []
        
        for ext in mask_exts:
            candidates.extend(mask_dir.glob(f"{prefix}*{ext}"))

        if len(candidates) == 0:
            print(f"[警告] 找不到对应的 mask: 前缀 {prefix}")
            continue

        candidates = sorted(candidates)
        mask_path = candidates[0]

        pairs.append((str(img_path),str(mask_path)))

        

    if not pairs:
        raise RuntimeError("没有成功匹配到任何 (image, mask) 对。请检查数据集路径和命名。")

    return pairs


def train_val_split(
    pairs: List[Tuple[str, str]],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    按 val_ratio 比例划分训练集 / 验证集。
    """
    rng = random.Random(seed)
    pairs = pairs.copy()
    rng.shuffle(pairs)

    n_total = len(pairs)
    n_val = int(n_total * val_ratio)
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    return train_pairs, val_pairs
