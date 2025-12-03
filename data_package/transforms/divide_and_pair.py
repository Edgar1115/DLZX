import os
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import torchvision.transforms.functional as TF



def extract_patches_128(
    img: torch.Tensor,
    patch_size: int = 128,
    padding_mode: str = "reflect"
) -> Tuple[torch.Tensor, List[Tuple[int, int]], Tuple[int, int], Tuple[int, int]]:
    
    # 如果是单通道的 (H, W)，加一个 channel 维度 -> (1, H, W)
    if img.dim() == 2:
        img = img.unsqueeze(0)  # (1, H, W)
    elif img.dim() != 3:
        raise ValueError(f"img must be 2D or 3D tensor, got shape {img.shape}")

    C, H, W = img.shape
    orig_size = (H, W)

    # 计算底部和右侧需要 padding 多少，保证 Hp 和 Wp 能被 patch_size 整除
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size

    # F.pad 的顺序是 (left, right, top, bottom)
    padding = (0, pad_w, 0, pad_h)

    if pad_h > 0 or pad_w > 0:
        img_padded = F.pad(img, padding, mode=padding_mode)
    else:
        img_padded = img

    _, Hp, Wp = img_padded.shape
    padded_size = (Hp, Wp)

    # 使用 unfold 做滑动窗口（这里 stride = patch_size，不重叠）
    # unfold(维度, kernel_size, stride)
    patches = img_padded.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    # 现在形状是 (C, num_patches_h, num_patches_w, patch_size, patch_size)

    num_patches_h = patches.size(1)
    num_patches_w = patches.size(2)

    # 调整维度顺序并展平为 (N, C, patch_size, patch_size)
    patches = (
        patches.permute(1, 2, 0, 3, 4)  # -> (num_patches_h, num_patches_w, C, patch_size, patch_size)
               .contiguous()
               .view(-1, C, patch_size, patch_size)
    )

    # 记录每个 patch 在“填充后图像”中的左上角坐标 (top, left)
    coords: List[Tuple[int, int]] = []
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            top = i * patch_size
            left = j * patch_size
            coords.append((top, left))

    return patches, coords, orig_size, padded_size

def pair_image_and_mask(
    image_dir: str,
    mask_dir: str,
    image_suffixes=("_training.tif",),
    mask_suffix=("_manual1.gif",)
) -> List[Tuple[str, str]]:

    pairs = []

    # 遍历 image 目录
    for fname in os.listdir(image_dir):
        img_name = os.path.basename(fname)
        if not fname.lower().endswith(image_suffixes):
            continue

        
        # img_stem = os.path.splitext(img_name)[0]  # A
        img_stem = img_name.split("_")[0] 
        # print(img_stem)

        # 尝试在 mask_dir 中找到对应的 mask
        found_mask = None
        
        for ms in mask_suffix:
            candidate = img_stem + ms
            
            for md in os.listdir(mask_dir):
                md = os.path.join(mask_dir, md)
                
                md_dir = os.path.dirname(md)
                candidate_path = os.path.join(md_dir, candidate)

            
                if os.path.exists(candidate_path):
                    found_mask = candidate_path
                    break
            if found_mask is None:
                break

        if found_mask is None:
            # 找不到就跳过或 raise，看你习惯
            print(f"[WARN] No mask found for image: {img_name}")
            continue

        img_path = os.path.join(image_dir, img_name)
        pairs.append((img_path, found_mask))

    return pairs



class SegPatchDataset(Dataset):

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        patch_size: int = 128,
        img_padding_mode: str = "reflect",
        mask_padding_mode: str = "constant"
    ):

        super().__init__()

        # 1) 先把整图按文件名对应起来
        self.pairs = pair_image_and_mask(image_dir, mask_dir)
        if len(self.pairs) == 0:
            raise RuntimeError("No image/mask pairs found. Check your paths and naming convention.")

        self.patch_size = patch_size
        self.img_padding_mode = img_padding_mode
        self.mask_padding_mode = mask_padding_mode

        # 最终要存所有的 patch 对
        self.img_patches: List[torch.Tensor] = []
        self.mask_patches: List[torch.Tensor] = []

        # 2) 对每一对 (image, mask) 做分块
        self._prepare_patches()

    def _prepare_patches(self):
        """
        把所有的整图对切成 patch，并存到列表中
        """
        for img_path, mask_path in self.pairs:
            # ---- 读图像 ----
            # 原图假设是 RGB，转为 tensor: (C, H, W), float32, [0,1]
            img = Image.open(img_path).convert("RGB")
            img_tensor = TF.to_tensor(img)

            # ---- 读 mask ----
            # mask 一般是单通道标签图，用 'L' 即可，再转为 long 类型标签
            mask = Image.open(mask_path).convert("L")
            mask_np = torch.from_numpy(
                # 注意：GIF 读进来可能是 uint8，直接转 tensor 即可
                # 你可以根据需要调整标签值（例如把 255 -> 1），这里假设原图就是 0/1/2 等 label
                # 如果用 numpy，需要先 import numpy as np，这里直接用 torch.frombuffer 不太方便
                # 为简单起见，我们先用 TF.to_tensor 再乘 255 做整数标签
                (TF.to_tensor(mask) * 255).squeeze(0).byte().numpy()
            ).long()  # (H, W), long

            # ---- 尺寸检查 ----
            if img_tensor.shape[1:] != mask_np.shape:
                raise ValueError(
                    f"Image and mask size mismatch: {img_path}, {mask_path}, "
                    f"img={img_tensor.shape}, mask={mask_np.shape}"
                )

            # ---- 对 image 分块 ----
            img_patches, _, _, _ = extract_patches_128(
                img_tensor,
                patch_size=self.patch_size,
                padding_mode=self.img_padding_mode
            )

            # ---- 对 mask 分块 ----
            # 注意：对 mask 我们用 constant padding，更合理（pad 区域当背景：0）
            mask_patches, _, _, _ = extract_patches_128(
                mask_np,
                patch_size=self.patch_size,
                padding_mode=self.mask_padding_mode
            )
            # mask_patches: (N, 1, H, W)  或者 (N, 1, 128, 128) 取决于上面的逻辑

            if img_patches.shape[0] != mask_patches.shape[0]:
                raise RuntimeError(
                    f"Patch number not equal for {img_path} and {mask_path}: "
                    f"{img_patches.shape[0]} vs {mask_patches.shape[0]}"
                )

            # ---- 存起来 ----
            # img_patches: (N, 3, 128, 128)
            # mask_patches: (N, 1, 128, 128)  或 (N, 128, 128)，这里统一成 (N, 1, H, W)
            if mask_patches.dim() == 3:
                # (N, H, W) -> (N, 1, H, W)
                mask_patches = mask_patches.unsqueeze(1)

            self.img_patches.extend([p for p in img_patches])
            self.mask_patches.extend([p for p in mask_patches])

        print(f"[INFO] Total patches: {len(self.img_patches)}")

    def __len__(self):
        return len(self.img_patches)

    def __getitem__(self, idx: int):

        img_patch = self.img_patches[idx]
        mask_patch = self.mask_patches[idx]

        # 这里你可以在后面加数据增强（几何 + 颜色）逻辑
        # 当前先返回原始 patch

        return img_patch, mask_patch