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


class ElasticDeformation(BaseTransform):
    """
    弹性形变 (Elastic Deformation) 数据增强

    思路：
      - 随机生成位移场 dx, dy（形状 [H, W]），在 [-1, 1] 之间
      - 用高斯模糊把位移场平滑，再乘以 alpha 得到真实位移
      - 构造采样网格，使用 cv2.remap 对 image / mask 做同样的弹性形变

    输入：
      sample["image"] : [H, W, C] 或 [H, W] 的 numpy 数组（通常是 uint8）
      sample["mask"]  : [H, W] 或 [H, W, 1] 的 numpy 数组（标签），如果存在
      sample["weight_map"] : [H, W] 的 numpy 数组，如果存在

    参数：
      alpha : 形变强度，越大形变越明显
      sigma : 高斯平滑标准差，越大形变越平滑
      p     : 应用该增强的概率
      border_mode : 边界填充模式，默认使用镜像反射
      random_state: 固定随机种子（可选）
    """
    def __init__(
        self,
        alpha: float = 40.0,
        sigma: float = 6.0,
        p: float = 0.5,
        border_mode: int = cv2.BORDER_REFLECT_101,
        random_state: int | None = None,
    ) -> None:
        self.alpha = float(alpha)
        self.sigma = float(sigma)
        self.p = float(p)
        self.border_mode = border_mode

        # 使用单独的随机数发生器，方便复现实验
        self._rng = np.random.RandomState(random_state) if random_state is not None \
            else np.random.RandomState()

    # 生成位移场：平滑的 dx, dy
    def _generate_displacement(self, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
        # 均匀随机噪声
        dx = self._rng.uniform(-1.0, 1.0, size=(h, w)).astype(np.float32)
        dy = self._rng.uniform(-1.0, 1.0, size=(h, w)).astype(np.float32)

        # 高斯模糊 + 放大
        dx = cv2.GaussianBlur(dx, ksize=(0, 0), sigmaX=self.sigma) * self.alpha
        dy = cv2.GaussianBlur(dy, ksize=(0, 0), sigmaX=self.sigma) * self.alpha

        return dx, dy

    def __call__(self, sample: Sample) -> Sample:
        # --------- 1. 取出 image，并转成 numpy ----------
        img = sample["image"]

        if not isinstance(img, np.ndarray):
            img = np.asarray(img)

        if img.ndim == 2:
            h, w = img.shape
        elif img.ndim == 3:
            h, w = img.shape[:2]
        else:
            raise ValueError(
                f"ElasticDeformation 期望 image 形状为 [H, W] 或 [H, W, C]，但得到 {img.shape}"
            )

        # 随机决定是否应用该增强
        if self._rng.rand() >= self.p:
            # 不做变换，只把 image 确保是 np.ndarray 放回去
            sample["image"] = img
            return sample

        # --------- 2. 生成位移场 + 采样网格 ----------
        dx, dy = self._generate_displacement(h, w)

        # 基础网格：每个像素的 (x, y) 坐标
        # 注意：np.meshgrid 的顺序是 (x, y) -> (W, H)
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

        map_x = (grid_x + dx).astype(np.float32)
        map_y = (grid_y + dy).astype(np.float32)

        # --------- 3. 封装一个 remap 函数 ----------
        def _remap(arr: np.ndarray, interpolation: int) -> np.ndarray:
            arr_np = np.asarray(arr)
            orig_dtype = arr_np.dtype
            warped = cv2.remap(
                arr_np,
                map_x,
                map_y,
                interpolation=interpolation,
                borderMode=self.border_mode,
            )
            return warped.astype(orig_dtype)

        # --------- 4. 形变 image（双线性插值） ----------
        img_deformed = _remap(img, interpolation=cv2.INTER_LINEAR)
        sample["image"] = img_deformed

        # --------- 5. 形变 mask（最近邻插值，避免标签混合） ----------
        if "mask" in sample and sample["mask"] is not None:
            mask = sample["mask"]
            if not isinstance(mask, np.ndarray):
                mask = np.asarray(mask)

            # [H, W, 1] -> [H, W]
            squeeze_last = False
            if mask.ndim == 3 and mask.shape[2] == 1:
                squeeze_last = True
                mask = mask[..., 0]

            mask_deformed = _remap(mask, interpolation=cv2.INTER_NEAREST)

            if squeeze_last:
                mask_deformed = mask_deformed[..., None]

            sample["mask"] = mask_deformed

        # --------- 6. 形变 weight_map（如果有的话，同样用最近邻） ----------
        if "weight_map" in sample and sample["weight_map"] is not None:
            wmap = sample["weight_map"]
            if not isinstance(wmap, np.ndarray):
                wmap = np.asarray(wmap)

            wmap_deformed = _remap(wmap, interpolation=cv2.INTER_NEAREST)
            sample["weight_map"] = wmap_deformed

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


class YCbCrEnhance(BaseTransform):
    """
    在 YCbCr 颜色空间中对 numpy 图像做增强，再转换回 RGB。

    使用场景：
        - 放在 ToTensor 之前使用；
        - 处理的是 numpy 格式的 RGB 图像。

    输入：
        sample["image"]: np.ndarray，形状 [H, W, 3]，RGB
            - 若 dtype 为 uint8，则认为数值范围为 [0, 255]
            - 若 dtype 为 float（float32/float64），则认为范围为 [0, 1]

    输出：
        sample["image"]: np.ndarray，形状 [H, W, 3]，RGB
            - 保持原始 dtype 不变（uint8 或 float）
    """

    def __init__(self, y_gamma: float = 1.0, cb_gain: float = 1.0, cr_gain: float = 1.0):
        # 亮度通道的 gamma 系数（<1 变亮，>1 变暗）
        self.y_gamma = float(y_gamma)
        # 两路色差信号的缩放系数
        self.cb_gain = float(cb_gain)
        self.cr_gain = float(cr_gain)

    def __call__(self, sample: Sample) -> Sample:
        img = sample["image"]

        # 统一为 ndarray
        if not isinstance(img, np.ndarray):
            img = np.asarray(img)

        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(
                f"YCbCrEnhanceNumpy 期望 image 形状为 [H, W, 3]，但得到 {img.shape}"
            )

        orig_dtype = img.dtype

        # 转到 float32，并归一化到 [0,1]
        if np.issubdtype(orig_dtype, np.integer):
            x = img.astype(np.float32) / 255.0
        else:
            x = img.astype(np.float32)
            x = np.clip(x, 0.0, 1.0)

        # 拆分 RGB 通道
        r = x[..., 0]
        g = x[..., 1]
        b = x[..., 2]

        # 1. RGB -> YCbCr（BT.601）
        y  = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
        # cb = 0.5 - 0.168736 * r - 0.331264 * g + 0.5 * b
        # cr = 0.5 + 0.5 * r - 0.418688 * g - 0.081312 * b
        cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b

        # 3. 在 Y、Cb、Cr 空间分别做增强
        # 亮度 Y: gamma 校正
        if self.y_gamma != 1.0:
            y = np.clip(y, 1e-6, 1.0) ** self.y_gamma

        # 色差信号: 以 0.5 为中心线性放大/压缩
        # if self.cb_gain != 1.0:
        #     cb = (cb - 0.5) * self.cb_gain + 0.5
        # if self.cr_gain != 1.0:
        #     cr = (cr - 0.5) * self.cr_gain + 0.5

        cb = np.clip(cb, 0.0, 1.0)
        cr = np.clip(cr, 0.0, 1.0)

        # 5. YCbCr -> RGB
        cb_shift = cb - 0.5
        cr_shift = cr - 0.5

        r2 = y + 1.402 * cr_shift
        g2 = y - 0.344136 * cb_shift - 0.714136 * cr_shift
        b2 = y + 1.772 * cb_shift

        rgb = np.stack([r2, g2, b2], axis=-1)
        rgb = np.clip(rgb, 0.0, 1.0)

        # 转回原始 dtype
        if np.issubdtype(orig_dtype, np.integer):
            rgb = (rgb * 255.0 + 0.5).astype(orig_dtype)
        else:
            rgb = rgb.astype(orig_dtype)

        sample["image"] = rgb
        return sample




class MultiBranchYCbCrPreprocess(BaseTransform):
    """
    多分支 + 融合的 YCbCr 预处理模块（numpy 版，放在 ToTensor 之前用）。

    流程：
    1) RGB -> YCbCr
    2) 对 Y 通道做三路分支：
        - 分支1：原始 Y
        - 分支2：CLAHE 增强
        - 分支3：Unsharp mask（Y + alpha * (Y - GaussianBlur(Y))）
    3) 三路按权重加权融合，得到 Y_fused
    4) 用 Y_fused + 原始 Cb / Cr 还原为 RGB

    输入：
        sample["image"]: np.ndarray, [H, W, 3], RGB
            - uint8: [0, 255]
            - float: [0, 1]

    输出：
        sample["image"]: np.ndarray, [H, W, 3], RGB
            - 保留原始 dtype
    """

    def __init__(
        self,
        # CLAHE 参数
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: tuple[int, int] = (8, 8),
        # unsharp 参数
        unsharp_alpha: float = 1.0,
        unsharp_sigma: float = 1.0,
        # 三个分支的融合权重（会自动归一化）
        w1: float = 0.4,   # 原始 Y
        w2: float = 0.3,   # CLAHE(Y)
        w3: float = 0.3,   # Unsharp(Y)
    ):
        self.clahe_clip_limit = float(clahe_clip_limit)
        self.clahe_tile_grid_size = tuple(clahe_tile_grid_size)
        self.unsharp_alpha = float(unsharp_alpha)
        self.unsharp_sigma = float(unsharp_sigma)

        w = np.array([w1, w2, w3], dtype=np.float32)
        w = w / (w.sum() + 1e-6)
        self.w1, self.w2, self.w3 = w.tolist()

    # ------------ 颜色空间转换（numpy 版） -----------------
    @staticmethod
    def _rgb_to_ycbcr(x: np.ndarray):
        # x: [H, W, 3], float32, [0,1]
        r = x[..., 0]
        g = x[..., 1]
        b = x[..., 2]

        y  = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 0.5 - 0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 0.5 + 0.5 * r - 0.418688 * g - 0.081312 * b
        return y, cb, cr

    @staticmethod
    def _ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray):
        cb_shift = cb - 0.5
        cr_shift = cr - 0.5

        r = y + 1.402 * cr_shift
        g = y - 0.344136 * cb_shift - 0.714136 * cr_shift
        b = y + 1.772 * cb_shift

        rgb = np.stack([r, g, b], axis=-1)
        return np.clip(rgb, 0.0, 1.0)

    # ------------ 主流程 -----------------
    def __call__(self, sample: Sample) -> Sample:
        img = sample["image"]

        if not isinstance(img, np.ndarray):
            img = np.asarray(img)

        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(
                f"MultiBranchYCbCrPreprocessNumpy 期望 image 形状为 [H, W, 3]，但得到 {img.shape}"
            )

        orig_dtype = img.dtype

        # 转 float32, 归一化到 [0,1]
        if np.issubdtype(orig_dtype, np.integer):
            x = img.astype(np.float32) / 255.0
        else:
            x = img.astype(np.float32)
            x = np.clip(x, 0.0, 1.0)

        # 1. RGB -> YCbCr
        y, cb, cr = self._rgb_to_ycbcr(x)

        # --- 分支1：原始 Y ---
        y1 = y

        # --- 分支2：CLAHE(Y) ---
        y_8u = (np.clip(y, 0.0, 1.0) * 255.0).astype("uint8")
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid_size,
        )
        y2 = clahe.apply(y_8u).astype(np.float32) / 255.0

        # --- 分支3：Unsharp mask ---
        # 先做高斯平滑，再加回细节
        y_blur = cv2.GaussianBlur(
            y.astype(np.float32),
            ksize=(0, 0),  # 根据 sigma 自动算核大小
            sigmaX=self.unsharp_sigma,
        )
        y3 = y + self.unsharp_alpha * (y - y_blur)
        y3 = np.clip(y3, 0.0, 1.0)

        # 3. 三路融合
        y_fused = self.w1 * y1 + self.w2 * y2 + self.w3 * y3
        y_fused = np.clip(y_fused, 0.0, 1.0)

        # 4. 用融合后的 Y + 原始 Cb/Cr 重建 RGB
        rgb = self._ycbcr_to_rgb(y_fused, cb, cr)

        # 5. 转回原始 dtype
        if np.issubdtype(orig_dtype, np.integer):
            rgb = (rgb * 255.0 + 0.5).astype(orig_dtype)
        else:
            rgb = rgb.astype(orig_dtype)

        sample["image"] = rgb
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
