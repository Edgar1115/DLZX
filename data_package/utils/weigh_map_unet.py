# utils/weight_map_unet.py

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, label as nd_label, binary_erosion

# 8 邻域结构
STRUCT_8 = np.ones((3, 3), dtype=bool)


class UNetWeightMapGenerator:
    """
    根据 U-Net 论文中的公式生成权重图：

        w(x) = w_c(x) + w_0 * exp(- (d1(x) + d2(x))^2 / (2 * sigma^2))

    - w_c(x): 类别权重（按整图频率或传入的 class_weights）
    - d1, d2: 到最近 & 次近前景实例边界的距离
    """

    def __init__(
        self,
        class_weights=None,
        w0: float = 10.0,
        sigma: float = 5.0,
        background_label: int = 0,
        struct: np.ndarray = STRUCT_8,
    ):
        """
        Args:
            class_weights: None / list / np.array / torch.Tensor，
                如果为 None，则按每张图的类别频率动态计算。
                如果提供，则长度最好等于 num_classes。
            w0, sigma: U-Net 论文中的 w0, sigma 参数
            background_label: 背景类别 id（默认 0）
            struct: 连通性结构元素，用于连通域和腐蚀
        """
        if class_weights is None:
            self.class_weights = None
        else:
            if isinstance(class_weights, torch.Tensor):
                cw = class_weights.detach().cpu().numpy().astype(np.float32)
            else:
                cw = np.asarray(class_weights, dtype=np.float32)
            self.class_weights = cw

        self.w0 = float(w0)
        self.sigma = float(sigma)
        self.background_label = int(background_label)
        self.struct = struct

    def __call__(
        self,
        target: torch.Tensor,
        num_classes: int,
        device=None,
    ) -> torch.Tensor:
        """
        Args:
            target: [B, H, W]，整型 mask（如 int64）
            num_classes: 类别数（通常 num_classes = logits.size(1)）
            device: 返回 tensor 的 device（默认跟 target 一样）

        Returns:
            weight_map: [B, H, W]，float32
        """
        if device is None:
            device = target.device

        # 放到 CPU 做 numpy 计算
        target_np = target.detach().cpu().numpy()  # [B, H, W]
        B, H, W = target_np.shape

        weight_map_tensor = torch.zeros(
            (B, H, W), dtype=torch.float32, device=device
        )

        cw_array = self.class_weights

        for b in range(B):
            t = target_np[b]  # [H, W], int
            h, w = t.shape

            # -------- 1) 类别权重 w_c(x) --------
            if cw_array is None:
                # 按当前图像的类别频率动态计算
                labels, counts = np.unique(t, return_counts=True)
                total = t.size
                class_weights_map = {
                    int(lbl): 1.0 - (cnt / total)
                    for lbl, cnt in zip(labels, counts)
                }
            else:
                # 使用预设 class_weights
                C = num_classes
                class_weights_map = {
                    c: float(cw_array[c]) if c < len(cw_array) else 1.0
                    for c in range(C)
                }

            wc = np.zeros_like(t, dtype=np.float32)
            for lbl, w_c in class_weights_map.items():
                wc[t == lbl] = w_c

            # -------- 2) 前景连通域 + 距离 --------
            # 前景像素：非背景
            fg_mask = (t != self.background_label)

            # 如果整张图都是背景，直接返回 w_c(x)
            if not np.any(fg_mask):
                weight_map_tensor[b] = torch.from_numpy(wc).to(device)
                continue

            # 连通域标记，得到“实例”
            labeled, num_obj = nd_label(fg_mask, structure=self.struct)

            # 如果只有一个实例，就没有“相互之间”的边界距离，论文里这种情况通常只用 w_c
            if num_obj < 2:
                weight_map_tensor[b] = torch.from_numpy(wc).to(device)
                continue

            # d1、d2 初始化为 +inf
            d1 = np.full((h, w), np.inf, dtype=np.float32)
            d2 = np.full((h, w), np.inf, dtype=np.float32)

            # 对每个实例，计算到该实例边界的距离
            for obj_id in range(1, num_obj + 1):
                obj_mask = (labeled == obj_id)

                # 形态学腐蚀后，边界 = obj_mask XOR eroded
                eroded = binary_erosion(
                    obj_mask, structure=self.struct, border_value=0
                )
                boundary = obj_mask ^ eroded
                if not np.any(boundary):
                    continue

                # 以边界为“0 距离”源，做欧式距离变换
                mask_dist = np.ones((h, w), dtype=bool)
                ys, xs = np.where(boundary)
                mask_dist[ys, xs] = False
                dist_map = distance_transform_edt(mask_dist)

                # 更新 d1, d2
                closer = dist_map < d1
                d2 = np.where(closer, d1, np.minimum(d2, dist_map))
                d1 = np.where(closer, dist_map, d1)

            # -------- 3) 计算边界权重项 w_b(x) --------
            dist_sum = d1 + d2
            wb = self.w0 * np.exp(-(dist_sum ** 2) / (2.0 * (self.sigma ** 2)))
            wb = wb.astype(np.float32)

            w_final = wc + wb
            weight_map_tensor[b] = torch.from_numpy(w_final).to(device)

        return weight_map_tensor
