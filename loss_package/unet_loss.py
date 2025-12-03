# loss_package/unet_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    """
    加权交叉熵 / 加权 BCE：
    - 对于 C == 1：二分类，使用 binary_cross_entropy_with_logits
    - 对于 C > 1：多分类，使用 cross_entropy
    - 权重图 weight_map: [B, H, W] 或 [B, 1, H, W]
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,      # [B, C, H, W]
        target: torch.Tensor,      # [B, H, W] 或 [B, 1, H, W]
        weight_map: torch.Tensor | None = None,  # [B, H, W] or [B, 1, H, W]
    ) -> torch.Tensor:
        if logits.dim() != 4:
            raise ValueError(f"logits 形状应为 [B, C, H, W]，但得到 {logits.shape}")

        B, C, H, W = logits.shape

        # 调整 target
        if target.dim() == 4:
            # [B, 1, H, W] -> [B, H, W]
            target = target.squeeze(1)
        if target.shape != (B, H, W):
            raise ValueError(f"target 形状应为 [B, H, W]，但得到 {target.shape}")

        # 调整 weight_map
        if weight_map is not None:
            if weight_map.dim() == 4:
                weight_map = weight_map.squeeze(1)
            if weight_map.shape != (B, H, W):
                raise ValueError(
                    f"weight_map 形状应为 [B, H, W]，但得到 {weight_map.shape}"
                )

        # ----------------- 二分类：C == 1 -----------------
        if C == 1:
            # BCEWithLogitsLoss：需要 target float、形状 [B, 1, H, W]
            target_f = target.float().unsqueeze(1)  # [B, 1, H, W]
            criterion_bce = nn.BCEWithLogitsLoss(reduction="none")
            
            if weight_map is not None:
                w = weight_map.unsqueeze(1)  # [B, 1, H, W]
                per_pixel = criterion_bce(logits, target_f) * w  # 按像素加权
            
                if self.reduction == "sum":
                    return per_pixel.sum()
                elif self.reduction == "mean":
                    return per_pixel.sum() / (w.sum() + 1e-8)
                else:
                    return per_pixel  # [B, 1, H, W]

                    
        else:
            # ----------------- 多分类：C > 1 -----------------
            target_ce = target.long()  # [B, H, W]
    
            per_pixel_loss = F.cross_entropy(
                logits, target_ce, reduction="none"
            )  # [B, H, W]
    
            if weight_map is not None:
                per_pixel_loss = per_pixel_loss * weight_map  # 加权
    
                if self.reduction == "sum":
                    return per_pixel_loss.sum()
                elif self.reduction == "mean":
                    return per_pixel_loss.sum() / (weight_map.sum() + 1e-8)
                else:  # "none"
                    return per_pixel_loss  # [B, H, W]
            else:
                if self.reduction == "none":
                    return per_pixel_loss
                elif self.reduction == "sum":
                    return per_pixel_loss.sum()
                else:
                    return per_pixel_loss.mean()
