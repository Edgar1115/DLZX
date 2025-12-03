# metrics.py
from typing import Dict, Tuple

import torch


def confusion_from_probs(
    probs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[int, int, int, int]:
    """
    给定预测概率 probs 和目标 targets (N,1,H,W)，计算 TP, FP, FN, TN。
    假设是二分类分割（前景/背景）。
    """
    preds = (probs >= threshold).int()
    t = (targets >= 0.5).int()

    preds = preds.view(-1)
    t = t.view(-1)

    tp = int(((preds == 1) & (t == 1)).sum().item())
    tn = int(((preds == 0) & (t == 0)).sum().item())
    fp = int(((preds == 1) & (t == 0)).sum().item())
    fn = int(((preds == 0) & (t == 1)).sum().item())
    return tp, fp, fn, tn


def metrics_from_confusion(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    eps: float = 1e-7,
) -> Dict[str, float]:
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    recall = tp / (tp + fn + eps)          # sensitivity
    precision = tp / (tp + fp + eps)
    specificity = tn / (tn + fp + eps)

    return {
        "dice": dice,
        "iou": iou,
        "recall": recall,
        "precision": precision,
        "specificity": specificity,
    }
