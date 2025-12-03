# engine.py
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader

from metrics import confusion_from_probs, metrics_from_confusion


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device | str = "cuda",
) -> Dict[str, float]:
    """
    训练一个 epoch，返回各项指标的平均值。
    """
    device = torch.device(device)
    model.train()

    running_loss = 0.0
    total_samples = 0

    total_tp = total_fp = total_fn = total_tn = 0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        logits = model(images)
        # 二分类：BCEWithLogitsLoss
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        probs = torch.sigmoid(logits)
        tp, fp, fn, tn = confusion_from_probs(probs, masks)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

    avg_loss = running_loss / max(total_samples, 1)
    metrics = metrics_from_confusion(total_tp, total_fp, total_fn, total_tn)
    metrics["loss"] = avg_loss
    return metrics


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device | str = "cuda",
) -> Dict[str, float]:
    """
    在验证集上跑一个 epoch，返回各项指标。
    """
    device = torch.device(device)
    model.eval()

    running_loss = 0.0
    total_samples = 0

    total_tp = total_fp = total_fn = total_tn = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = loss_fn(logits, masks)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            probs = torch.sigmoid(logits)
            tp, fp, fn, tn = confusion_from_probs(probs, masks)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

    avg_loss = running_loss / max(total_samples, 1)
    metrics = metrics_from_confusion(total_tp, total_fp, total_fn, total_tn)
    metrics["loss"] = avg_loss
    return metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    num_epochs: int,
    device: torch.device | str = "cuda",
    patience: int = 50,
) -> Dict[str, List[float]]:
    """
    训练若干 epoch，返回一个 history 字典：
        {
          'train_loss': [...],
          'val_loss': [...],
          'train_dice': [...],
          'val_dice': [...],
          ...
        }
    """
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_dice": [],
        "val_dice": [],
        "train_iou": [],
        "val_iou": [],
        "train_recall": [],
        "val_recall": [],
        "train_precision": [],
        "val_precision": [],
        "train_specificity": [],
        "val_specificity": [],
    }

    best_dice = 0
    best_model_path = "./outputs/seunet/best_model.pth"

    # ------- Early Stopping -------
    best_epoch = 0
    no_improve = 0
    
    for epoch in range(1, num_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = validate_one_epoch(model, val_loader, loss_fn, device)

        print(
            f"Epoch [{epoch}/{num_epochs}] "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Dice: {val_metrics['dice']:.4f}, "
            f"Val IoU: {val_metrics['iou']:.4f}"
        )

        current_dice = val_metrics["dice"]
        if current_dice > best_dice:
            best_dice = current_dice
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model at epoch {epoch} (Dice={best_dice:.4f})")
        else:
            no_improve += 1

        # ---------- Early Stopping 条件 ----------
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}, best epoch was {best_epoch}.")
            break
        # ---------------------------------------

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])

        history["train_dice"].append(train_metrics["dice"])
        history["val_dice"].append(val_metrics["dice"])

        history["train_iou"].append(train_metrics["iou"])
        history["val_iou"].append(val_metrics["iou"])

        history["train_recall"].append(train_metrics["recall"])
        history["val_recall"].append(val_metrics["recall"])

        history["train_precision"].append(train_metrics["precision"])
        history["val_precision"].append(val_metrics["precision"])

        history["train_specificity"].append(train_metrics["specificity"])
        history["val_specificity"].append(val_metrics["specificity"])

    return history
