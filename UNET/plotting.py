# plotting.py
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def plot_history(
    history: Dict[str, List[float]],
    save_dir: str | Path = "./training_plots",
) -> None:
    """
    根据 history 画出 Loss / Dice / IoU / Recall / Precision / Specificity 的曲线，
    每个指标一张图，保存到 save_dir。
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    def _plot_pair(train_key: str, val_key: str, title: str, filename: str) -> None:
        plt.figure()
        plt.plot(epochs, history[train_key], label=train_key)
        plt.plot(epochs, history[val_key], label=val_key)
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.savefig(save_dir / filename)
        plt.close()

    _plot_pair("train_loss", "val_loss", "Loss", "loss_curve.png")
    _plot_pair("train_dice", "val_dice", "Dice", "dice_curve.png")
    _plot_pair("train_iou", "val_iou", "IoU", "iou_curve.png")
    _plot_pair("train_recall", "val_recall", "Recall", "recall_curve.png")
    _plot_pair("train_precision", "val_precision", "Precision", "precision_curve.png")
    _plot_pair("train_specificity", "val_specificity", "Specificity", "specificity_curve.png")

    print(f"曲线图已保存到: {save_dir.resolve()}")
