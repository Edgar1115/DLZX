import torch
import pathlib
from PIL import Image



# 加载数据集
class MyImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = pathlib.Path(root_dir)
        self.transform = transform

        self.samples = []
        self.class_to_idx = {}

        class_names = []
        for d in self.root_dir.iterdir():
            if d.is_dir():
                class_names.append(d.name)
        for idx, class_name in enumerate(class_names):
            self.class_to_idx[class_name] = idx

        exts = {".png"}
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            for img_path in class_dir.rglob("*"):
                if img_path.suffix.lower() in exts:
                    self.samples.append((img_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label



# 划分数据集
import random

# 划分数据集索引
def split_dataset_indices(samples, train_ratio, val_ratio, test_ratio, seed=42):

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    indices = list(range(len(samples)))
    random.Random(seed).shuffle(indices)

    train = int(train_ratio * len(samples))
    val = int(val_ratio * len(samples))

    train_indices = indices[:train]
    val_indices = indices[train:train+val]
    test_indices = indices[train+val:]

    return train_indices, val_indices, test_indices

# 通过索引得到划分后的数据集
class split_dataset(torch.utils.data.Dataset):
    def __init__(self, datasets, indices):

        self.datasets = datasets
        self.indices = list(indices)


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, label = self.datasets[real_idx]

        return img, label