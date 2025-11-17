config = {
    "data_root": "../data/DRIVE",   # 数据集根目录

    "split_mode": "ratio",   # "folder" or "ratio"

    # 如果是 folder 模式，就用这两个
    "train_dir": "train",
    "val_dir": "val",

    # 如果是 ratio 模式，就用这个
    "split": {
        "train": 0.8,
        "val": 0.2,
        # 也可以将来加 "test": 0.0
        "seed": 42,   # 保证每次切分一样
    },

    "num_classes": 10,                 # 分类类别数
    "image_size": (256, 256),          # 统一的输入尺寸 (H, W)

    # 归一化参数（这里先用 ImageNet 常用的）
    "mean": [0.485, 0.456, 0.406],
    "std":  [0.229, 0.224, 0.225],

    # 数据增强相关配置
    "augment": {
        "random_flip": True,
        "random_crop": False,
        "color_jitter": False,
    },

    "batch_size": 32,
    "num_workers": 4,
    "shuffle_train": True,
}
