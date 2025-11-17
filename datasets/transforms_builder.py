# transforms_builder.py
from torchvision import transforms

def build_classification_transform(cfg, is_train: bool):
    image_size = cfg["image_size"]
    mean = cfg["mean"]
    std = cfg["std"]
    aug_cfg = cfg["augment"]

    t_list = []

    if is_train:
        # ====== 训练集特有：数据增强 ======
        if aug_cfg.get("random_flip", False):
            t_list.append(transforms.RandomHorizontalFlip())

        if aug_cfg.get("random_crop", False):
            # 这里先简单用 Resize + RandomCrop，你以后可以换成 RandomResizedCrop
            t_list.append(transforms.Resize((image_size[0] + 32, image_size[1] + 32)))
            t_list.append(transforms.RandomCrop(image_size))
        else:
            t_list.append(transforms.Resize(image_size))

        # 如果想搞颜色增强，可以像这样加：
        # if aug_cfg.get("color_jitter", False):
        #     t_list.append(transforms.ColorJitter(...))

    else:
        # ====== 验证/测试集：尽量“稳定、可复现” ======
        t_list.append(transforms.Resize(image_size))

    # 训练/验证都要做的步骤
    t_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return transforms.Compose(t_list)
