```python
from typing import Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):


    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
        *,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(mid_channels),
            activation(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            activation(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):


    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):


    def __init__(
        self,
        decoder_channels: int,
        skip_channels: int,
        out_channels: int,
        *,
        bilinear: bool = True,
    ) -> None:
        super().__init__()

        if bilinear:
            # 只做空间上采样，不改变通道数
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            # 反卷积上采样，同时保持通道数不变（也可以自己改成减半）
            self.up = nn.ConvTranspose2d(
                decoder_channels, decoder_channels, kernel_size=2, stride=2
            )

        # 上采样后和 skip 连接，所以 conv 的输入通道 = decoder + skip
        conv_in_channels = decoder_channels + skip_channels
        self.conv = DoubleConv(conv_in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # x: decoder 分支输入
        # skip: encoder 的特征 (用于 skip-connection)
        x = self.up(x)

        # 处理一下 H, W 对不齐的情况（偶数/奇数尺寸）
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(
            x,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        *,
        channels: Sequence[int] = (64, 128, 256, 512, 1024),
        bilinear: bool = True,
    ) -> None:
        super().__init__()

        if len(channels) < 2:
            raise ValueError("channels 长度至少为 2，例如 (64, 128)。")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.channels = tuple(channels)
        self.bilinear = bilinear

        # ---- Encoder 部分 ----
        # 第一层: 直接从输入图像卷积到 channels[0]
        self.inc = DoubleConv(in_channels, self.channels[0])

        # 后续的 Down 模块 (不含 bottleneck)
        self.down_blocks = nn.ModuleList()
        for in_ch, out_ch in zip(self.channels[:-2], self.channels[1:-1]):
            self.down_blocks.append(Down(in_ch, out_ch))

        # Bottleneck: 最底层
        self.bottom = Down(self.channels[-2], self.channels[-1])

        # ---- Decoder 部分 ----
        # 逐级上采样，使用对应的 skip connection
        self.up_blocks = nn.ModuleList()
        decoder_ch = self.channels[-1]  # 从 bottleneck 输出通道开始
        # 对应的 skip 通道: 反向遍历 encoder 的各层 (不含 bottleneck)
        for skip_ch in reversed(self.channels[:-1]):
            out_ch = skip_ch  # 经典 UNet: 每次输出通道和 skip 一样
            self.up_blocks.append(
                Up(
                    decoder_channels=decoder_ch,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                    bilinear=bilinear,
                )
            )
            decoder_ch = out_ch  # 下一层 decoder 输入通道

        # 输出层
        self.outc = OutConv(self.channels[0], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder: 记录每一层的特征用于 skip connection
        x0 = self.inc(x)
        enc_feats: list[torch.Tensor] = [x0]

        x_enc = x0
        for down in self.down_blocks:
            x_enc = down(x_enc)
            enc_feats.append(x_enc)

        # Bottleneck
        x_bottom = self.bottom(x_enc)

        # Decoder: 反向使用 skip 特征
        x_dec = x_bottom
        for up, skip in zip(self.up_blocks, reversed(enc_feats)):
            x_dec = up(x_dec, skip)

        logits = self.outc(x_dec)
        return logits

    def forward_with_softmax(self, x: torch.Tensor) -> torch.Tensor:

        logits = self.forward(x)
        if self.num_classes == 1:
            return torch.sigmoid(logits)
        return F.softmax(logits, dim=1)

```

上面是我的网络结构，读懂它，等待我的下一步指令。

我要用这个网络结构进行训练，在python=3.10的条件下,用python，pytorch为我实现如下工作：
1. 数据预处理：
    1. 编写函数，读取数据集，并对数据集按照 测试集:验证集 8:2 的比例划分；
    2. 编写函数，将读取的每一张图片做分patch，每张图片按128*128大小分patch，边缘与四角进行镜像填充，注意，图像与掩码图要对应；
    3. 编写函数，对图像做分割任务常见的增强，并归一化；
2. 加载模型；
3. 编写训练函数；
4. 编写验证函数；
5. 编写函数，计算dice，iou，recall，precision，specificity，展示曲线图并保存；
6. 以上的所有函数，最终都用notebook调用