# model_package/unet.py
from typing import Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    经典 UNet 里的两次 3x3 卷积 + BN + ReLU

    in_channels -> out_channels
    """

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
            nn.Dropout2d(p=0.2),
            activation(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.Dropout2d(p=0.2),
            activation(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class MultiConvBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        branch_channels: int | None = None,
        *,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        activation: type[nn.Module] = nn.ReLU,
    ):
        """
        in_channels:  输入特征图的通道数 C_in
        branch_channels: 每个分支（1x1, 3x3, 5x5）输出的通道数,节省计算量，可以让他更小，最后再
        """
        super().__init__()

        if branch_channels is None:
            branch_channels = out_channels


        # 1x1 分支
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=1, bias=False),
            norm_layer(branch_channels),
            nn.Dropout2d(p=0.2),
            activation(inplace=True),
        )

        # 3x3 分支
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=1, stride=1, bias=False),
            norm_layer(branch_channels),
            nn.Dropout2d(p=0.2),
            activation(inplace=True),
        )

        
        # 5x5 分支
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=5, padding=2, stride=1, bias=False),
            norm_layer(branch_channels),
            nn.Dropout2d(p=0.2),
            activation(inplace=True),
        )

        # 融合卷积，输出通道 = out_channels，方便和输入做残差
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_channels * 3, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            norm_layer(out_channels),
            nn.Dropout2d(p=0.2),
            activation(inplace=True),
        )

        if in_channels == out_channels:
            self.res = nn.Identity()
        else:
            self.res = nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0,stride=1,bias=False)

    def forward(self, x_in):

        x_1 = self.branch1x1(x_in)  # [N, branch_channels, H, W]

        x_3 = self.branch3x3(x_in)  # [N, branch_channels, H, W]

        x_5 = self.branch5x5(x_in)  # [N, branch_channels, H, W]


        x_c = torch.cat([x_1, x_3, x_5], dim=1)  # [N, 3*branch_channels, H, W]


        x_f = self.fuse(x_c)


        x_out = x_f + self.res(x_in)  # [N, out_channels, H, W]

        return x_out


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block

    输入:  [B, C, H, W]
    输出:  [B, C, H, W]，每个通道被自适应重新加权

    参数:
        channels: 输入/输出通道数 C
        reduction: 压缩比 r，默认 16
        activation: 使用 ReLU，保持和你现有模块一致
    """
    def __init__(self, channels: int, reduction: int = 16,
                 activation: type[nn.Module] = nn.ReLU) -> None:
        super().__init__()

        hidden = max(channels // reduction, 4)

        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            activation(inplace=True),
            nn.Linear(hidden, channels, bias=True),
            nn.Sigmoid(),
        )
        # 残差（恒等映射）
        self.res = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()

        # ----- Squeeze -----
        # 全局平均池化: [B, C, H, W] -> [B, C]
        
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)  # [B, C]

        # ----- Excitation -----
        y = self.fc(y)                              # [B, C]
        y = y.view(b, c, 1, 1)                      # [B, C, 1, 1]
        
        return x * y + self.res(x)                          # [B, C, H, W]


class Down(nn.Module):
    """
    下采样模块: MaxPool(2x2) + DoubleConv
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            MultiConvBlock(in_channels,out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    上采样模块:
    - 先上采样 (bilinear / ConvTranspose2d)
    - 与 encoder 的特征拼接
    - 再 MC

    注意: 这里把“decoder 通道数”和“skip 通道数”分开传入，便于以后修改结构
    """

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
        self.conv = MultiConvBlock(conv_in_channels,out_channels)

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
    """
    最后一层 1x1 卷积，用来把通道数映射到类别数
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MC_UNet(nn.Module):
    """
    可配置的经典 U-Net 实现 (2D)

    参数:
        in_channels: 输入图像通道数 (例如灰度图=1, RGB=3)
        num_classes: 输出类别数 (二分类可用 1 或 2，看你用的 loss)
        channels: 各层特征图通道数列表，例如 (64, 128, 256, 512, 1024)
                  长度 = 深度 + 1 (最后一个是 bottleneck)
        bilinear: True 用双线性插值上采样; False 用 ConvTranspose2d
    """

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
        self.inc = MultiConvBlock(in_channels,self.channels[0])

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
        """
        方便直接拿到 softmax / sigmoid 输出:
        - num_classes == 1 时: 使用 sigmoid (二分类 / 二值分割)
        - num_classes  > 1 时: 使用 softmax
        """
        logits = self.forward(x)
        if self.num_classes == 1:
            return torch.sigmoid(logits)
        return F.softmax(logits, dim=1)


def create_model(
    in_channels: int = 1,
    num_classes: int = 1,
    channels: Sequence[int] = (64, 128, 256, 512, 1024),
    bilinear: bool = True,
    device: torch.device | str = "cuda",
) -> nn.Module:
    """
    创建并移动 UNet 到 device 上。
    """
    model = MC_UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        channels=channels,
        bilinear=bilinear,
    )
    device = torch.device(device)
    model.to(device)
    return model