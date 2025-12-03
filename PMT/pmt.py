import torch
from torch import nn

class InitConv(nn.Module):

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        
        super().__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False),
            norm_layer(out_channels),
            activation(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(out_channels),
            activation(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(out_channels),
            activation(inplace=True),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xc = x.clone()
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        return x3