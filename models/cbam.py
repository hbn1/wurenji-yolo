import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        r = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(r, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg, max_out], dim=1)))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, *args, reduction=16, kernel_size=7):
        super().__init__()
        self.reduction = reduction
        self.kernel_size = kernel_size
        self.cam = None
        self.sam = None

    def forward(self, x):
        if self.cam is None:
            ch = x.shape[1]
            self.cam = ChannelAttention(ch, self.reduction)
            self.sam = SpatialAttention(self.kernel_size)
        x = self.cam(x)
        x = self.sam(x)
        return x
