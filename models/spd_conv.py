import torch
import torch.nn as nn


class SPDConv(nn.Module):
    """Space-to-Depth Convolution (SPD-Conv) — lossless downsampling.

    Replaces stride-2 conv that discards 75% of pixels with a
    space-to-depth reshuffle followed by a 1x1 conv, preserving
    all fine-grained information critical for small object detection.

    Reference: Sunkara et al., "No More Strided Convolutions or Pooling", NeurIPS 2022
    """

    def __init__(self, c1: int, c2: int, kernel_size: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1 * 4, c2, kernel_size, 1, kernel_size // 2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        # (N, C, H, W) -> (N, C, H/2, 2, W/2, 2) -> (N, 4C, H/2, W/2)
        x = x.view(n, c, h // 2, 2, w // 2, 2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(n, c * 4, h // 2, w // 2)
        return self.conv(x)
