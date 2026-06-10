"""
CGHR-Neck: Context-Guided High-Resolution Neck

A lightweight feature fusion neck for UAV small object detection.
Key components:
  - CAA (Context Anchor Attention): dual-branch spatial-channel fusion gate
  - P5-as-context: retains P5 for semantic guidance only (no detection head)
  - Deformable Conv injection: uses DeformConv2d from torchvision for learned
    spatial offset alignment between semantic and detail features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as tv_ops


class CAA(nn.Module):
    """Context Anchor Attention — lightweight dual-branch attention gate.

    Fuses high-level semantic features (P4/P5 context) with low-level
    detail features (P2/P3) via a learnable spatial-channel gate:
      - Channel branch: SE-like squeeze on concatenated features
      - Spatial branch: single-channel spatial attention map
      - The two branches are mixed via learnable alpha
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        r = max(4, channels // reduction)

        # Channel attention (SE-variant)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.ch_fc1 = nn.Conv2d(channels * 2, r, 1, bias=False)
        self.ch_act = nn.SiLU()
        self.ch_fc2 = nn.Conv2d(r, channels, 1, bias=False)
        self.ch_sig = nn.Sigmoid()

        # Spatial attention
        self.sp_conv = nn.Conv2d(channels * 2, 1, 7, padding=3, bias=False)
        self.sp_sig = nn.Sigmoid()

        # Learnable fusion weight
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x_det: torch.Tensor, x_ctx: torch.Tensor) -> torch.Tensor:
        # Align spatial dimensions
        if x_det.shape[2:] != x_ctx.shape[2:]:
            x_ctx = F.interpolate(x_ctx, size=x_det.shape[2:],
                                   mode='bilinear', align_corners=False)

        fused = torch.cat([x_det, x_ctx], dim=1)

        # Channel gate
        ch = self.ch_sig(self.ch_fc2(self.ch_act(self.ch_fc1(self.pool(fused)))))

        # Spatial gate
        sp = self.sp_sig(self.sp_conv(fused))

        # Dynamic mix
        gate = self.alpha * ch + (1.0 - self.alpha) * sp

        return x_det + gate * x_ctx


class DeformConvInject(nn.Module):
    """Deformable conv injection for semantic-to-detail alignment.

    Uses torchvision.ops.deform_conv2d (Deformable Conv v2) with
    learned spatial offsets to align high-level semantic features
    to the spatial layout of high-resolution detail features.

    Memory-conscious: offset path uses only 64 inner channels.
    """

    def __init__(self, ctx_ch: int, det_ch: int):
        super().__init__()
        self.det_ch = det_ch

        # Offset prediction head: 2 * K*K offsets per pixel (K=3 -> 18 channels)
        self.offset_conv = nn.Conv2d(ctx_ch, 2 * 9, 3, padding=1)

        # Weight for deformable conv (in_features, out_features, kernel, ...)
        self.weight = nn.Parameter(torch.randn(det_ch, ctx_ch, 3, 3))

        self.norm = nn.BatchNorm2d(det_ch)
        self.act = nn.SiLU()

    def forward(self, x_ctx: torch.Tensor, x_det: torch.Tensor) -> torch.Tensor:
        # Resize context to detail resolution
        if x_ctx.shape[2:] != x_det.shape[2:]:
            x_ctx = F.interpolate(x_ctx, size=x_det.shape[2:],
                                   mode='bilinear', align_corners=False)

        # Predict offsets
        offsets = self.offset_conv(x_ctx)

        # Deformable Conv2d
        x_aligned = tv_ops.deform_conv2d(
            x_ctx, offsets, self.weight, None, stride=1, padding=1
        )

        return self.act(self.norm(x_aligned))


class CGHRNeck(nn.Module):
    """CGHR-Neck: Context-Guided High-Resolution Feature Fusion Neck.

    Replaces YOLO's standard PAN-FPN with:
      - P5 context-only layer (no detection head)
      - P2 high-resolution branch for tiny objects
      - Deformable Conv injection from P5 -> P4, P4 -> P3, P4 -> P2
      - CAA fusion gates at each injection point

    Input: [F2, F3, F4, F5] from backbone (shallow to deep)
    Output: [P2, P3, P4] fused features (P5 excluded from detection)
    """

    def __init__(self, channels: list = (128, 256, 512, 1024)):
        super().__init__()
        c2, c3, c4, c5 = channels

        # P5 -> P4
        self.p5_to_p4_deform = DeformConvInject(c5, c4)
        self.p5_to_p4_caa = CAA(c4)

        # P4 -> P3
        self.p4_to_p3_deform = DeformConvInject(c4, c3)
        self.p4_to_p3_caa = CAA(c3)

        # P4 -> P2
        self.p4_to_p2_deform = DeformConvInject(c4, c2)
        self.p4_to_p2_caa = CAA(c2)

    def forward(self, features: list) -> list:
        f2, f3, f4, f5 = features

        # P5 -> P4: deep semantic into mid-level
        p4_ctx = self.p5_to_p4_deform(f5, f4)
        p4 = self.p5_to_p4_caa(f4, p4_ctx)

        # P4 -> P3: mid semantic into small-object level
        p3_ctx = self.p4_to_p3_deform(p4, f3)
        p3 = self.p4_to_p3_caa(f3, p3_ctx)

        # P4 -> P2: mid semantic into tiny-object level
        p2_ctx = self.p4_to_p2_deform(p4, f2)
        p2 = self.p4_to_p2_caa(f2, p2_ctx)

        return [p2, p3, p4]


def build_cghr_neck_config():
    """Return a standard CGHR-Neck config for YOLOv8s-scale models."""
    return {
        "channels": [128, 256, 512, 1024],
    }
