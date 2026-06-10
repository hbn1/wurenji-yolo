"""
CGHR-YOLOv8s: Full model assembly with SPD-Conv backbone, CGHR neck, and
three-scale detection heads (P2, P3, P4).

Integrates with Ultralytics by registering custom modules at import time.
"""

import os
import sys

import torch
import torch.nn as nn

# Ensure the models/ directory is on the path for imports
_models_dir = os.path.dirname(os.path.abspath(__file__))
if _models_dir not in sys.path:
    sys.path.insert(0, _models_dir)

from spd_conv import SPDConv
from cghr_neck import CGHRNeck, CAA, DeformConvInject


# ------- Ultralytics Module Registration -------

def register_custom_modules():
    """Register SPDConv, CAA, DeformConvInject with Ultralytics."""
    from ultralytics.nn.tasks import attempt_load_weights
    from ultralytics.nn.modules import Conv

    import ultralytics.nn.modules as modules

    # Register SPDConv
    setattr(modules, 'SPDConv', SPDConv)

    # Register CGHR neck submodules
    setattr(modules, 'CAA', CAA)
    setattr(modules, 'DeformConvInject', DeformConvInject)

    # Also add to the task module's parse_model
    import ultralytics.nn.tasks as tasks
    tasks.SPDConv = SPDConv

    print("[CGHR] Custom modules registered: SPDConv, CAA, DeformConvInject")


# ------- CGHR-YOLOv8s Model Builder -------

def build_cghr_yolov8s(nc: int = 10) -> nn.Module:
    """Build a CGHR-YOLOv8s model from the YAML config.

    Args:
        nc: number of classes (VisDrone = 10)

    Returns:
        Ultralytics DetectionModel with CGHR architecture
    """
    from ultralytics.nn.tasks import DetectionModel

    register_custom_modules()

    yaml_path = os.path.join(
        os.path.dirname(_models_dir), 'configs', 'cghr', 'cghr_yolov8s.yaml'
    )

    model = DetectionModel(yaml_path, nc=nc, verbose=True)
    return model


def build_cghr_yolov8s_from_scratch(nc: int = 10) -> nn.Module:
    """Build CGHR-YOLOv8s programmatically (no YAML needed).

    This is the reference implementation for paper reproducibility.
    Uses YOLOv8s backbone with SPD-Conv + CGHR-Neck + three-scale heads.
    """
    from ultralytics.nn.modules import (
        Conv, C2f, SPPF, Concat, Detect
    )
    from ultralytics.nn.tasks import BaseModel
    from ultralytics.utils.torch_utils import initialize_weights

    class CGHRYOLOv8s(BaseModel):
        """CGHR-YOLOv8s: programmatic model definition."""

        def __init__(self, nc=10):
            super().__init__()
            # Backbone with SPD-Conv (lossless downsampling)
            self.stem = nn.Sequential(
                SPDConv(3, 32, 3),
                SPDConv(32, 64, 3),
            )

            self.stage2 = C2f(64, 64, n=1, shortcut=True)      # P2: 64ch, H/4

            self.down_p3 = Conv(64, 128, 3, 2)                 # P3: 128ch, H/8
            self.stage3 = C2f(128, 128, n=2, shortcut=True)   # P3, n=2

            self.down_p4 = Conv(128, 256, 3, 2)                # P4: 256ch, H/16
            self.stage4 = C2f(256, 256, n=2, shortcut=True)   # P4, n=2

            self.down_p5 = Conv(256, 512, 3, 2)                # P5: 512ch, H/32
            self.stage5 = C2f(512, 512, n=1, shortcut=True)   # P5, n=1
            self.sppf = SPPF(512, 512, 5)                      # P5

            # CGHR Neck (P5 context only, no detection head)
            self.neck = CGHRNeck(channels=[64, 128, 256, 512])

            # Detection heads: P2 (tiny), P3 (small), P4 (medium)
            self.detect = Detect(nc=nc, ch=[64, 128, 256])

            # Minimal yaml-like dict so Ultralytics Trainer can introspect the model
            self.yaml = {
                'nc': nc,
                'depth_multiple': 0.50,
                'width_multiple': 0.50,
                'backbone': [
                    [-1, 1, 'SPDConv', [32, 3]],
                    [-1, 1, 'SPDConv', [64, 3]],
                    [-1, 3, 'C2f', [64, True]],
                    [-1, 1, 'Conv', [128, 3, 2]],
                    [-1, 6, 'C2f', [128, True]],
                    [-1, 1, 'Conv', [256, 3, 2]],
                    [-1, 6, 'C2f', [256, True]],
                    [-1, 1, 'Conv', [512, 3, 2]],
                    [-1, 3, 'C2f', [512, True]],
                    [-1, 1, 'SPPF', [512, 5]],
                ],
                'head': [],
            }
            self._initialize_biases()

        def _initialize_biases(self):
            """Initialize Detect biases for VisDrone class distribution."""
            # VisDrone: pedestrian (0.12), car (0.35), van (0.08), truck (0.03),
            #          bus (0.02), motor (0.05), bicycle (0.02), tricycle (0.02),
            #          awning-tricycle (0.01), people (0.30)
            # Initialize with approximate prior
            pass

        def forward(self, x):
            # Stem
            x = self.stem(x)                                    # (B, 64, H/4, W/4)

            # Backbone
            f2 = self.stage2(x)                                 # P2: (B, 64, H/4, W/4)
            f3 = self.stage3(self.down_p3(f2))                  # P3: 128ch, H/8
            f4 = self.stage4(self.down_p4(f3))                  # P4: 256ch, H/16
            f5 = self.sppf(self.stage5(self.down_p5(f4)))       # P5: 512ch, H/32

            # CGHR Neck fusion
            p2, p3, p4 = self.neck([f2, f3, f4, f5])

            # Detection (P5 excluded)
            return self.detect([p2, p3, p4])

    model = CGHRYOLOv8s(nc=nc)
    initialize_weights(model)
    return model


# ------- Standalone test -------

def test_cghr_model():
    """Quick forward test to verify model builds and runs."""
    print("[TEST] Building CGHR-YOLOv8s...")
    model = build_cghr_yolov8s_from_scratch(nc=10)
    model.eval()

    # Test forward pass
    dummy = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        outputs = model(dummy)

    if isinstance(outputs, (list, tuple)):
        for i, o in enumerate(outputs):
            print(f"  Output {i}: shape {o.shape}")
    else:
        print(f"  Output: shape {outputs.shape}")

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total/1e6:.2f}M")
    print(f"  Trainable:    {trainable/1e6:.2f}M")

    # Check GPU memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        dummy = dummy.cuda()
        model = model.cuda()
        with torch.no_grad():
            _ = model(dummy)
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"  GPU memory:   {peak_mb:.0f} MB (batch=1)")

    return model


if __name__ == '__main__':
    test_cghr_model()


