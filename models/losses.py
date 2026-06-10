"""
NWD Loss + Repulsion Loss for dense small-object detection.

NWD (Normalized Wasserstein Distance):
  Models bounding boxes as 2D Gaussian distributions to compute
  Wasserstein distance. Much more stable than IoU for objects < 32x32px.
  Reference: Wang et al., "A Normalized Gaussian Wasserstein Distance for Tiny Object Detection", AAAI 2023.

Repulsion Loss:
  Push predicted boxes away from other GT boxes they should NOT match.
  Reduces false merges in dense crowds/vehicle clusters.
  Reference: Wang et al., "Repulsion Loss: Detecting Pedestrians in a Crowd", CVPR 2018.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------- NWD Loss ---------------

def gaussian_wasserstein_distance(box1: torch.Tensor, box2: torch.Tensor,
                                   eps: float = 1e-7) -> torch.Tensor:
    """Compute Normalized Wasserstein Distance between box distributions.

    Each box [cx, cy, w, h] is treated as a 2D Gaussian N(μ, Σ),
    where μ = (cx, cy) and Σ = diag(w²/4, h²/4).

    Args:
        box1: (N, 4) [cx, cy, w, h]
        box2: (M, 4) [cx, cy, w, h]

    Returns:
        (N, M) NWD distance matrix
    """
    # Means
    mu1_x, mu1_y = box1[:, 0], box1[:, 1]
    mu2_x, mu2_y = box2[:, 0], box2[:, 1]

    # Wh (treat as standard deviations squared for diagonal covariance)
    # σ² = w²/4, h²/4  =>  std approx. scaled by width/height
    w1, h1 = box1[:, 2].clamp(min=eps), box1[:, 3].clamp(min=eps)
    w2, h2 = box2[:, 2].clamp(min=eps), box2[:, 3].clamp(min=eps)

    # Center distance (N, M)
    c_dist = (mu1_x[:, None] - mu2_x[None, :]) ** 2 + \
             (mu1_y[:, None] - mu2_y[None, :]) ** 2

    # Wasserstein distance for diagonal 2D Gaussians:
    # W² = ||μ1 - μ2||² + ||Σ1^(1/2) - Σ2^(1/2)||_F²
    # For diagonal Σ = diag(w²/4, h²/4): Σ^(1/2) = diag(w/2, h/2)
    w_dist = c_dist + \
             (w1[:, None] / 2 - w2[None, :] / 2) ** 2 + \
             (h1[:, None] / 2 - h2[None, :] / 2) ** 2

    # Normalized Wasserstein Distance: exp(-sqrt(W²) / C)
    # C is a dataset-specific normalization constant
    sqrt_w = torch.sqrt(w_dist + eps)
    c_constant = 12.8  # calibrated for VisDrone (typical small object size)
    nwd = torch.exp(-sqrt_w / c_constant)

    return nwd


class NWDLoss(nn.Module):
    """NWD-based regression loss for objects below a size threshold.

    Uses sigmoid-smoothed transition between NWD and IoU regimes
    to avoid loss discontinuity at the boundary.
    """

    def __init__(self, area_threshold: float = 32.0 * 32.0, tau: float = 50.0):
        """
        Args:
            area_threshold: below this area (in px), NWD is dominant
            tau: temperature for sigmoid transition (higher = sharper switch)
        """
        super().__init__()
        self.threshold = area_threshold
        self.tau = tau

    def forward(self, pred_boxes: torch.Tensor, gt_boxes: torch.Tensor,
                gt_areas: torch.Tensor) -> torch.Tensor:
        """Compute per-box regression loss with NWD.

        Args:
            pred_boxes: (N, 4) predicted boxes [cx, cy, w, h]
            gt_boxes: (N, 4) GT boxes [cx, cy, w, h]
            gt_areas: (N,) area of each GT box

        Returns:
            scalar loss value
        """
        # Soft weight: 1.0 for tiny objects, 0.0 for large
        weight = torch.sigmoid((self.threshold - gt_areas) / self.tau)

        # NWD similarity for all boxes
        nwd_sim = gaussian_wasserstein_distance(pred_boxes, gt_boxes)
        nwd_sim = nwd_sim.diag()  # (N,)

        # NWD loss: 1 - similarity
        nwd_loss = (1.0 - nwd_sim) * weight

        return nwd_loss.mean()


# --------------- Repulsion Loss ---------------

class RepulsionLoss(nn.Module):
    """Repulsion loss for dense crowded scenes.

    RepGT: pushes each predicted box away from other GT boxes
           that are NOT its assigned target.
    RepBox: prevents multiple predictions from collapsing onto the
            same target (optional, disabled for YOLO one-to-many assign).
    """

    def __init__(self, sigma: float = 0.5, alpha: float = 0.6):
        """
        Args:
            sigma: smooth-l1 threshold for RepGT
            alpha: RepBox weight (set 0 to disable)
        """
        super().__init__()
        self.sigma = sigma
        self.alpha = alpha

    def forward(self, pred_boxes: torch.Tensor, gt_boxes: torch.Tensor,
                assign_indices: torch.Tensor) -> torch.Tensor:
        """Compute RepGT loss.

        Args:
            pred_boxes: (N, 4) all predicted boxes [cx, cy, w, h]
            gt_boxes: (M, 4) all GT boxes [cx, cy, w, h]
            assign_indices: (N,) assigned GT index for each prediction (-1 for background)

        Returns:
            scalar repulsion loss
        """
        device = pred_boxes.device
        n_pos = (assign_indices >= 0).sum()

        if n_pos == 0:
            return torch.tensor(0.0, device=device)

        # Only positive predictions
        pos_mask = assign_indices >= 0
        pos_pred = pred_boxes[pos_mask]  # (P, 4)
        pos_gt_idx = assign_indices[pos_mask]  # (P,)

        loss_rep = torch.tensor(0.0, device=device)

        for i in range(pos_pred.shape[0]):
            gt_i = pos_gt_idx[i].item()

            # Other GT boxes (excluding assigned GT)
            other_mask = torch.arange(gt_boxes.shape[0], device=device) != gt_i
            if not other_mask.any():
                continue

            other_gts = gt_boxes[other_mask]  # (K, 4)

            # IoG: Intersection over GT — how much of the predicted box
            # overlaps with OTHER GT boxes
            pred = pos_pred[i]  # (4,)
            iog = _box_iog(pred, other_gts)  # (K,)
            iog_max = iog.max()

            if iog_max > 0:
                loss_rep += _smooth_ln(iog_max, self.sigma)

        return loss_rep / max(n_pos, 1)


def _box_iog(pred_box: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """Compute Intersection-over-Groundtruth (IoG).

    Args:
        pred_box: (4,) single predicted box [cx, cy, w, h]
        gt_boxes: (K, 4) GT boxes [cx, cy, w, h]

    Returns:
        (K,) IoG values
    """
    px, py, pw, ph = pred_box
    gx, gy, gw, gh = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]

    # Convert to xyxy
    p_x1, p_y1 = px - pw / 2, py - ph / 2
    p_x2, p_y2 = px + pw / 2, py + ph / 2
    g_x1, g_y1 = gx - gw / 2, gy - gh / 2
    g_x2, g_y2 = gx + gw / 2, gy + gh / 2

    # Intersection area
    inter_w = (torch.min(p_x2, g_x2) - torch.max(p_x1, g_x1)).clamp(min=0)
    inter_h = (torch.min(p_y2, g_y2) - torch.max(p_y1, g_y1)).clamp(min=0)
    inter = inter_w * inter_h

    # GT area
    gt_area = gw * gh + 1e-7

    return inter / gt_area


def _smooth_ln(x: torch.Tensor, sigma: float = 0.5) -> torch.Tensor:
    """Smooth Ln loss for repulsion."""
    if x <= sigma:
        return -torch.log(1 - x + 1e-7) / sigma
    else:
        return (x - sigma) / (1 - sigma) + 1


# --------------- Combined Loss (for YOLOv8 integration) ---------------

class DroneDetectionLoss(nn.Module):
    """Combined loss: NWD (small objects) + CIoU (large) + Repulsion (dense)."""

    def __init__(self, nwd_weight: float = 0.25, ciou_weight: float = 0.75,
                 rep_weight: float = 0.1, small_threshold: float = 32.0 * 32.0):
        super().__init__()
        self.nwd_loss = NWDLoss(area_threshold=small_threshold)
        self.rep_loss = RepulsionLoss()
        self.nwd_w = nwd_weight
        self.ciou_w = ciou_weight
        self.rep_w = rep_weight

    def forward(self, pred_boxes, gt_boxes, gt_areas, assign_indices,
                ciou_loss_val):
        nwd = self.nwd_loss(pred_boxes, gt_boxes, gt_areas)
        rep = self.rep_loss(pred_boxes, gt_boxes, assign_indices)
        return self.nwd_w * nwd + self.ciou_w * ciou_loss_val + self.rep_w * rep
