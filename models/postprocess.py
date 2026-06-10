"""
Adaptive NMS — scale-dependent IoU thresholds for dense small-object scenes.

Standard NMS uses a fixed IoU threshold (default 0.5), which severely
under-detects densely packed small objects. Adaptive NMS varies the
threshold based on predicted box area:
  - Small objects (< 32x32):  higher threshold (0.7) → keep more overlaps
  - Medium objects (32-64):   moderate threshold (0.6)
  - Large objects (> 64x64):  lower threshold (0.5) → standard suppression

This introduces zero extra parameters, zero training overhead, and
zero GPU memory cost at inference time.
"""

import torch
import torchvision.ops as tv_ops


class AdaptiveNMS:
    """Scale-adaptive Non-Maximum Suppression.

    Instead of a single IoU threshold for all boxes, uses a
    piecewise-constant function of box area.
    """

    # Threshold schedule: (area_min, area_max, iou_threshold)
    SCHEDULE = [
        (0,       32 * 32,  0.70),   # tiny:  be more permissive
        (32 * 32, 64 * 64,  0.60),   # small: moderate
        (64 * 64, float('inf'), 0.50),  # medium+: standard
    ]

    @staticmethod
    def compute_areas(boxes_xyxy: torch.Tensor) -> torch.Tensor:
        """Compute area of boxes in xyxy format."""
        w = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        h = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
        return w * h

    @classmethod
    def adaptive_nms(cls, boxes: torch.Tensor, scores: torch.Tensor,
                     labels: torch.Tensor = None, max_det: int = 300) -> tuple:
        """Apply scale-adaptive NMS.

        Args:
            boxes: (N, 4) boxes in xyxy format
            scores: (N,) confidence scores
            labels: (N,) class labels (optional)
            max_det: maximum detections to keep

        Returns:
            (kept_boxes, kept_scores, kept_indices)
        """
        if boxes.numel() == 0:
            return boxes, scores, torch.zeros(0, dtype=torch.long, device=boxes.device)

        areas = cls.compute_areas(boxes)

        # Assign IoU threshold per box based on area
        iou_thresholds = torch.full_like(scores, 0.5)
        for a_min, a_max, thresh in cls.SCHEDULE:
            mask = (areas >= a_min) & (areas < a_max)
            iou_thresholds[mask] = thresh

        # Sort by score descending
        order = scores.argsort(descending=True)

        keep = []
        suppressed = torch.zeros(boxes.shape[0], dtype=torch.bool, device=boxes.device)

        for idx in order:
            if suppressed[idx]:
                continue
            if len(keep) >= max_det:
                break

            keep.append(idx.item())

            # Compute IoU with remaining boxes
            ious = tv_ops.box_iou(boxes[idx:idx + 1], boxes[~suppressed])[0]

            # Suppress based on each box's own threshold
            remaining_idx = (~suppressed).nonzero(as_tuple=True)[0]
            for j, rem_idx in enumerate(remaining_idx):
                if ious[j] > iou_thresholds[rem_idx]:
                    suppressed[rem_idx] = True

        keep = torch.tensor(keep, device=boxes.device, dtype=torch.long)
        return boxes[keep], scores[keep], keep


def apply_adaptive_nms(predictions: torch.Tensor, conf_thres: float = 0.25,
                       max_det: int = 300) -> torch.Tensor:
    """Apply Adaptive NMS to YOLO-style predictions.

    Args:
        predictions: (N, 6) [x1, y1, x2, y2, conf, cls]
        conf_thres: minimum confidence threshold
        max_det: maximum detections

    Returns:
        (K, 6) filtered predictions
    """
    if predictions.numel() == 0:
        return predictions

    # Filter by confidence
    mask = predictions[:, 4] >= conf_thres
    predictions = predictions[mask]
    if predictions.numel() == 0:
        return predictions

    boxes = predictions[:, :4]
    scores = predictions[:, 4]

    kept_boxes, kept_scores, _ = AdaptiveNMS.adaptive_nms(
        boxes, scores, None, max_det
    )

    # Reconstruct output
    kept_preds = torch.cat([
        kept_boxes,
        kept_scores.unsqueeze(1),
        predictions[_:1, 5:6].repeat(len(kept_boxes), 1) if predictions.shape[1] > 5
        else torch.zeros(len(kept_boxes), 0, device=predictions.device)
    ], dim=1)

    return kept_preds


# Export for Ultralytics integration
def adaptive_nms_postprocess(predictions, conf=0.25, iou=0.5, max_det=300, **kwargs):
    """Drop-in replacement for Ultralytics' NMS postprocessing."""
    return apply_adaptive_nms(predictions, conf_thres=conf, max_det=max_det)
