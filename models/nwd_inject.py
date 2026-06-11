import torch


def _box_xyxy_to_cxcywh(boxes):
    cx = (boxes[..., 0] + boxes[..., 2]) / 2.0
    cy = (boxes[..., 1] + boxes[..., 3]) / 2.0
    w = (boxes[..., 2] - boxes[..., 0]).clamp(min=1e-6)
    h = (boxes[..., 3] - boxes[..., 1]).clamp(min=1e-6)
    return torch.stack([cx, cy, w, h], dim=-1)


def nwd_distance(pred_boxes, target_boxes, constant=12.8, eps=1e-7):
    p = _box_xyxy_to_cxcywh(pred_boxes)
    t = _box_xyxy_to_cxcywh(target_boxes)
    d2 = (p[:, 0] - t[:, 0]) ** 2 + (p[:, 1] - t[:, 1]) ** 2
    w_diff = (p[:, 2] / 2.0 - t[:, 2] / 2.0) ** 2
    h_diff = (p[:, 3] / 2.0 - t[:, 3] / 2.0) ** 2
    W2 = d2 + w_diff + h_diff
    W = torch.sqrt(W2 + eps)
    return torch.exp(-W / constant)


def _foreground_stride(stride, fg_mask, target_bboxes):
    if stride is None:
        return None

    if stride.ndim == 1:
        stride = stride.view(1, -1, 1)
    elif stride.ndim == 2:
        stride = stride.view(1, stride.shape[0], stride.shape[1])

    if stride.shape[0] == 1 and target_bboxes.shape[0] > 1:
        stride = stride.expand(target_bboxes.shape[0], -1, -1)

    return stride[fg_mask]


_original_forward = None
_nwd_weight = 0.0
_nwd_threshold = 0.0
_nwd_constant = 12.8


def _patched_forward(self, pred_dist, pred_bboxes, anchor_points,
                     target_bboxes, target_scores, target_scores_sum, fg_mask,
                     imgsz=None, stride=None):
    global _nwd_weight, _nwd_threshold, _nwd_constant, _original_forward

    if imgsz is None and stride is None:
        loss_iou, loss_dfl = _original_forward(
            self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
            target_scores, target_scores_sum, fg_mask)
    else:
        loss_iou, loss_dfl = _original_forward(
            self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
            target_scores, target_scores_sum, fg_mask, imgsz, stride)

    if _nwd_weight > 0 and fg_mask.sum() > 0:
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        pb = pred_bboxes[fg_mask]
        tb = target_bboxes[fg_mask]

        stride_fg = _foreground_stride(stride, fg_mask, target_bboxes)
        if stride_fg is not None:
            pb = pb * stride_fg
            tb = tb * stride_fg

        tw = (tb[:, 2] - tb[:, 0]).clamp(min=1e-6)
        th = (tb[:, 3] - tb[:, 1]).clamp(min=1e-6)
        areas = tw * th
        small = areas < _nwd_threshold
        if small.any():
            nwd_sim = nwd_distance(pb[small], tb[small], constant=_nwd_constant)
            nwd_val = (1.0 - nwd_sim) * weight[small].squeeze(-1)
            loss_iou = loss_iou + _nwd_weight * nwd_val.sum() / target_scores_sum

    return loss_iou, loss_dfl


def inject_nwd_loss(area_threshold=1024.0, nwd_weight=0.25, constant=12.8):
    from ultralytics.utils.loss import BboxLoss
    global _nwd_weight, _nwd_threshold, _nwd_constant, _original_forward
    _nwd_weight = nwd_weight
    _nwd_threshold = area_threshold
    _nwd_constant = constant
    if _original_forward is None:
        _original_forward = BboxLoss.forward
    BboxLoss.forward = _patched_forward
    if nwd_weight > 0:
        print(
            f"[NWD] Injected: area_threshold={area_threshold:.0f} px^2, "
            f"weight={nwd_weight:.2f}, constant={constant:.1f}"
        )


def remove_nwd_loss():
    global _original_forward
    if _original_forward is not None:
        from ultralytics.utils.loss import BboxLoss
        BboxLoss.forward = _original_forward
        _original_forward = None

