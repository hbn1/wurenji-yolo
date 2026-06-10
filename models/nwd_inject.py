import torch


@torch.no_grad()
def _box_xyxy_to_cxcywh(boxes):
    cx = (boxes[..., 0] + boxes[..., 2]) / 2.0
    cy = (boxes[..., 1] + boxes[..., 3]) / 2.0
    w = (boxes[..., 2] - boxes[..., 0]).clamp(min=1e-6)
    h = (boxes[..., 3] - boxes[..., 1]).clamp(min=1e-6)
    return torch.stack([cx, cy, w, h], dim=-1)


def nwd_distance(pred_boxes, target_boxes, eps=1e-7):
    p = _box_xyxy_to_cxcywh(pred_boxes)
    t = _box_xyxy_to_cxcywh(target_boxes)
    d2 = (p[:, 0] - t[:, 0]) ** 2 + (p[:, 1] - t[:, 1]) ** 2
    w_diff = (p[:, 2] / 2.0 - t[:, 2] / 2.0) ** 2
    h_diff = (p[:, 3] / 2.0 - t[:, 3] / 2.0) ** 2
    W2 = d2 + w_diff + h_diff
    W = torch.sqrt(W2 + eps)
    return torch.exp(-W / 12.8)


_original_forward = None
_nwd_weight = 0.0
_nwd_threshold = 0.0


def _patched_forward(self, pred_dist, pred_bboxes, anchor_points,
                     target_bboxes, target_scores, target_scores_sum, fg_mask):
    from ultralytics.utils.metrics import bbox_iou
    global _nwd_weight, _nwd_threshold

    weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
    iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask],
                    xywh=False, CIoU=True)
    loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

    if self.dfl_loss:
        from ultralytics.utils.loss import bbox2dist
        target_ltrb = bbox2dist(anchor_points, target_bboxes,
                                 self.dfl_loss.reg_max - 1)
        loss_dfl = self.dfl_loss(
            pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
            target_ltrb[fg_mask]) * weight
        loss_dfl = loss_dfl.sum() / target_scores_sum
    else:
        loss_dfl = torch.tensor(0.0).to(pred_dist.device)

    if _nwd_weight > 0 and fg_mask.sum() > 0:
        pb = pred_bboxes[fg_mask]
        tb = target_bboxes[fg_mask]
        tw = tb[:, 2] - tb[:, 0]
        th = tb[:, 3] - tb[:, 1]
        areas = tw * th
        small = areas < _nwd_threshold
        if small.any():
            nwd_sim = nwd_distance(pb[small], tb[small])
            nwd_val = (1.0 - nwd_sim) * weight[small].squeeze(-1)
            loss_iou = loss_iou + _nwd_weight * nwd_val.sum() / target_scores_sum

    return loss_iou, loss_dfl


def inject_nwd_loss(area_threshold=1024.0, nwd_weight=0.25):
    from ultralytics.utils.loss import BboxLoss
    global _nwd_weight, _nwd_threshold, _original_forward
    _nwd_weight = nwd_weight
    _nwd_threshold = area_threshold
    if _original_forward is None:
        _original_forward = BboxLoss.forward
    BboxLoss.forward = _patched_forward
    if nwd_weight > 0:
        print(f"[NWD] Injected: area_threshold={area_threshold:.0f} px^2, weight={nwd_weight:.2f}")


def remove_nwd_loss():
    global _original_forward
    if _original_forward is not None:
        from ultralytics.utils.loss import BboxLoss
        BboxLoss.forward = _original_forward
        _original_forward = None
