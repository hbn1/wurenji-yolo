import torch
import re
import ultralytics.nn.modules as modules
import ultralytics.nn.tasks as tasks


def register_custom_modules():
    from models.cbam import CBAM
    setattr(modules, 'CBAM', CBAM)
    setattr(tasks, 'CBAM', CBAM)
    print("[Course] CBAM registered in modules + tasks")


def load_pretrained_for_cbam(cbam_model, pretrained_path="yolov8s.pt", verbose=True):
    cbam_sd = cbam_model.state_dict()
    ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    try:
        ref_sd = ckpt["model"].float().state_dict()
    except Exception:
        ref_sd = ckpt["model"].state_dict()

    # Build ordered key lists, skip CBAM sub-keys and BN tracking buffers
    cbam_keys = [k for k in cbam_sd.keys()
                 if ".cam." not in k and ".sam." not in k
                 and "num_batches_tracked" not in k]
    ref_keys  = [k for k in ref_sd.keys()
                 if "num_batches_tracked" not in k]

    ri = 0
    new_sd = {}
    for ck in cbam_keys:
        if ri >= len(ref_keys):
            break
        rk = ref_keys[ri]
        ri += 1
        if cbam_sd[ck].shape == ref_sd[rk].shape:
            new_sd[ck] = ref_sd[rk]
        else:
            # Shape mismatch means alignment is off; try to find matching ref key
            found = False
            for j in range(max(0, ri - 3), min(len(ref_keys), ri + 5)):
                if cbam_sd[ck].shape == ref_sd[ref_keys[j]].shape:
                    new_sd[ck] = ref_sd[ref_keys[j]]
                    ri = j + 1
                    found = True
                    break
            if not found:
                ri -= 1  # retry at same ri

    missing, unexpected = cbam_model.load_state_dict(new_sd, strict=False)
    n_cbam = sum(1 for k in cbam_sd if ".cam." in k or ".sam." in k)
    if verbose:
        print(f"[CBAM] Transferred {len(new_sd)} params ({n_cbam} CBAM random-init)")
    return len(new_sd)
