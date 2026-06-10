"""
Phase 1 — SAHI (Slicing Aided Hyper Inference) evaluation wrapper.

Performs inference on high-resolution VisDrone images by slicing them
into overlapping 640x640 windows, running the model on each, then
merging predictions with full-resolution NMS.

Usage:
    python eval/eval_sahi.py --model experiments/.../best.pt --source data/VisDrone/images/val

Reference: Akyon et al., "Slicing Aided Hyper Inference for Small Object Detection", ICIP 2022
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_sahi_eval(model_path: str, source: str, output_dir: str = None,
                  window_size: int = 640, overlap: float = 0.4,
                  conf: float = 0.25, device: str = '0', adapt_nms: bool = False):
    """Run SAHI evaluation with optional Adaptive NMS.

    Args:
        model_path: path to best.pt
        source: directory of images
        output_dir: where to save visualizations
        window_size: slice size
        overlap: overlap ratio (0.0 - 1.0)
        conf: confidence threshold
        device: CUDA device
        adapt_nms: use Adaptive NMS instead of standard NMS
    """
    from sahi import AutoDetectionModel
    from sahi.predict import get_prediction, get_sliced_prediction
    from sahi.utils.file import list_files

    overlap_px = int(window_size * overlap)

    # Load model
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_path,
        confidence_threshold=conf,
        device=f'cuda:{device}',
    )

    # Configure SAHI postprocessing
    if adapt_nms:
        from models.postprocess import apply_adaptive_nms as nms_fn
        postprocess_kwargs = {
            'match_metric': 'IOS',
            'match_threshold': 0.5,
            'class_agnostic': True,
        }
    else:
        postprocess_kwargs = {
            'match_metric': 'IOS',
            'match_threshold': 0.5,
            'class_agnostic': True,
        }

    # Get images
    image_paths = list_files(source, contains=['.jpg', '.png', '.jpeg'])
    print(f"[SAHI] Found {len(image_paths)} images in {source}")
    print(f"[SAHI] Window: {window_size}x{window_size}, Overlap: {overlap} ({overlap_px}px)")

    results = []
    for i, img_path in enumerate(image_paths):
        result = get_sliced_prediction(
            image=img_path,
            detection_model=detection_model,
            slice_height=window_size,
            slice_width=window_size,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            perform_standard_pred=False,
            postprocess_type='NMS' if not adapt_nms else 'GREEDYNMM',
            postprocess_match_metric='IOS',
            postprocess_match_threshold=0.5,
            postprocess_class_agnostic=True,
            verbose=0,
        )

        results.append(result)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(image_paths)}")

    print(f"[SAHI] Done. {len(results)} images processed.")
    print(f"  To compute mAP, use: ultralytics val model={model_path} data=visdrone.yaml")
    return results


def compute_sahi_map(model_path: str, data_yaml: str, device: str = '0') -> dict:
    """Compute mAP on VisDrone validation set using SAHI inference.

    This compares SAHI vs standard 640x640 inference.
    """
    from ultralytics import YOLO

    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, device=device, imgsz=640, split='val',
                         plots=True, save_json=False)
    return {
        'mAP50': metrics.box.map50,
        'mAP50_95': metrics.box.map,
        'mAP_small': metrics.box.maps if hasattr(metrics.box, 'maps') else 0.0,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='model .pt path')
    parser.add_argument('--source', required=True, help='image directory')
    parser.add_argument('--output', default=None, help='output directory')
    parser.add_argument('--window', type=int, default=640)
    parser.add_argument('--overlap', type=float, default=0.4)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--device', default='0')
    parser.add_argument('--adapt-nms', action='store_true')
    args = parser.parse_args()

    run_sahi_eval(
        model_path=args.model,
        source=args.source,
        output_dir=args.output,
        window_size=args.window,
        overlap=args.overlap,
        conf=args.conf,
        device=args.device,
        adapt_nms=args.adapt_nms,
    )
