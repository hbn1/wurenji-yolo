"""
Phase 0 — Baseline Training: YOLOv8s on VisDrone

Usage:
    python train/train_baseline.py [--model yolov8s] [--epochs 100]

Output:
    experiments/baseline_yolov8s/weights/best.pt
"""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolov8s.pt', help='pretrained model')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default='0')
    parser.add_argument('--name', default='baseline_yolov8s')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    project_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'experiments')
    data_yaml = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'visdrone.yaml')

    # Check dataset exists
    if not os.path.exists(data_yaml):
        print(f"[ERROR] Data config not found: {data_yaml}")
        print("  Run: python scripts/download_visdrone.py first")
        sys.exit(1)

    model = YOLO(args.model)

    print(f"\n{'='*60}")
    print(f"Baseline Training: {args.model}")
    print(f"Dataset: {data_yaml}")
    print(f"Image size: {args.imgsz}, Batch: {args.batch}, Epochs: {args.epochs}")
    print(f"{'='*60}\n")

    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=4,
        device=args.device,
        project=project_dir,
        name=args.name,
        resume=args.resume,
        cache='disk', 
        # Optimizer
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,
        # Augmentation
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        close_mosaic=8,
        # Evaluation
        val=True,
        save=True,
        save_period=10,
        patience=10,
        # Reporting
        plots=True,
        exist_ok=True,
    )

    print(f"\n[OK] Baseline training complete.")
    print(f"  Best model: {results.save_dir}/weights/best.pt")
    print(f"  mAP@50:     {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"  mAP@50-95:  {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")


if __name__ == '__main__':
    main()
