# CGHR-YOLOv8s VisDrone Experiments

This repository contains a YOLOv8s-based VisDrone object detection course project.
The experiments compare:

- **A**: YOLOv8s baseline
- **B**: YOLOv8s + NWD loss
- **C**: YOLOv8s + NWD loss + CBAM attention

## Project Structure

```text
augment/                 data augmentation utilities
configs/                 YOLO model and experiment configs
data/visdrone.yaml       dataset config template
eval/                    evaluation scripts
models/                  CBAM and NWD loss injection code
scripts/                 dataset preparation helpers
train/                   training entry points
utils/                   dataset/statistics utilities
experiments/             lightweight result CSVs and curves
docs/                    experiment report
```

Large files such as datasets, model weights, TensorBoard logs, and training cache files are intentionally excluded from Git.

## Training

Install Ultralytics YOLO and prepare VisDrone in YOLO format, then run:

```powershell
python train/train_course.py --ablation A --epochs 50 --batch 8 --imgsz 640 --device 0
python train/train_course.py --ablation B --epochs 50 --batch 8 --imgsz 640 --device 0
python train/train_course.py --ablation C --epochs 50 --batch 8 --imgsz 640 --device 0
```

## Current Results

| Experiment | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| YOLOv8s | 0.48500 | 0.37211 | 0.37922 | 0.22443 |
| YOLOv8s + NWD | 0.48928 | 0.37458 | 0.39832 | 0.23851 |
| YOLOv8s + NWD + CBAM | 0.47021 | 0.36510 | 0.40758 | 0.24561 |

See `experiments/` for CSV logs and summary curves, and `docs/` for the Word report.
