"""
Download and prepare VisDrone2019 dataset in YOLO format.

Dataset: VisDrone2019-DET (Object Detection in Images)
Source: https://github.com/VisDrone/VisDrone-Dataset
Classes: pedestrian, people, bicycle, car, van, truck, tricycle,
         awning-tricycle, bus, motor

After download, converts VisDrone annotations (x1,y1,w,h) to YOLO format
(cx, cy, w, h normalized).
"""

import os
import sys
import shutil
import argparse
import zipfile
from pathlib import Path
from collections import defaultdict


# VisDrone official Google Drive / mirror links
VISDRONE_URLS = {
    'train_images': 'https://github.com/VisDrone/VisDrone2019-DET-toolkit/raw/master/VisDrone2019-DET-train.zip',
    'val_images': 'https://github.com/VisDrone/VisDrone2019-DET-toolkit/raw/master/VisDrone2019-DET-val.zip',
}

# Alternative: VisDrone provides BaiduPan links, but GitHub mirror is easier
# If GitHub links are broken, use:
# https://drive.google.com/drive/folders/1-5nibmKjA1v9bnZlPdReQSYtBn3hGx5A


def convert_visdrone_to_yolo(anno_dir: str, img_dir: str, out_dir: str):
    """Convert VisDrone annotation .txt to YOLO format.

    VisDrone format per line:
      <bbox_left>, <bbox_top>, <bbox_width>, <bbox_height>,
      <score>, <object_category>, <truncation>, <occlusion>

    We need: class_id cx cy w h (normalized)
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'labels'), exist_ok=True)

    anno_files = [f for f in os.listdir(anno_dir) if f.endswith('.txt')]
    stats = defaultdict(int)

    for anno_name in anno_files:
        anno_path = os.path.join(anno_dir, anno_name)
        img_name = anno_name.replace('.txt', '.jpg')
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path):
            print(f"  [WARN] Image not found: {img_name}, skipping")
            continue

        # Read image dimensions
        import cv2
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [WARN] Cannot read: {img_name}, skipping")
            continue
        img_h, img_w = img.shape[:2]

        # Copy image
        shutil.copy2(img_path, os.path.join(out_dir, 'images', img_name))

        # Read annotations
        yolo_lines = []
        with open(anno_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue

                x1 = float(parts[0])
                y1 = float(parts[1])
                w = float(parts[2])
                h = float(parts[3])
                cls_id = int(parts[5]) - 1  # VisDrone is 1-indexed

                # Skip ignored regions (class 0 → -1) and truncation=2 (fully truncated)
                if cls_id < 0 or cls_id > 9:
                    continue
                truncation = int(parts[6]) if len(parts) > 6 else 0
                if truncation == 2:
                    continue

                # VisDrone xywh -> YOLO cx cy w h normalized
                cx = (x1 + w / 2) / img_w
                cy = (y1 + h / 2) / img_h
                nw = w / img_w
                nh = h / img_h

                # Clamp to [0, 1]
                cx = max(0, min(1, cx))
                cy = max(0, min(1, cy))
                nw = max(0, min(1, nw))
                nh = max(0, min(1, nh))

                yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                stats[cls_id] += 1

        # Write YOLO labels
        label_out = os.path.join(out_dir, 'labels', anno_name)
        with open(label_out, 'w') as f:
            f.write('\n'.join(yolo_lines))

    print(f"  Converted {len(anno_files)} files")
    print(f"  Class distribution:")
    class_names = {0: 'pedestrian', 1: 'people', 2: 'bicycle', 3: 'car',
                   4: 'van', 5: 'truck', 6: 'tricycle', 7: 'awning-tricycle',
                   8: 'bus', 9: 'motor'}
    for cls_id in sorted(stats.keys()):
        print(f"    {cls_id:2d} {class_names.get(cls_id, '?'):<18s}: {stats[cls_id]:6d}")

    return stats


def download_visdrone(data_dir: str):
    """Download VisDrone dataset manually (requires user action).

    Since VisDrone is >2GB and hosted on Google Drive / BaiduPan,
    provides clear instructions for manual download.

    VisDrone2019-DET:
      - Google Drive: https://drive.google.com/drive/folders/1-5nibmKjA1v9bnZlPdReQSYtBn3hGx5A
      - Official site: http://aiskyeye.com/
    """
    print("=" * 60)
    print("VisDrone2019 Dataset Download Instructions")
    print("=" * 60)
    print()
    print("The dataset is ~2.5 GB and cannot be auto-downloaded via script.")
    print()
    print("Option 1 — Google Drive (recommended):")
    print("  https://drive.google.com/drive/folders/1-5nibmKjA1v9bnZlPdReQSYtBn3hGx5A")
    print("  Download: VisDrone2019-DET-train.zip, VisDrone2019-DET-val.zip")
    print()
    print("Option 2 — Official Website:")
    print("  http://aiskyeye.com/")
    print("  Navigate: Dataset → Object Detection in Images → VisDrone2019-DET")
    print()
    print("=" * 60)
    print("After downloading, place files in:")
    print(f"  {data_dir}/VisDrone/VisDrone2019-DET-train.zip")
    print(f"  {data_dir}/VisDrone/VisDrone2019-DET-val.zip")
    print()
    print("Then run:")
    print(f"  python scripts/convert_visdrone.py --data-dir {data_dir}")
    print("=" * 60)


def extract_and_convert(data_dir: str):
    """Extract downloaded VisDrone zips and convert to YOLO format."""
    base = Path(data_dir) / 'VisDrone'

    # Check for downloaded zips
    train_zip = base / 'VisDrone2019-DET-train.zip'
    val_zip = base / 'VisDrone2019-DET-val.zip'

    if not train_zip.exists():
        print(f"[ERROR] {train_zip} not found. Please download first.")
        download_visdrone(data_dir)
        return

    # Extract
    for zip_path, split in [(train_zip, 'train'), (val_zip, 'val')]:
        print(f"[EXTRACT] {zip_path.name} ...")
        extract_to = base / 'raw' / split
        os.makedirs(extract_to, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(base / 'raw')

    # Convert
    for split in ['train', 'val']:
        raw_split = base / 'raw' / f'VisDrone2019-DET-{split}'
        if not raw_split.exists():
            print(f"[WARN] {raw_split} not found after extraction, skipping {split}")
            continue

        anno_dir = raw_split / 'annotations'
        img_dir = raw_split / 'images'
        out_dir = base / split

        print(f"\n[CONVERT] {split} ...")
        stats = convert_visdrone_to_yolo(str(anno_dir), str(img_dir), str(out_dir))

    print("\n[OK] VisDrone dataset prepared in YOLO format.")
    print(f"  Data root: {base}")
    print(f"  Training images:  {base}/train/images/")
    print(f"  Validation images: {base}/val/images/")
    print("\n  Update data/visdrone.yaml if needed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--convert-only', action='store_true',
                        help='Skip download, just convert existing extracted data')
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.data_dir)

    if args.convert_only:
        extract_and_convert(data_dir)
    else:
        extract_and_convert(data_dir)
