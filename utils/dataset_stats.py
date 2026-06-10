"""
Dataset statistics for VisDrone — analyze small-object distribution
to calibrate hyperparameters (NWD threshold, RepLoss weights, etc.).
"""

import os
from collections import defaultdict

import numpy as np
import cv2


def analyze_dataset(label_dir: str, img_dir: str = None) -> dict:
    """Analyze a YOLO-format VisDrone dataset.

    Args:
        label_dir: path to YOLO-format label .txt files
        img_dir: (optional) path to images for size checking

    Returns:
        dict with statistics
    """
    stats = {
        'total_images': 0,
        'total_objects': 0,
        'objects_per_class': defaultdict(int),
        'object_areas': [],       # absolute areas (px^2)
        'object_scales': [],      # sqrt(area) — rough "size"
        'density_per_image': [],  # objects per image
        'small_count': 0,         # area < 32²
        'medium_count': 0,        # 32² <= area < 96²
        'large_count': 0,         # area >= 96²
    }

    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    stats['total_images'] = len(label_files)

    for lf in label_files:
        path = os.path.join(label_dir, lf)
        with open(path, 'r') as f:
            lines = f.readlines()

        obj_count = 0
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            w_norm, h_norm = float(parts[3]), float(parts[4])

            stats['objects_per_class'][cls_id] += 1
            stats['total_objects'] += 1
            obj_count += 1

            # If image available, compute absolute area
            if img_dir:
                img_name = lf.replace('.txt', '.jpg')
                img_path = os.path.join(img_dir, img_name)
                if os.path.exists(img_path):
                    h, w = cv2.imread(img_path).shape[:2]
                    area = w_norm * h_norm * w * h
                    stats['object_areas'].append(area)
                    stats['object_scales'].append(np.sqrt(area))

        stats['density_per_image'].append(obj_count)

    # Categorize by COCO/VisDrone convention
    areas = np.array(stats['object_areas'])
    if len(areas) > 0:
        stats['small_count'] = (areas < 32 * 32).sum()
        stats['medium_count'] = ((areas >= 32 * 32) & (areas < 96 * 96)).sum()
        stats['large_count'] = (areas >= 96 * 96).sum()

    return stats


def print_statistics(stats: dict):
    """Pretty-print dataset statistics."""
    print("=" * 60)
    print("VisDrone Dataset Statistics")
    print("=" * 60)

    print(f"\nTotal images:    {stats['total_images']}")
    print(f"Total objects:   {stats['total_objects']}")
    print(f"Avg obj/image:   {stats['total_objects'] / max(stats['total_images'], 1):.1f}")

    if stats['density_per_image']:
        densities = np.array(stats['density_per_image'])
        print(f"Max density:     {densities.max()}")
        print(f"Std density:     {densities.std():.1f}")

    if stats['object_areas']:
        areas = np.array(stats['object_areas'])
        print(f"\nObject area (px^2):")
        print(f"  Mean:  {areas.mean():.0f}")
        print(f"  Median: {np.median(areas):.0f}")
        print(f"  Min:    {areas.min():.0f}")
        print(f"  Max:    {areas.max():.0f}")

        total = len(areas) or 1
        print(f"\nSize distribution:")
        print(f"  Small (< 32x32):     {stats['small_count']:5d}  ({100*stats['small_count']/total:.1f}%)")
        print(f"  Medium (32x32-96x96): {stats['medium_count']:5d}  ({100*stats['medium_count']/total:.1f}%)")
        print(f"  Large (> 96x96):      {stats['large_count']:5d}  ({100*stats['large_count']/total:.1f}%)")

    if stats['objects_per_class']:
        print(f"\nObjects per class:")
        class_names = {0: 'pedestrian', 1: 'people', 2: 'bicycle', 3: 'car',
                       4: 'van', 5: 'truck', 6: 'tricycle', 7: 'awning-tricycle',
                       8: 'bus', 9: 'motor'}
        for cls_id in sorted(stats['objects_per_class'].keys()):
            name = class_names.get(cls_id, f'class_{cls_id}')
            count = stats['objects_per_class'][cls_id]
            pct = 100 * count / stats['total_objects']
            print(f"  {cls_id:2d} {name:<18s}: {count:6d}  ({pct:.1f}%)")

    print("=" * 60)


if __name__ == '__main__':
    import sys
    label_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/VisDrone/train/labels'
    img_dir = sys.argv[2] if len(sys.argv) > 2 else 'data/VisDrone/train/images'
    print_statistics(analyze_dataset(label_dir, img_dir))
