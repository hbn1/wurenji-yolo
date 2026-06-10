"""
Mini-VisDrone: fast-iteration subset for debugging NWD Loss, P2 head, etc.
======================================================================

Samples 20% of VisDrone train set (images + labels) into a mini subset.
Use this for quick 5-epoch sanity checks before running full training.

Usage:
    python utils/mini_visdrone.py --ratio 0.2                    # create subset
    python utils/mini_visdrone.py --ratio 0.05 --seed 42        # 5% subset
    python utils/mini_visdrone.py --clean                        # remove subset
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path


def create_mini_visdrone(ratio: float = 0.2, seed: int = 42):
    """Sample a subset of VisDrone train set for fast debugging.

    Copies images and labels into data/VisDrone/mini_train/.
    Also generates data/mini_visdrone.yaml for direct use.
    """
    random.seed(seed)

    project_root = Path(__file__).resolve().parents[1]
    train_img = project_root / "data" / "VisDrone" / "train" / "images"
    train_lbl = project_root / "data" / "VisDrone" / "train" / "labels"

    mini_img = project_root / "data" / "VisDrone" / "mini_train" / "images"
    mini_lbl = project_root / "data" / "VisDrone" / "mini_train" / "labels"

    # Clean old
    if mini_img.exists():
        shutil.rmtree(mini_img.parent)

    mini_img.mkdir(parents=True, exist_ok=True)
    mini_lbl.mkdir(parents=True, exist_ok=True)

    # List all training images
    img_files = sorted([f for f in os.listdir(train_img)
                        if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    n_total = len(img_files)
    n_sample = max(1, int(n_total * ratio))

    sampled = random.sample(img_files, n_sample)

    for fname in sampled:
        # Copy image
        shutil.copy2(train_img / fname, mini_img / fname)
        # Copy label
        lbl_name = os.path.splitext(fname)[0] + ".txt"
        lbl_src = train_lbl / lbl_name
        if lbl_src.exists():
            shutil.copy2(lbl_src, mini_lbl / lbl_name)

    # Generate mini yaml
    yaml_path = project_root / "data" / "mini_visdrone.yaml"
    yaml_content = f"""# Mini-VisDrone ({ratio*100:.0f}% subset, seed={seed})
# {n_sample} images sampled from {n_total} total.
# For fast debugging: use this yaml instead of visdrone.yaml

path: {project_root / 'data' / 'VisDrone'}
train: mini_train/images
val: val/images

nc: 10
names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")

    return n_sample, n_total, yaml_path


def clean_mini_visdrone():
    """Remove mini-VisDrone data and yaml."""
    project_root = Path(__file__).resolve().parents[1]
    mini_dir = project_root / "data" / "VisDrone" / "mini_train"
    yaml_path = project_root / "data" / "mini_visdrone.yaml"

    for p in [mini_dir, yaml_path]:
        if p.exists():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mini-VisDrone subset manager")
    parser.add_argument("--ratio", type=float, default=0.2,
                        help="Sampling ratio (default: 0.2 = 20%%)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true",
                        help="Remove mini subset")
    args = parser.parse_args()

    if args.clean:
        clean_mini_visdrone()
        print("[OK] Mini-VisDrone removed.")
    else:
        n_sample, n_total, yaml_path = create_mini_visdrone(args.ratio, args.seed)
        print(f"[OK] Mini-VisDrone created:")
        print(f"  {n_sample}/{n_total} images ({100*n_sample/n_total:.0f}%)")
        print(f"  YAML: {yaml_path}")
        print(f"  Usage with ablation: --ablation D --epochs 5 --data data/mini_visdrone.yaml")
