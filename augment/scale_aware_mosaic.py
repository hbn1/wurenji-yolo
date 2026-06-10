"""
Scale-Aware Mosaic: Mosaic augmentation that preserves original resolution
for small-object regions in UAV aerial images.

Standard Mosaic: 4 random images, each resized to target_size → stitched.
Scale-Aware Mosaic: at least 1 image is a random crop from the ORIGINAL
high-resolution image (not resized), ensuring small targets retain their
true pixel footprint during training.

Algorithm:
  1. For each batch, select a high-res source image
  2. Divide it into a 4x4 grid, count small-objects (< 32x32) per cell
  3. Weighted sample a grid cell → crop at original resolution
  4. Fill remaining 3 slots with standard Mosaic crops
  5. Assemble 2x2 mosaic output
"""

import random
import math
from typing import Tuple, List, Optional

import cv2
import numpy as np


def count_small_objects(labels: np.ndarray, img_w: int, img_h: int,
                        small_threshold: float = 32.0 * 32.0) -> dict:
    """Count small objects in each grid cell of an image.

    Args:
        labels: (N, 5) [class, cx, cy, w, h] in normalized coordinates
        img_w, img_h: image dimensions
        small_threshold: area threshold for 'small' objects (in px^2)

    Returns:
        dict with 'grid_counts' (4x4 array) and 'total_small'
    """
    grid = np.zeros((4, 4), dtype=np.int32)

    if len(labels) == 0:
        return {'grid_counts': grid, 'total_small': 0}

    # Denormalize
    cx = labels[:, 1] * img_w
    cy = labels[:, 2] * img_h
    bw = labels[:, 3] * img_w
    bh = labels[:, 4] * img_h

    areas = bw * bh
    small_mask = areas < small_threshold

    if not small_mask.any():
        return {'grid_counts': grid, 'total_small': 0}

    small_cx = cx[small_mask]
    small_cy = cy[small_mask]

    # Assign to grid cells
    cell_w = img_w / 4
    cell_h = img_h / 4
    gx = np.clip((small_cx / cell_w).astype(np.int32), 0, 3)
    gy = np.clip((small_cy / cell_h).astype(np.int32), 0, 3)

    for i, j in zip(gx, gy):
        grid[j, i] += 1  # grid[y, x]

    return {
        'grid_counts': grid,
        'total_small': small_mask.sum()
    }


def sample_highres_crop_region(img_w: int, img_h: int, grid_counts: np.ndarray,
                                target_size: int = 640) -> Tuple[int, int, int, int]:
    """Sample a crop region from the high-res image, weighted by small-object density.

    Returns:
        (x1, y1, x2, y2) crop coordinates
    """
    total = grid_counts.sum()

    if total > 0:
        # Weighted sampling: cells with more small objects have higher probability
        probs = grid_counts.ravel().astype(np.float32) / total
        idx = np.random.choice(16, p=probs)
    else:
        # Uniform sampling if no small objects
        idx = np.random.randint(0, 16)

    gy, gx = divmod(idx, 4)

    cell_w = img_w / 4
    cell_h = img_h / 4

    # Add randomness within the cell (±20% of cell size)
    x1 = int(gx * cell_w + np.random.uniform(-0.2, 0.2) * cell_w)
    y1 = int(gy * cell_h + np.random.uniform(-0.2, 0.2) * cell_h)

    # Ensure crop size >= target_size with some margin
    crop_w = max(target_size, int(cell_w * 1.3))
    crop_h = max(target_size, int(cell_h * 1.3))

    x2 = min(x1 + crop_w, img_w)
    y2 = min(y1 + crop_h, img_h)

    # Adjust x1, y1 if we hit the right/bottom boundary
    x1 = max(0, x2 - crop_w)
    y1 = max(0, y2 - crop_h)

    return (x1, y1, x2, y2)


def crop_and_resize(img: np.ndarray, labels: np.ndarray,
                    crop_region: Tuple[int, int, int, int],
                    target_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Crop image at region, then resize to target_size.

    Args:
        img: (H, W, 3) image
        labels: (N, 5) normalized labels [cls, cx, cy, w, h]
        crop_region: (x1, y1, x2, y2)
        target_size: output size

    Returns:
        cropped_and_resized (target_size, target_size, 3) image,
        adjusted (N, 5) normalized labels
    """
    x1, y1, x2, y2 = crop_region
    crop_w = x2 - x1
    crop_h = y2 - y1

    # Crop image
    cropped = img[y1:y2, x1:x2]

    # Adjust labels: shift to crop coordinates, then renormalize
    if len(labels) > 0:
        labels = labels.copy()
        # denormalize to absolute
        abs_cx = labels[:, 1] * img.shape[1]
        abs_cy = labels[:, 2] * img.shape[0]
        abs_w = labels[:, 3] * img.shape[1]
        abs_h = labels[:, 4] * img.shape[0]

        # Shift to crop coordinates
        abs_cx -= x1
        abs_cy -= y1

        # Renormalize to crop
        labels[:, 1] = abs_cx / crop_w
        labels[:, 2] = abs_cy / crop_h
        labels[:, 3] = abs_w / crop_w
        labels[:, 4] = abs_h / crop_h

        # Filter out labels outside crop
        valid = (abs_cx >= 0) & (abs_cx <= crop_w) & \
                (abs_cy >= 0) & (abs_cy <= crop_h)
        labels = labels[valid]

    # Resize to target_size
    cropped_resized = cv2.resize(cropped, (target_size, target_size),
                                  interpolation=cv2.INTER_LINEAR)

    return cropped_resized, labels


def assemble_mosaic(tiles: List[Tuple[np.ndarray, np.ndarray]],
                    target_size: int = 640) -> Tuple[np.ndarray, np.ndarray]:
    """Assemble 4 tiles into a 2x2 mosaic with adjusted labels.

    Args:
        tiles: list of 4 (img, labels) tuples, each img is (S, S, 3)

    Returns:
        mosaic (2S, 2S, 3) image and concatenated labels
    """
    s = target_size
    mosaic = np.zeros((s * 2, s * 2, 3), dtype=np.uint8)
    all_labels = []

    positions = [
        (0, 0),         # top-left
        (s, 0),         # top-right
        (0, s),         # bottom-left
        (s, s),         # bottom-right
    ]

    for (img, labels), (off_x, off_y) in zip(tiles, positions):
        h, w = img.shape[:2]
        mosaic[off_y:off_y + h, off_x:off_x + w] = img

        if len(labels) > 0:
            labels = labels.copy()
            # Scale labels: from tile's SxS to mosaic's 2Sx2S
            # cx/w in tile space → in mosaic space
            labels[:, 1] = (labels[:, 1] * w + off_x) / (s * 2)
            labels[:, 2] = (labels[:, 2] * h + off_y) / (s * 2)
            labels[:, 3] = labels[:, 3] * w / (s * 2)
            labels[:, 4] = labels[:, 4] * h / (s * 2)
            all_labels.append(labels)

    if all_labels:
        all_labels = np.concatenate(all_labels, axis=0)
    else:
        all_labels = np.zeros((0, 5), dtype=np.float32)

    return mosaic, all_labels


class ScaleAwareMosaic:
    """Drop-in augmentation for YOLO-style training dataloaders.

    Usage (conceptual — integrate into Ultralytics' BaseDataset._load_mosaic):
        mosaic = ScaleAwareMosaic(target_size=640)
        img, labels = mosaic(images, labels_list, highres_indices=[2])
    """

    def __init__(self, target_size: int = 640, small_threshold: float = 32.0 * 32.0,
                 prob_scale_aware: float = 0.5):
        """
        Args:
            target_size: output mosaic size per tile
            small_threshold: area threshold for 'small' objects
            prob_scale_aware: probability of using scale-aware mode (vs standard)
        """
        self.target_size = target_size
        self.small_threshold = small_threshold
        self.prob_scale_aware = prob_scale_aware

    def __call__(self, images: List[np.ndarray],
                 labels_list: List[np.ndarray],
                 image_shapes: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            images: 4 images to mosaic (each can be different sizes)
            labels_list: 4 label arrays [cls, cx, cy, w, h] normalized
            image_shapes: original (W, H) of each image

        Returns:
            mosaic_img (1280, 1280, 3), concatenated labels
        """
        if np.random.random() > self.prob_scale_aware:
            # Fall back to standard mosaic — caller handles this
            return None, None

        # Pick one image as the high-res source (highest resolution)
        areas = [w * h for w, h in image_shapes]
        highres_idx = int(np.argmax(areas))

        tiles = []

        for i in range(4):
            img = images[i]
            labels = labels_list[i]
            w, h = image_shapes[i]

            if i == highres_idx and max(w, h) > self.target_size:
                # Scale-aware: crop from original resolution
                stats = count_small_objects(labels, w, h, self.small_threshold)
                region = sample_highres_crop_region(w, h, stats['grid_counts'],
                                                     self.target_size)
                img_processed, labels_processed = crop_and_resize(
                    img, labels, region, self.target_size
                )
            else:
                # Standard resize
                img_processed = cv2.resize(img, (self.target_size, self.target_size))
                labels_processed = labels

            tiles.append((img_processed, labels_processed))

        return assemble_mosaic(tiles, self.target_size)
