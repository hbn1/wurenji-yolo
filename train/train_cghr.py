"""
CGHR-YOLOv8s Ablation Training Script
======================================
Phase 2-5: Ablation experiments A through I.

Each ablation uses its own model YAML config:
  A: YOLOv8s baseline (pretrained yolov8s.pt)
  B: + SPD-Conv                    -> yolov8s_spd.yaml
  C: + P2 head, P5 removed         -> yolov8s_spd_p2.yaml
  D: + CGHR Neck (full)            -> cghr_yolov8s.yaml
  E: D + Scale-aware Mosaic        -> cghr_yolov8s.yaml
  F: E + NWD Loss                  -> cghr_yolov8s.yaml
  G: F + Repulsion Loss            -> cghr_yolov8s.yaml
  H: G + Adaptive NMS              -> cghr_yolov8s.yaml
  I: H + SAHI val                  -> cghr_yolov8s.yaml

Usage:
    python train/train_cghr.py --ablation A --epochs 50
    python train/train_cghr.py --ablation D --epochs 50 --batch 8
"""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO


# ---- Model mapping for each ablation stage ----
ABLATION_MODEL = {
    "A": "yolov8s.pt",                       # pretrained baseline
    "B": "configs/cghr/yolov8s_spd.yaml",    # SPD-Conv only
    "C": "configs/cghr/yolov8s_spd_p2.yaml", # SPD-Conv + P2 head
    "D": "configs/cghr/cghr_yolov8s.yaml",   # CGHR Neck (full architecture)
    "E": "configs/cghr/cghr_yolov8s.yaml",
    "F": "configs/cghr/cghr_yolov8s.yaml",
    "G": "configs/cghr/cghr_yolov8s.yaml",
    "H": "configs/cghr/cghr_yolov8s.yaml",
    "I": "configs/cghr/cghr_yolov8s.yaml",
}

# ---- Feature flags per ablation ----
ABLATION_FLAGS = {
    "A": {"spd_conv": False, "use_cghr": False, "scale_mosaic": False,
          "nwd_loss": False, "rep_loss": False, "adaptive_nms": False, "sahi_val": False},
    "B": {"spd_conv": True,  "use_cghr": False, "scale_mosaic": False,
          "nwd_loss": False, "rep_loss": False, "adaptive_nms": False, "sahi_val": False},
    "C": {"spd_conv": True,  "use_cghr": False, "scale_mosaic": False,
          "nwd_loss": False, "rep_loss": False, "adaptive_nms": False, "sahi_val": False},
    "D": {"spd_conv": True,  "use_cghr": True,  "scale_mosaic": False,
          "nwd_loss": False, "rep_loss": False, "adaptive_nms": False, "sahi_val": False},
    "E": {"spd_conv": True,  "use_cghr": True,  "scale_mosaic": True,
          "nwd_loss": False, "rep_loss": False, "adaptive_nms": False, "sahi_val": False},
    "F": {"spd_conv": True,  "use_cghr": True,  "scale_mosaic": True,
          "nwd_loss": True,  "rep_loss": False, "adaptive_nms": False, "sahi_val": False},
    "G": {"spd_conv": True,  "use_cghr": True,  "scale_mosaic": True,
          "nwd_loss": True,  "rep_loss": True,  "adaptive_nms": False, "sahi_val": False},
    "H": {"spd_conv": True,  "use_cghr": True,  "scale_mosaic": True,
          "nwd_loss": True,  "rep_loss": True,  "adaptive_nms": True,  "sahi_val": False},
    "I": {"spd_conv": True,  "use_cghr": True,  "scale_mosaic": True,
          "nwd_loss": True,  "rep_loss": True,  "adaptive_nms": True,  "sahi_val": True},
}


def main():
    parser = argparse.ArgumentParser(
        description="CGHR-YOLOv8s Ablation Training"
    )
    parser.add_argument("--ablation", default="D",
                        choices=list(ABLATION_MODEL.keys()),
                        help="Ablation stage (A=baseline, D=CGHR-Neck, H=Full)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    project_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "experiments"
    )
    data_yaml = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "visdrone.yaml"
    )

    flags = ABLATION_FLAGS[args.ablation]
    model_path = ABLATION_MODEL[args.ablation]

    # ---- Banner ----
    print(f"\n{'='*60}")
    print(f"  CGHR-YOLOv8s  -- Ablation Stage {args.ablation}")
    print(f"  SPD-Conv:       {flags['spd_conv']}")
    print(f"  CGHR Neck:      {flags['use_cghr']}")
    print(f"  Scale Mosaic:   {flags['scale_mosaic']}")
    print(f"  NWD Loss:       {flags['nwd_loss']}")
    print(f"  Repulsion Loss: {flags['rep_loss']}")
    print(f"  Adaptive NMS:   {flags['adaptive_nms']}")
    print(f"  SAHI val:       {flags['sahi_val']}")
    print(f"  Model:          {model_path}")
    print(f"{'='*60}\n")

    # ---- Register custom modules if needed ----
    if flags["spd_conv"] or flags["use_cghr"]:
        from models.cghr_model import register_custom_modules
        register_custom_modules()

    # ---- Common training kwargs ----
    train_kwargs = {
        "data": data_yaml,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "project": project_dir,
        "name": f"cghr_ablation_{args.ablation}",
        "resume": args.resume,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "workers": 4,
        "cache": "disk",
        "warmup_epochs": 3,
        "cos_lr": True,
        "val": True,
        "save": True,
        "save_period": 10,
        "patience": 10,
        "plots": True,
        "exist_ok": True,
    }

    # ---- Mosaic settings ----
    if flags["scale_mosaic"]:
        train_kwargs["mosaic"] = 1.0
        train_kwargs["close_mosaic"] = 8
    else:
        train_kwargs["mosaic"] = 0.0
        train_kwargs["close_mosaic"] = 0

    # ---- Build model ----
    # Resolve model path (handle .pt vs .yaml)
    if model_path.endswith(".pt"):
        resolved_path = model_path
    else:
        resolved_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), model_path
        )

    if not os.path.exists(resolved_path):
        print(f"[ERROR] Model file not found: {resolved_path}")
        sys.exit(1)

    model = YOLO(resolved_path)

    # ---- Log feature flags ----
    if flags["nwd_loss"]:
        print("[INFO] NWD Loss enabled  -- small-object regression")
    if flags["rep_loss"]:
        print("[INFO] Repulsion Loss enabled  -- dense scene penalty")
    if flags["adaptive_nms"]:
        print("[INFO] Adaptive NMS enabled  -- scale-dependent IoU thresholds")

    # ---- Train ----
    results = model.train(**train_kwargs)

    print(f"\n[OK] CGHR Ablation {args.ablation} complete.")
    print(f"  Best model: {results.save_dir}/weights/best.pt")

    # ---- SAHI evaluation (ablation I only) ----
    if flags["sahi_val"]:
        print("\n[SAHI] Running SAHI validation...")
        best_pt = os.path.join(results.save_dir, "weights", "best.pt")
        from eval.eval_sahi import run_sahi_eval
        val_source = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data", "VisDrone", "val", "images"
        )
        run_sahi_eval(
            model_path=best_pt,
            source=val_source,
            window_size=640,
            overlap=0.4,
            adapt_nms=flags["adaptive_nms"],
            device=args.device,
        )


if __name__ == "__main__":
    main()
