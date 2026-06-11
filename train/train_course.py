import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Course Project Ablation")
    parser.add_argument("--ablation", default="A", choices=["A", "B", "C"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    project_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments")
    data_yaml = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "visdrone.yaml")

    if args.ablation == "A":
        model_path = "yolov8s.pt"
        use_nwd, use_cbam = False, False
    elif args.ablation == "B":
        model_path = "yolov8s.pt"
        use_nwd, use_cbam = True, False
    else:
        # Ablation C: CBAM yaml + pretrained weights
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                   "configs", "cghr", "yolov8s_cbam.yaml")
        use_nwd, use_cbam = True, True

    print(f"\n{'='*60}")
    print(f"  Course Project -- Ablation {args.ablation}")
    print(f"  NWD Loss: {use_nwd}   CBAM: {use_cbam}")
    print(f"  Epochs: {args.epochs}   Batch: {args.batch}   ImgSz: {args.imgsz}")
    print(f"{'='*60}\n")

    # ---- Register CBAM if needed ----
    if use_cbam:
        from models.cghr_model import register_custom_modules
        register_custom_modules()

    # ---- Inject NWD if needed (BEFORE model.train) ----
    if use_nwd:
        from models.nwd_inject import inject_nwd_loss
        inject_nwd_loss(area_threshold=1024.0, nwd_weight=0.25)

    # ---- Build model ----
    model = YOLO(model_path)

    # ---- Load pretrained weights for CBAM model ----
    if use_cbam and not args.resume:
        from models.cghr_model import load_pretrained_for_cbam; load_pretrained_for_cbam(model.model, "yolov8s.pt")

    # ---- Common training kwargs ----
    train_kwargs = {
        "data": data_yaml,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "project": project_dir,
        "name": f"course_ablation_{args.ablation}",
        "resume": args.resume,
        "optimizer": "AdamW",
        "lr0": 0.001, "lrf": 0.01, "weight_decay": 0.0005,
        "workers": 4, "cache": "disk",
        "warmup_epochs": 3, "cos_lr": True,
        "mosaic": 1.0, "close_mosaic": 8, "mixup": 0.1,
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "degrees": 10.0, "translate": 0.1, "scale": 0.5,
        "fliplr": 0.5,
        "val": True, "save": True, "save_period": 10,
        "patience": 10, "plots": True, "exist_ok": True,
    }

    results = model.train(**train_kwargs)

    print(f"\n[OK] Course Ablation {args.ablation} done.")
    print(f"  Best: {results.save_dir}/weights/best.pt")

    if use_nwd:
        from models.nwd_inject import remove_nwd_loss
        remove_nwd_loss()


if __name__ == "__main__":
    main()
