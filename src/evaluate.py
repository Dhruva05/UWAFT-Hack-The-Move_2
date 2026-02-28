"""Run validation metrics for a trained traffic light YOLO detector."""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO traffic light detector")
    parser.add_argument("--weights", required=True, help="Path to trained model weights")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size")
    parser.add_argument("--batch", type=int, default=16, help="Validation batch size")
    parser.add_argument("--device", default="0", help="CUDA device id or cpu")
    return parser.parse_args()


def print_metrics(metrics: object) -> None:
    results_dict = getattr(metrics, "results_dict", None)
    if isinstance(results_dict, dict):
        print("Validation metrics")
        print("------------------")
        for key in sorted(results_dict.keys()):
            print(f"{key}: {results_dict[key]}")
        return

    box = getattr(metrics, "box", None)
    if box is not None:
        print("Validation metrics")
        print("------------------")
        for attr in ("map", "map50", "map75"):
            if hasattr(box, attr):
                print(f"{attr}: {getattr(box, attr)}")
        return

    print("Validation completed, but metrics format was not recognized for this Ultralytics version.")


def main() -> int:
    args = parse_args()

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )

    print_metrics(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
