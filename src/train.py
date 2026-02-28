"""Train a YOLO traffic light detector with class-state multitask encoding."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from utils import resolve_model_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train traffic light detector with Ultralytics YOLO")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--model", default="yolo8n.pt", help="Model checkpoint or architecture file")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--device", default="0", help="CUDA device id or cpu")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--project", default="runs/tl", help="Output project directory")
    parser.add_argument("--name", default="yolo8n_traffic_lights", help="Run name")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    return parser.parse_args()


def load_model_with_fallback(preferred: str) -> tuple[YOLO, str]:
    candidates = resolve_model_candidates(preferred)
    last_error: Exception | None = None

    for candidate in candidates:
        try:
            return YOLO(candidate), candidate
        except Exception as exc:  # pylint: disable=broad-except
            last_error = exc
            print(f"Model candidate failed: {candidate} -> {exc}")

    raise RuntimeError(
        f"Could not initialize YOLO model from any candidate: {candidates}"
    ) from last_error


def main() -> int:
    args = parse_args()

    model, selected = load_model_with_fallback(args.model)
    print(f"Using model: {selected}")

    results = model.train(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        epochs=args.epochs,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
    )

    save_dir = Path(getattr(results, "save_dir", Path(args.project) / args.name)).resolve()
    best_path = save_dir / "weights" / "best.pt"

    print("\nTraining complete")
    print(f"Run directory: {save_dir}")
    print(f"Best checkpoint: {best_path}")
    print(f"Best checkpoint exists: {best_path.exists()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
