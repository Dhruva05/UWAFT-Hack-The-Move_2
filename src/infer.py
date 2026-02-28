"""Inference pipeline for traffic light detection/state classification."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from ultralytics import YOLO

from state_smoothing import StableState, TemporalStateFilter
from utils import (
    CLASS_ID_TO_NAME,
    IMAGE_SUFFIXES,
    ensure_dir,
    list_image_files,
    parse_roi,
    pretty_bbox,
    roi_norm_to_pixels,
)

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO inference for traffic light states")
    parser.add_argument("--weights", required=True, help="Path to trained model weights")
    parser.add_argument("--source", required=True, help="Image file, image folder, or video path")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", default="0", help="CUDA device id or cpu")
    parser.add_argument(
        "--roi",
        default=None,
        help="Optional normalized ROI as x1,y1,x2,y2 (e.g. 0,0,1,0.6)",
    )
    parser.add_argument("--save_dir", default="outputs", help="Directory for annotated outputs")
    parser.add_argument("--smooth_window", type=int, default=5, help="Temporal smoothing window")
    parser.add_argument(
        "--min_consecutive",
        type=int,
        default=3,
        help="Consecutive frames needed to commit a stable state",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--best_only",
        action="store_true",
        help="Keep only highest-confidence traffic light detection",
    )
    mode_group.add_argument(
        "--all_detections",
        action="store_true",
        help="Keep all detections above threshold",
    )

    return parser.parse_args()


def _class_color(class_name: str) -> Tuple[int, int, int]:
    if class_name == "redlight":
        return (0, 0, 255)
    if class_name == "yellowlight":
        return (0, 255, 255)
    if class_name == "greenlight":
        return (0, 255, 0)
    return (255, 255, 255)


def run_detector_on_frame(
    model: YOLO,
    frame,
    conf: float,
    imgsz: int,
    device: str,
    roi_norm: Optional[Tuple[float, float, float, float]],
    best_only: bool,
) -> Tuple[List[Detection], Optional[Tuple[int, int, int, int]]]:
    height, width = frame.shape[:2]

    x_off = 0
    y_off = 0
    roi_pixels: Optional[Tuple[int, int, int, int]] = None
    infer_frame = frame

    if roi_norm is not None:
        x1, y1, x2, y2 = roi_norm_to_pixels(roi_norm, width, height)
        roi_pixels = (x1, y1, x2, y2)
        infer_frame = frame[y1:y2, x1:x2]
        x_off, y_off = x1, y1

        if infer_frame.size == 0:
            return [], roi_pixels

    results = model.predict(source=infer_frame, conf=conf, imgsz=imgsz, device=device, verbose=False)
    boxes = results[0].boxes

    detections: List[Detection] = []
    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls.item())
            class_name = CLASS_ID_TO_NAME.get(cls_id)
            if class_name is None:
                continue

            confidence = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=confidence,
                    bbox_xyxy=(x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off),
                )
            )

    detections.sort(key=lambda d: d.confidence, reverse=True)
    if best_only and detections:
        detections = [detections[0]]

    return detections, roi_pixels


def annotate_frame(
    frame,
    detections: List[Detection],
    stable_state: StableState,
    roi_pixels: Optional[Tuple[int, int, int, int]],
):
    annotated = frame.copy()

    if roi_pixels is not None:
        x1, y1, x2, y2 = roi_pixels
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 128, 0), 2)
        cv2.putText(
            annotated,
            "ROI",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 128, 0),
            2,
            cv2.LINE_AA,
        )

    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.bbox_xyxy)
        color = _class_color(det.class_name)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        annotated,
        f"Stable: {stable_state.value}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return annotated


def log_detections(prefix: str, detections: List[Detection], stable_state: StableState) -> None:
    if not detections:
        print(f"{prefix} no detections | stable={stable_state.value}")
        return

    for det in detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        print(
            f"{prefix} class={det.class_name} "
            f"conf={det.confidence:.3f} bbox={pretty_bbox(x1, y1, x2, y2)} "
            f"stable={stable_state.value}"
        )


def process_image(
    model: YOLO,
    image_path: Path,
    save_dir: Path,
    conf: float,
    imgsz: int,
    device: str,
    roi_norm: Optional[Tuple[float, float, float, float]],
    best_only: bool,
    smoother: TemporalStateFilter,
) -> None:
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"Skipping unreadable image: {image_path}")
        return

    detections, roi_pixels = run_detector_on_frame(model, frame, conf, imgsz, device, roi_norm, best_only)
    raw_class = detections[0].class_name if detections else None
    stable_state = smoother.update(raw_class)
    log_detections(f"[{image_path.name}]", detections, stable_state)

    annotated = annotate_frame(frame, detections, stable_state, roi_pixels)
    out_path = save_dir / image_path.name
    cv2.imwrite(str(out_path), annotated)


def process_folder(
    model: YOLO,
    folder_path: Path,
    save_dir: Path,
    conf: float,
    imgsz: int,
    device: str,
    roi_norm: Optional[Tuple[float, float, float, float]],
    best_only: bool,
    smoother: TemporalStateFilter,
) -> None:
    image_files = list_image_files(folder_path)
    if not image_files:
        print(f"No supported images found in folder: {folder_path}")
        return

    for image_path in image_files:
        process_image(
            model=model,
            image_path=image_path,
            save_dir=save_dir,
            conf=conf,
            imgsz=imgsz,
            device=device,
            roi_norm=roi_norm,
            best_only=best_only,
            smoother=smoother,
        )


def process_video(
    model: YOLO,
    video_path: Path,
    save_dir: Path,
    conf: float,
    imgsz: int,
    device: str,
    roi_norm: Optional[Tuple[float, float, float, float]],
    best_only: bool,
    smoother: TemporalStateFilter,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Unable to open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = save_dir / f"{video_path.stem}_annotated.mp4"
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections, roi_pixels = run_detector_on_frame(model, frame, conf, imgsz, device, roi_norm, best_only)
        raw_class = detections[0].class_name if detections else None
        stable_state = smoother.update(raw_class)
        log_detections(f"[frame {frame_idx}]", detections, stable_state)

        annotated = annotate_frame(frame, detections, stable_state, roi_pixels)
        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Saved annotated video: {out_path}")


def main() -> int:
    args = parse_args()

    best_only = args.best_only or not args.all_detections

    roi_norm = parse_roi(args.roi)
    save_dir = ensure_dir(Path(args.save_dir))

    model = YOLO(args.weights)
    source_path = Path(args.source)

    smoother = TemporalStateFilter(
        window_size=args.smooth_window,
        min_consecutive=args.min_consecutive,
    )

    if source_path.is_dir():
        process_folder(
            model=model,
            folder_path=source_path,
            save_dir=save_dir,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            roi_norm=roi_norm,
            best_only=best_only,
            smoother=smoother,
        )
    elif source_path.is_file():
        suffix = source_path.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            process_image(
                model=model,
                image_path=source_path,
                save_dir=save_dir,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                roi_norm=roi_norm,
                best_only=best_only,
                smoother=smoother,
            )
        elif suffix in VIDEO_SUFFIXES:
            process_video(
                model=model,
                video_path=source_path,
                save_dir=save_dir,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                roi_norm=roi_norm,
                best_only=best_only,
                smoother=smoother,
            )
        else:
            print(f"Unsupported file type: {source_path}")
            return 2
    else:
        print(f"Source does not exist: {source_path}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
