"""Shared utilities for traffic light training and inference scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

CLASS_NAMES: List[str] = ["redlight", "yellowlight", "greenlight"]
CLASS_ID_TO_NAME = {idx: name for idx, name in enumerate(CLASS_NAMES)}
CLASS_NAME_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_image_files(path: Path) -> List[Path]:
    """Return sorted image files under a directory."""
    return sorted(
        p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def ensure_dir(path: Path) -> Path:
    """Create directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_roi(roi_text: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    """Parse normalized ROI string x1,y1,x2,y2 into a validated tuple."""
    if not roi_text:
        return None

    parts = [p.strip() for p in roi_text.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must have exactly 4 comma-separated values: x1,y1,x2,y2")

    x1, y1, x2, y2 = (float(p) for p in parts)
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError("ROI values must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1")

    return x1, y1, x2, y2


def roi_norm_to_pixels(
    roi: Tuple[float, float, float, float],
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    """Convert normalized ROI to pixel ROI."""
    x1n, y1n, x2n, y2n = roi
    x1 = int(x1n * width)
    y1 = int(y1n * height)
    x2 = int(x2n * width)
    y2 = int(y2n * height)
    return x1, y1, x2, y2


def resolve_model_candidates(primary: str) -> List[str]:
    """Build a model candidate list with sensible YOLOv8 filename fallbacks."""
    candidates: List[str] = [primary]
    for fallback in ("yolo8n.pt", "yolov8n.pt"):
        if fallback not in candidates:
            candidates.append(fallback)
    return candidates


def pretty_bbox(x1: float, y1: float, x2: float, y2: float) -> str:
    """Format bbox for readable logs."""
    return f"[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"


def is_class_id_valid(class_id: int) -> bool:
    """Check class id against known classes."""
    return 0 <= class_id < len(CLASS_NAMES)


def stem_set(paths: Sequence[Path]) -> set[str]:
    """Return set of file stems for path sequence."""
    return {p.stem for p in paths}


def chunked(iterable: Iterable[str], size: int) -> Iterable[list[str]]:
    """Yield fixed-size chunks from an iterable."""
    batch: list[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
