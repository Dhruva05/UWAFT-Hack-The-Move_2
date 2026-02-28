"""Validate YOLO-format traffic light detection dataset integrity."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List

from utils import IMAGE_SUFFIXES, is_class_id_valid, list_image_files, stem_set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate YOLO-format traffic light dataset")
    parser.add_argument(
        "--dataset_root",
        type=Path,
        required=True,
        help="Dataset root containing images/, labels/, and data.yaml",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Dataset splits to validate (default: train val)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any validation error is found",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=20,
        help="Max example errors to print per category",
    )
    return parser.parse_args()


def _validate_label_line(
    line: str,
    line_num: int,
    label_path: Path,
    errors: DefaultDict[str, List[str]],
) -> None:
    tokens = line.split()
    if len(tokens) != 5:
        errors["malformed_rows"].append(
            f"{label_path}:{line_num} expected 5 values, found {len(tokens)}"
        )
        return

    cls_raw, x_raw, y_raw, w_raw, h_raw = tokens

    try:
        cls = int(cls_raw)
    except ValueError:
        errors["invalid_class_id"].append(f"{label_path}:{line_num} class '{cls_raw}' is not an int")
        return

    if not is_class_id_valid(cls):
        errors["invalid_class_id"].append(
            f"{label_path}:{line_num} class {cls} is outside expected range [0,2]"
        )

    try:
        x, y, w, h = (float(x_raw), float(y_raw), float(w_raw), float(h_raw))
    except ValueError:
        errors["malformed_rows"].append(
            f"{label_path}:{line_num} one or more bbox values are not floats"
        )
        return

    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
        errors["malformed_boxes"].append(
            f"{label_path}:{line_num} normalized values must be in [0,1], got {(x, y, w, h)}"
        )

    if w <= 0.0 or h <= 0.0:
        errors["malformed_boxes"].append(
            f"{label_path}:{line_num} width/height must be > 0, got {(w, h)}"
        )

    # YOLO x/y are centers, so extents must also stay within [0,1].
    if x - (w / 2.0) < 0.0 or x + (w / 2.0) > 1.0 or y - (h / 2.0) < 0.0 or y + (h / 2.0) > 1.0:
        errors["malformed_boxes"].append(
            f"{label_path}:{line_num} box extents exceed image bounds, got {(x, y, w, h)}"
        )


def validate_split(dataset_root: Path, split: str) -> Dict[str, List[str]]:
    errors: DefaultDict[str, List[str]] = defaultdict(list)

    img_dir = dataset_root / "images" / split
    lbl_dir = dataset_root / "labels" / split

    if not img_dir.exists():
        errors["missing_directories"].append(f"Missing image directory: {img_dir}")
        return errors
    if not lbl_dir.exists():
        errors["missing_directories"].append(f"Missing label directory: {lbl_dir}")
        return errors

    image_files = list_image_files(img_dir)
    label_files = sorted(p for p in lbl_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt")

    image_stems = stem_set(image_files)
    label_stems = stem_set(label_files)

    missing_labels = sorted(image_stems - label_stems)
    missing_images = sorted(label_stems - image_stems)

    for stem in missing_labels:
        errors["missing_labels"].append(f"{split}: image '{stem}' has no corresponding label file")

    for stem in missing_images:
        errors["missing_images"].append(f"{split}: label '{stem}.txt' has no corresponding image")

    for label_path in label_files:
        if label_path.stem not in image_stems:
            continue

        lines = label_path.read_text(encoding="utf-8").splitlines()
        for idx, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line:
                continue
            _validate_label_line(line=line, line_num=idx, label_path=label_path, errors=errors)

    return errors


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()

    if not dataset_root.exists():
        print(f"ERROR: dataset_root does not exist: {dataset_root}")
        return 2

    yaml_path = dataset_root / "data.yaml"
    if not yaml_path.exists():
        print(f"WARNING: data.yaml not found at {yaml_path}; continuing validation of files only")

    all_errors: DefaultDict[str, List[str]] = defaultdict(list)

    print(f"Validating dataset at: {dataset_root}")
    print(f"Expected image extensions: {sorted(IMAGE_SUFFIXES)}")

    for split in args.splits:
        split_errors = validate_split(dataset_root, split)
        for key, values in split_errors.items():
            all_errors[key].extend(values)

    total_errors = sum(len(v) for v in all_errors.values())
    categories = sorted(all_errors.keys())

    print("\nValidation summary")
    print("------------------")
    print(f"Splits checked: {', '.join(args.splits)}")
    print(f"Error categories: {len(categories)}")
    print(f"Total errors: {total_errors}")

    if categories:
        print("\nError details")
        print("-------------")
        for category in categories:
            items = all_errors[category]
            print(f"{category}: {len(items)}")
            for example in items[: args.max_examples]:
                print(f"  - {example}")
            if len(items) > args.max_examples:
                print(f"  - ... {len(items) - args.max_examples} more")

    if total_errors == 0:
        print("\nPASS: dataset structure and labels look valid.")
        return 0

    print("\nFAIL: dataset validation found issues.")
    if args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
