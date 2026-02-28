# Traffic Light Detection Pipeline (Colab + Ultralytics YOLO)

Single-model object detector for traffic lights where class encodes state:

- `redlight`
- `yellowlight`
- `greenlight`

The detector outputs bounding boxes, class labels, and confidence scores in one pass.

## Repository Layout

- `traffic_light_colab_train_smoketest.ipynb`: Colab-ready notebook
- `config/data.yaml.example`: dataset YAML template
- `src/validate_dataset.py`: dataset integrity checks
- `src/train.py`: training entrypoint
- `src/evaluate.py`: validation metrics entrypoint
- `src/infer.py`: image/folder/video inference + annotated outputs
- `src/state_smoothing.py`: temporal smoothing (`NONE/RED/YELLOW/GREEN`)

## Dataset Format

Expected structure:

```text
traffic_lights/
  images/
    train/
    val/
  labels/
    train/
    val/
  data.yaml
```

Class mapping:

- `0 = redlight`
- `1 = yellowlight`
- `2 = greenlight`

## Local Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Validate dataset:

```bash
python src/validate_dataset.py --dataset_root /path/to/traffic_lights --strict
```

Train:

```bash
python src/train.py \
  --data /path/to/traffic_lights/data.yaml \
  --model yolo8n.pt \
  --imgsz 640 \
  --batch 16 \
  --epochs 50 \
  --project runs/tl \
  --name yolo8n_traffic_lights
```

Evaluate:

```bash
python src/evaluate.py \
  --weights runs/tl/yolo8n_traffic_lights/weights/best.pt \
  --data /path/to/traffic_lights/data.yaml
```

Infer (best detection only, with optional ROI):

```bash
python src/infer.py \
  --weights runs/tl/yolo8n_traffic_lights/weights/best.pt \
  --source /path/to/images_or_video \
  --conf 0.35 \
  --best_only \
  --roi 0,0,1,0.6 \
  --save_dir outputs
```

## Colab Usage

1. Upload this repo to GitHub.
2. Open `traffic_light_colab_train_smoketest.ipynb` in Colab.
3. Enable GPU runtime.
4. In the Kaggle setup cell, set `DATASET_SLUG` to your dataset.
5. Upload `kaggle.json` when prompted.
6. Run all cells top-to-bottom.
7. Smoke test cell performs dataset validation + 1-epoch CUDA training.

The notebook now downloads and extracts the dataset directly in Colab and auto-detects a YOLO dataset root with:

- `images/train`
- `images/val`
- `labels/train`
- `labels/val`

## Runtime Integration Note

This model is vision-only. Downstream control logic should consume output and decide stop/go separately:

- `RED` -> stop candidate
- `YELLOW` -> caution
- `GREEN` -> continue

Combine with stop-line/map distance logic outside this detector.
