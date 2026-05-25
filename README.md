# Medical Image Segmentation with U-Net

[![CI](https://github.com/mrsddq/medical-segmentation/actions/workflows/ci.yml/badge.svg)](https://github.com/mrsddq/medical-segmentation/actions/workflows/ci.yml)

Portfolio-ready PyTorch U-Net project for medical image segmentation experiments.

The repository provides the model, configuration, reusable metrics, and script skeletons needed to train and evaluate on private or public medical imaging datasets. It does not include clinical data, model weights, or unverified metrics.

## Highlights

- PyTorch U-Net with encoder-decoder skip connections
- Dice and Dice+BCE metric/loss helpers
- YAML experiment configuration
- Training, evaluation, and inference entry points
- Unit tests for model shape, config loading, and metrics
- Results template for reproducible experiment reporting

## Architecture

```text
Input image
  -> encoder blocks with max pooling
  -> bottleneck
  -> decoder blocks with transpose convolutions
  -> skip concatenation from matching encoder stages
  -> 1x1 segmentation head
  -> sigmoid mask probability
```

## Structure

```text
configs/
  unet.yaml
docs/
  ABLATION_PLAN.md
  ARCHITECTURE_RATIONALE.md
  DEPLOYMENT_NOTES.md
  REPRODUCIBILITY.md
  RESULTS_TEMPLATE.md
models/
  unet.py
scripts/
  train.py
  evaluate.py
  infer.py
  metrics.py
  utils.py
tests/
  test_unet.py
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data Layout

Clinical data is not committed. Use a private dataset or a public dataset such as:

- Medical Segmentation Decathlon
- CHAOS Challenge
- KiTS kidney tumor segmentation dataset

Recommended local layout:

```text
data/
  raw/
  processed/
  splits/
    train.txt
    val.txt
    test.txt
```

## Train

```bash
python -m scripts.train --config configs/unet.yaml
```

The current script initializes the model, optimizer, scheduler, and checkpoint directory. Connect your dataset class and DataLoader before running full training.

## Evaluate

```bash
python -m scripts.evaluate --checkpoint outputs/logs/best_model.pt --split test
```

## Inference

```bash
python -m scripts.infer --input data/processed/case_001.nii.gz --checkpoint outputs/logs/best_model.pt
```

## Testing

```bash
pytest
```

## Results

No verified public metrics are committed yet. After training, record results in [docs/RESULTS_TEMPLATE.md](docs/RESULTS_TEMPLATE.md) and add de-identified sample overlays under `assets/`.

Research support docs:

- [Reproducibility Plan](docs/REPRODUCIBILITY.md)
- [Architecture Rationale](docs/ARCHITECTURE_RATIONALE.md)
- [Ablation Plan](docs/ABLATION_PLAN.md)
- [Deployment Notes](docs/DEPLOYMENT_NOTES.md)

`outputs/metrics/smoke_test_results.csv` is a schema artifact only, not a benchmark.

Recommended artifacts:

- `assets/input-slice.png`
- `assets/ground-truth-overlay.png`
- `assets/prediction-overlay.png`
- `assets/training-curve.png`
- `assets/failure-case.png`

## Limitations

- Dataset and weights are not included.
- Script entry points are designed for extension with a project-specific dataset class.
- 2D slice-level U-Net does not capture full 3D context.
- Any metric should be treated as dataset-specific until externally validated.
