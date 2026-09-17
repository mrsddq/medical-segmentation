# Medical Image Segmentation with U-Net

[![CI](https://github.com/mrsddq/medical-segmentation/actions/workflows/ci.yml/badge.svg)](https://github.com/mrsddq/medical-segmentation/actions/workflows/ci.yml)

Portfolio-ready PyTorch U-Net project for medical image segmentation experiments.

The repository provides the model, configuration, reusable metrics, and implemented training, evaluation, and inference needed to experiment on private or public medical imaging datasets. It does not include clinical data, model weights, or unverified metrics.

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

The script loads paired NIfTI slices, trains with validation-based checkpoint selection, and rejects training/validation case-path overlap. Prepare split files using `scripts.prepare_data` before training.

## Evaluate

```bash
python -m scripts.evaluate --checkpoint outputs/logs/best.pt --split test
```

## Inference

```bash
python -m scripts.infer --input data/processed/case_001.nii.gz --checkpoint outputs/logs/best.pt
```

## Testing

```bash
pytest
```

## Results

No verified public metrics are committed yet. After training, record results in [docs/RESULTS_TEMPLATE.md](docs/RESULTS_TEMPLATE.md) and add de-identified sample overlays under `assets/`.

Research support docs:

- [Portfolio Evidence Plan](docs/PORTFOLIO_EVIDENCE.md)
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
- The supported workflow is binary segmentation of paired 3D NIfTI volumes using 2D slices.
- 2D slice-level U-Net does not capture full 3D context.
- Any metric should be treated as dataset-specific until externally validated.

## Correctness gates and offline verification

Install `requirements-test.txt`, then run `python -m pytest -q`. Synthetic NIfTI
tests exercise real model inference, geometry preservation, background inclusion,
metric aggregation, and malformed input handling without clinical data or weights.
Inference writes `<case>_mask.nii.gz` at the **original volume shape and affine**;
resizing is only an internal model step. Evaluation computes binary Dice and IoU
per slice and averages all slices equally; IoU is not inferred from mean Dice.
An empty prediction/target pair scores 1. Patient/volume-level scores are not yet
reported and may differ from these slice averages.

Validation and test retain background-only slices. Training may opt into
`data.train_min_foreground_fraction`; its default is zero. Paired volumes must
have matching shape and affine. Split files contain `image_path,label_path` rows;
paths are interpreted relative to the working directory. Keep case-level splits
and use de-identified data. No clinical validation or deployment claim is made.
