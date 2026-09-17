# Medical segmentation engineering runbook

This repository implements binary segmentation of paired 3D NIfTI volumes using
2D U-Net slices. Follow the [README setup](../README.md#setup),
[training](../README.md#train), [evaluation](../README.md#evaluate), and
[inference](../README.md#inference) commands for data-backed experiments.

## Local verification

Run from the repository root with Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-test.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

Dependency installation needs package-network access. Once installed, the test
suite runs on CPU with generated fixtures and does not download model weights or
datasets. On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell
and run `python -m pytest -q`.

Tests execute a small U-Net, synthetic NIfTI inference/evaluation, image/mask
geometry checks, background-slice inclusion, and metric validation. These are
software checks, not clinical performance measurements.

## Data and artifact contract

- Prepare case-level `image_path,label_path` split files; paths are relative to the
  working directory. Train/validation case paths must not overlap. Paired volumes
  need identical shape and affine, and evaluation retains background slices.
- Keep clinical data and trained weights outside version control. Only explicitly
  de-identified example artifacts belong in a public repository.
- Training writes `best.pt` under the configured checkpoint directory. Retain the
  exact configuration and data-split provenance with it. The trainer currently uses
  fixed Adam/cosine/Dice+BCE choices; per-transform augmentation knobs are not all
  independently honored, as explained in the README.
- Inference writes a binary NIfTI mask at the source volume's shape and affine.
  Evaluation reports mean per-slice Dice and IoU, not patient-level metrics.
- Nonfinite probabilities, nonbinary targets, or mismatched geometry are errors.
  Fix the underlying checkpoint/data issue instead of replacing invalid values
  with background or omitting failed samples.

No clinical deployment or externally validated accuracy is claimed.
