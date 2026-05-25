# Reproducibility Plan

## Environment

- Python: 3.10
- Dependencies: pinned in `requirements.txt`
- Config: `configs/unet.yaml`

## Dataset Contract

Clinical data is never committed. For every real run, record:

- dataset name and version
- task/anatomy
- number of subjects and slices
- train/val/test split method
- preprocessing steps
- de-identification confirmation
- checksum or DVC hash

## Run Order

1. Validate config and model shape with tests.
2. Build dataset-specific loaders.
3. Train with fixed seed and logged hardware.
4. Evaluate Dice, IoU, precision, recall, and failure cases.
5. Save de-identified overlays under `assets/`.

## Non-Benchmark Artifact

`outputs/metrics/smoke_test_results.csv` is a schema example only. It is not a clinical result.
