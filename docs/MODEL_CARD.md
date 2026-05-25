# Model Card: Cardiac MRI Segmentation

## Dataset
Target dataset: Medical Segmentation Decathlon Task02_Heart. The repository expects local NIfTI image and label volumes and creates a deterministic 70/15/15 split with `scripts/prepare_data.py`.

## Preprocessing
Volumes are converted into 2D axial slices. Slices with less than 5% foreground are filtered out, intensities are clipped to the 1st and 99th percentile, and values are scaled to `[-1, 1]`.

## Model
The baseline model is a 2D U-Net. The main model is an Attention U-Net with attention gates applied to skip connections before decoder concatenation.

## Training
Planned main run: Adam, learning rate `1e-4`, cosine scheduler, Dice+BCE loss, seed `42`, and early stopping on validation Dice.

## Evaluation
Primary metric: Dice score. Secondary metric: IoU. Results must be produced from `scripts/evaluate.py` after training and written to `outputs/metrics/`.

## Limitations
This is a 2D slice model and does not use 3D context. It is a portfolio/research implementation, not a clinical device.

## Ethical Considerations
Do not commit patient-identifiable metadata. Use only licensed or public research data and report dataset splits clearly.
