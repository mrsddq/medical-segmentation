# Medical Image Segmentation with U-Net

Semantic segmentation of CT and MRI scans using a PyTorch U-Net with skip connections. Trained and evaluated on volumetric medical imaging data.

## Results

| Metric | Value |
|---|---|
| Dice Coefficient | 0.87 |
| Training Loss Reduction | ~30% |
| IoU | _add after re-run_ |
| Precision | _add after re-run_ |
| Recall | _add after re-run_ |

> Evaluation performed on held-out test split. Dataset kept private — see Data section.

## Architecture

```
Input (1×512×512)
  └─ Encoder (4× downsample blocks: Conv→BN→ReLU×2 + MaxPool)
       └─ Bottleneck (1024 channels)
            └─ Decoder (4× upsample blocks: ConvTranspose + skip concat + Conv×2)
                 └─ Output head (1×1 Conv → sigmoid)
```

Skip connections bridge each encoder stage to the corresponding decoder stage, preserving spatial detail critical for boundary accuracy.

## Quickstart

```bash
git clone https://github.com/your-username/medical-segmentation
cd medical-segmentation
pip install -r requirements.txt
```

## Data

Dataset is private clinical CT/MRI data and cannot be redistributed. To reproduce with public data, use one of:

- [Medical Segmentation Decathlon](http://medicaldecathlon.com/) — Task03 Liver or Task09 Spleen
- [CHAOS Challenge](https://chaos.grand-challenge.org/) — abdominal MRI

Place data as:
```
data/
  raw/          ← original NIfTI/DICOM files
  processed/    ← normalised, resampled volumes
  splits/
    train.txt
    val.txt
    test.txt
```

## Training

```bash
python scripts/train.py --config configs/unet.yaml
```

Key config options in `configs/unet.yaml`:

```yaml
model:
  in_channels: 1
  out_channels: 1
  features: [64, 128, 256, 512]

training:
  epochs: 100
  batch_size: 8
  lr: 1e-4
  optimizer: adam
  loss: dice_bce

data:
  image_size: 512
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
```

## Evaluation

```bash
python scripts/evaluate.py --checkpoint outputs/logs/best_model.pt --split test
```

Outputs a per-case Dice table and saves overlay PNGs to `outputs/predictions/`.

## Inference

```bash
python scripts/infer.py --input path/to/scan.nii.gz --output outputs/predictions/
```

## Sample Outputs

_Add screenshots to `assets/` — see naming convention below._

| File | Contents |
|---|---|
| `assets/01_input_scan.png` | Raw CT/MRI slice |
| `assets/02_ground_truth.png` | GT mask overlay |
| `assets/03_prediction.png` | Model prediction overlay |
| `assets/04_comparison.png` | GT vs predicted side-by-side |
| `assets/05_error_case.png` | Failure case with annotation |
| `assets/06_training_curve.png` | Loss + Dice vs epoch |

## Limitations

- Model trained on a single private dataset — generalisation to out-of-distribution scanners not verified
- 2D slice-level inference; 3D context not exploited
- Dice coefficient reported at dataset level; per-class breakdown pending

## Environment

```
Python 3.10
PyTorch 2.1
CUDA 11.8
```

Full dependency list in `requirements.txt`.
