# Portfolio Evidence Plan

This project should be shown as a medical image segmentation engineering project. Avoid medical-performance claims until a public dataset run is documented with reproducible metrics.

## Reproducible Demo

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_portfolio_contract.py"
python -m scripts.train --config configs/unet.yaml
python -m scripts.evaluate --checkpoint outputs/logs/best_model.pt --data data/processed
python -m scripts.infer --checkpoint outputs/logs/best_model.pt --input data/samples --output outputs/predictions
```

## Evidence To Capture

| Artifact | Portfolio Use |
|---|---|
| `assets/sample-input.png` | Shows the image type used in the demo. |
| `assets/sample-mask.png` | Shows the ground-truth mask format. |
| `assets/sample-prediction.png` | Shows model output and thresholding behavior. |
| `outputs/metrics/evaluation.csv` | Records Dice, IoU, precision, recall, and dataset split. |
| `docs/RESULTS.md` | Summarizes only verified runs. |

## Demo Narrative

1. Explain the U-Net baseline and why segmentation quality is measured with Dice/IoU.
2. Show config-driven training with `configs/unet.yaml`.
3. Run inference on one sample and compare input, mask, and prediction.
4. Discuss failure cases such as fuzzy boundaries, class imbalance, and small structures.

## Evidence Checklist Before Pinning

- [ ] Public or synthetic sample dataset identified.
- [ ] One input/mask/prediction triptych in `assets/`.
- [ ] Real metric table added to `docs/RESULTS.md`.
- [ ] CI badge green on the latest commit.
- [ ] Model limitations documented in `docs/MODEL_CARD.md`.
