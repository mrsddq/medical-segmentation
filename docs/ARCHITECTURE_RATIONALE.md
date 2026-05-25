# Architecture Rationale

## Why U-Net

U-Net remains a strong segmentation baseline for small-to-medium medical imaging datasets because skip connections preserve spatial detail while the encoder learns semantic context.

## Current Scope

- 2D slice-level segmentation
- binary mask output
- Dice/BCE-style optimization

## Upgrade Path

- add `models/unet3d.py` for volumetric context
- add Attention U-Net gates for organ-boundary focus
- add albumentations data augmentation in the dataset loader
- add MLflow or W&B experiment tracking

## Clinical Caution

This repo is research engineering only. Any medical claim requires validated datasets, clinical review, and external reproducibility.
