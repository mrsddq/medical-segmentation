# Ablation Plan

| Experiment | Variable | Fixed Controls | Metric |
|---|---|---|---|
| baseline | vanilla U-Net | split, preprocessing | Dice |
| loss | Dice vs Dice+BCE | model, split | Dice, recall |
| augmentation | on/off | model, split | Dice and failure cases |
| encoder width | feature list | split, loss | Dice vs memory |
| 2D vs 3D | architecture | dataset version | Dice, inference cost |

Every ablation should include config, seed, hardware, metrics CSV, and sample overlays.
