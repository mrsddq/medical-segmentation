from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import HeartDataset
from models import AttentionUNet, UNet
from scripts.metrics import dice_score
from scripts.utils import load_config


def build_model(cfg: dict) -> torch.nn.Module:
    model_cfg = dict(cfg["model"])
    model_type = model_cfg.pop("type", "unet")
    return AttentionUNet(**model_cfg) if model_type == "attention_unet" else UNet(**model_cfg)


def main(checkpoint: str, split: str, config: str, output: str) -> Path:
    cfg = load_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    split_file = Path(cfg["data"].get("splits_dir", "data/splits")) / f"{split}.txt"
    dataset = HeartDataset(split_file=split_file, image_size=int(cfg["data"].get("image_size", 512)))
    loader = DataLoader(dataset, batch_size=int(cfg["training"]["batch_size"]), shuffle=False)

    dice_values = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            preds = model(images)
            dice_values.append(float(dice_score(preds.cpu(), masks.cpu())))

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mean_dice = float(np.mean(dice_values)) if dice_values else 0.0
    mean_iou = mean_dice / (2.0 - mean_dice) if mean_dice < 1.0 else 1.0
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "samples", "dice", "iou"])
        writer.writeheader()
        writer.writerow({"split": split, "samples": len(dataset), "dice": mean_dice, "iou": mean_iou})
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a medical segmentation checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--config", default="configs/unet.yaml")
    parser.add_argument("--output", default="outputs/metrics/test_results.csv")
    args = parser.parse_args()
    print(main(args.checkpoint, args.split, args.config, args.output))
