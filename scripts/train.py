from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import HeartDataset
from models import AttentionUNet, UNet
from scripts.metrics import dice_bce_loss, dice_score
from scripts.utils import load_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(cfg: dict) -> torch.nn.Module:
    model_cfg = dict(cfg["model"])
    model_type = model_cfg.pop("type", "unet")
    if model_type == "attention_unet":
        return AttentionUNet(**model_cfg)
    if model_type == "unet":
        return UNet(**model_cfg)
    raise ValueError(f"Unsupported model type: {model_type}")


def build_loader(cfg: dict, split: str, shuffle: bool) -> DataLoader:
    data_cfg = cfg["data"]
    split_file = Path(data_cfg.get("splits_dir", "data/splits")) / f"{split}.txt"
    augmentation = data_cfg.get("augmentation", {})
    augment = split == "train" and bool(augmentation.get("enabled", augmentation.get("horizontal_flip", False)))
    dataset = HeartDataset(
        split_file=split_file,
        image_size=int(data_cfg.get("image_size", 512)),
        augment=augment,
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(model, loader, optimizer, device, train: bool) -> float:
    model.train(train)
    scores = []
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        with torch.set_grad_enabled(train):
            preds = model(images)
            loss = dice_bce_loss(preds, masks)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        scores.append(float(dice_score(preds.detach().cpu(), masks.detach().cpu())))
    return float(np.mean(scores)) if scores else 0.0


def main(cfg_path: str) -> Path:
    cfg = load_config(cfg_path)
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    train_loader = build_loader(cfg, "train", shuffle=True)
    val_loader = build_loader(cfg, "val", shuffle=False)

    checkpoint_dir = Path(cfg["logging"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(cfg["logging"].get("metrics_path", "outputs/metrics/train_results.csv"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["lr"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(cfg["training"]["epochs"]))
    best_dice = -1.0
    patience = int(cfg["training"].get("early_stopping_patience", 15))
    stale_epochs = 0

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_dice", "val_dice"])
        writer.writeheader()
        for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
            train_dice = run_epoch(model, train_loader, optimizer, device, train=True)
            val_dice = run_epoch(model, val_loader, optimizer, device, train=False)
            scheduler.step()
            writer.writerow({"epoch": epoch, "train_dice": train_dice, "val_dice": val_dice})
            f.flush()
            if val_dice > best_dice:
                best_dice = val_dice
                stale_epochs = 0
                torch.save(model.state_dict(), checkpoint_dir / "best.pt")
            else:
                stale_epochs += 1
            if stale_epochs >= patience:
                break
    return checkpoint_dir / "best.pt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train medical segmentation model.")
    parser.add_argument("--config", default="configs/unet.yaml")
    args = parser.parse_args()
    print(main(args.config))
