"""Train U-Net on medical segmentation data."""
import argparse
from pathlib import Path

import torch

from scripts.utils import load_config


def main(cfg_path):
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from models.unet import UNet

    model = UNet(**cfg["model"]).to(device)
    checkpoint_dir = Path(cfg["logging"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["training"]["epochs"])

    print("Model initialized successfully.")
    print("Next step: connect a dataset class and DataLoader for your preprocessed scans.")
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    return model, optimizer, scheduler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unet.yaml")
    args = parser.parse_args()
    main(args.config)
