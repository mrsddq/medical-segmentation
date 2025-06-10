"""Train U-Net on medical segmentation data.
Usage: python scripts/train.py --config configs/unet.yaml
"""
import argparse, yaml, torch
from pathlib import Path


def dice_loss(pred, target, smooth=1.0):
    p, t = pred.view(-1), target.view(-1)
    return 1 - (2.0 * (p * t).sum() + smooth) / (p.sum() + t.sum() + smooth)


def combined_loss(pred, target, dw=0.5, bw=0.5):
    return bw * torch.nn.functional.binary_cross_entropy(pred, target) + dw * dice_loss(pred, target)


def main(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    from models.unet import UNet
    model = UNet(**cfg["model"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["training"]["epochs"])
    ckpt_dir = Path(cfg["logging"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Plug in your DataLoader here:
    # from data.dataset import MedicalSegDataset
    # train_loader = DataLoader(MedicalSegDataset("train", cfg["data"]), ...)
    print("Plug in DataLoader to begin training.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/unet.yaml")
    main(p.parse_args().config)
