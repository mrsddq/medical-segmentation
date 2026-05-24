"""Evaluate a U-Net checkpoint on a data split."""
import argparse
from pathlib import Path

import torch


def main(ckpt, split):
    checkpoint = Path(ckpt)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from models.unet import UNet

    model = UNet()
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval().to(device)
    print(f"Loaded: {checkpoint} | Split: {split}")
    print("Next step: connect the split DataLoader and aggregate Dice/IoU per case.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    main(args.checkpoint, args.split)
