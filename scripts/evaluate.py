"""Evaluate checkpoint on a data split.
Usage: python scripts/evaluate.py --checkpoint outputs/logs/best_model.pt --split test
"""
import argparse, torch, numpy as np


def dice(pred, target, thr=0.5, smooth=1.0):
    p = (pred > thr).float().view(-1)
    t = target.view(-1)
    return ((2.0 * (p * t).sum() + smooth) / (p.sum() + t.sum() + smooth)).item()


def main(ckpt, split):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from models.unet import UNet
    model = UNet()
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval().to(device)
    print(f"Loaded: {ckpt}  |  Split: {split}")
    # Plug in DataLoader, iterate, collect dice scores
    # scores = [dice(model(img.to(device)).cpu(), mask) for img, mask in loader]
    # print(f"Mean Dice: {np.mean(scores):.4f}")
    print("Plug in DataLoader to compute metrics.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test")
    a = p.parse_args()
    main(a.checkpoint, a.split)
