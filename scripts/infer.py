"""Run inference on a single NIfTI scan.
Usage: python scripts/infer.py --input scan.nii.gz --output outputs/predictions/
"""
import argparse, torch
from pathlib import Path


def main(inp, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from models.unet import UNet
    model = UNet()
    # model.load_state_dict(torch.load("outputs/logs/best_model.pt", map_location=device))
    model.eval().to(device)
    print(f"Input: {inp}  →  Output: {out_dir}")
    # import nibabel as nib
    # vol = nib.load(inp).get_fdata()  # preprocess, slice, predict, save


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="outputs/predictions/")
    a = p.parse_args()
    main(a.input, a.output)
