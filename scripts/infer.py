"""Run inference on a single medical image volume."""
import argparse
from pathlib import Path

import torch


def main(inp, out_dir, checkpoint=None):
    input_path = Path(inp)
    if not input_path.exists():
        raise FileNotFoundError(f"Input scan not found: {input_path}")

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from models.unet import UNet

    model = UNet()
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval().to(device)
    print(f"Input: {input_path} -> Output: {output_dir}")
    print("Next step: load the scan, preprocess slices, run prediction, and save mask overlays.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/predictions/")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    main(args.input, args.output, args.checkpoint)
