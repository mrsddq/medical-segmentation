from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data.dataset import HeartDataset
from models import AttentionUNet, UNet
from scripts.utils import load_config


def build_model(cfg: dict) -> torch.nn.Module:
    model_cfg = dict(cfg["model"])
    model_type = model_cfg.pop("type", "unet")
    return AttentionUNet(**model_cfg) if model_type == "attention_unet" else UNet(**model_cfg)


def main(inp: str, out_dir: str, checkpoint: str, config: str) -> Path:
    input_path = Path(inp)
    if not input_path.exists():
        raise FileNotFoundError(f"Input scan not found: {input_path}")
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    nib = _require_nibabel()
    volume = np.asarray(nib.load(str(input_path)).dataobj, dtype=np.float32)
    predictions = []
    with torch.no_grad():
        for slice_index in range(volume.shape[2]):
            image = HeartDataset._normalize(volume[:, :, slice_index])
            image = _resize_image(image, int(cfg["data"].get("image_size", 512)))
            tensor = torch.from_numpy(image[None, None]).float().to(device)
            pred = model(tensor).squeeze().cpu().numpy()
            predictions.append((pred > 0.5).astype(np.uint8))
    mask = np.stack(predictions, axis=-1)
    output_path = output_dir / f"{input_path.stem}_mask.npy"
    np.save(output_path, mask)
    return output_path


def _require_nibabel():
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("nibabel is required for NIfTI inference.") from exc
    return nib


def _resize_image(image: np.ndarray, image_size: int) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required for inference resizing.") from exc
    return cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR).astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run segmentation inference on one NIfTI volume.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/predictions/")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/unet.yaml")
    args = parser.parse_args()
    print(main(args.input, args.output, args.checkpoint, args.config))
