from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data.dataset import HeartDataset
from scripts.train import build_model
from scripts.utils import load_config


def main(inp: str, out_dir: str, checkpoint: str, config: str) -> Path:
    input_path = Path(inp)
    if not input_path.exists():
        raise FileNotFoundError(f"Input scan not found: {input_path}")
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    nib = _require_nibabel()
    source = nib.load(str(input_path))
    volume = np.asarray(source.dataobj, dtype=np.float32)
    if volume.ndim != 3 or not np.isfinite(volume).all():
        raise ValueError("Input must be a finite 3D NIfTI volume")
    predictions = []
    with torch.no_grad():
        for slice_index in range(volume.shape[2]):
            image = HeartDataset._normalize(volume[:, :, slice_index])
            image = _resize_image(image, int(cfg["data"].get("image_size", 512)))
            tensor = torch.from_numpy(image[None, None]).float().to(device)
            pred = model(tensor).squeeze().cpu().numpy()
            import cv2
            # Restore probabilities before thresholding; retain original voxel geometry.
            pred = cv2.resize(pred, (volume.shape[1], volume.shape[0]), interpolation=cv2.INTER_LINEAR)
            predictions.append((pred > 0.5).astype(np.uint8))
    mask = np.stack(predictions, axis=-1)
    name = input_path.name.removesuffix(".gz").removesuffix(".nii")
    output_path = output_dir / f"{name}_mask.nii.gz"
    header = source.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(mask, source.affine, header), str(output_path))
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
