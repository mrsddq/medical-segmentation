from __future__ import annotations

import argparse
import random
from pathlib import Path


def create_splits(images_dir: Path, labels_dir: Path, output_dir: Path, seed: int = 42) -> None:
    pairs = []
    for image_path in sorted(images_dir.glob("*.nii*")):
        label_path = labels_dir / image_path.name
        if label_path.exists():
            pairs.append((image_path, label_path))
    if not pairs:
        raise FileNotFoundError("No matching image/label NIfTI pairs found.")
    random.Random(seed).shuffle(pairs)
    n_total = len(pairs)
    n_train = max(1, round(n_total * 0.70))
    n_val = max(1, round(n_total * 0.15))
    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train : n_train + n_val],
        "test": pairs[n_train + n_val :],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, split_pairs in splits.items():
        lines = [f"{image},{label}" for image, label in split_pairs]
        (output_dir / f"{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Task02_Heart train/val/test split files.")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_splits(args.images_dir, args.labels_dir, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
