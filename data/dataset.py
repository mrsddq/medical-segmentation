from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SliceRecord:
    image_path: Path
    label_path: Path
    slice_index: int


class HeartDataset(Dataset):
    """2D axial-slice dataset for Medical Segmentation Decathlon Task02_Heart."""

    def __init__(
        self,
        split_file: str | Path,
        image_size: int = 512,
        min_foreground_fraction: float = 0.05,
        augment: bool = False,
    ) -> None:
        self.split_file = Path(split_file)
        self.image_size = image_size
        self.min_foreground_fraction = min_foreground_fraction
        self.augment = augment
        self.records = self._index_records()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        nib = _require_nibabel()
        record = self.records[index]
        image = np.asarray(nib.load(str(record.image_path)).dataobj, dtype=np.float32)
        mask = np.asarray(nib.load(str(record.label_path)).dataobj, dtype=np.float32)
        image_slice = self._normalize(image[:, :, record.slice_index])
        mask_slice = (mask[:, :, record.slice_index] > 0).astype(np.float32)

        image_slice, mask_slice = self._resize(image_slice, mask_slice)
        image_slice, mask_slice = self._augment(image_slice, mask_slice)
        return {
            "image": torch.from_numpy(image_slice[None]).float(),
            "mask": torch.from_numpy(mask_slice[None]).float(),
        }

    def _index_records(self) -> list[SliceRecord]:
        if not self.split_file.exists():
            raise FileNotFoundError(
                f"Missing split file {self.split_file}. Run scripts/prepare_data.py first."
            )
        records: list[SliceRecord] = []
        nib = _require_nibabel()
        for line in self.split_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            image_path, label_path = [Path(part) for part in line.split(",")]
            mask = np.asarray(nib.load(str(label_path)).dataobj)
            for slice_index in range(mask.shape[2]):
                foreground = float((mask[:, :, slice_index] > 0).mean())
                if foreground >= self.min_foreground_fraction:
                    records.append(SliceRecord(image_path, label_path, slice_index))
        return records

    @staticmethod
    def _normalize(image: np.ndarray) -> np.ndarray:
        low, high = np.percentile(image, [1, 99])
        clipped = np.clip(image, low, high)
        denom = float(clipped.max() - clipped.min())
        if denom == 0.0:
            return np.zeros_like(clipped, dtype=np.float32)
        return ((clipped - clipped.min()) / denom * 2.0 - 1.0).astype(np.float32)

    def _resize(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("opencv-python is required for resizing slices.") from exc
        size = (self.image_size, self.image_size)
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        return image.astype(np.float32), mask.astype(np.float32)

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.augment:
            return image, mask
        try:
            import albumentations as A
        except ImportError as exc:
            raise ImportError("albumentations is required when augmentation is enabled.") from exc
        transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.ElasticTransform(alpha=20, sigma=5, p=0.2),
            ]
        )
        augmented = transform(image=image, mask=mask)
        return augmented["image"].astype(np.float32), augmented["mask"].astype(np.float32)


def _require_nibabel():
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("nibabel is required for NIfTI loading.") from exc
    return nib
