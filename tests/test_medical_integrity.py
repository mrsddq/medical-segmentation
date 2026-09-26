import numpy as np
import nibabel as nib
import pytest
import torch
import yaml
from data.dataset import HeartDataset
from models import UNet
from scripts.metrics import per_sample_overlap
from scripts.infer import main as infer
from scripts.evaluate import main as evaluate


def test_metrics_are_sample_weighted_and_iou_is_computed_directly():
    prediction = torch.tensor([[[[1., 0.]]], [[[1., 1.]]]])
    target = torch.tensor([[[[1., 0.]]], [[[1., 0.]]]])
    dice, iou = per_sample_overlap(prediction, target)
    assert dice.mean().item() == pytest.approx(5 / 6)
    assert iou.mean().item() == pytest.approx(3 / 4)
    assert iou.mean().item() != pytest.approx(dice.mean().item() / (2 - dice.mean().item()))
    empty, _ = per_sample_overlap(torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2))
    assert empty.item() == 1


def setup_case(tmp_path):
    affine = np.diag([2., 3., 4., 1.])
    image, mask = tmp_path / "image.nii.gz", tmp_path / "mask.nii.gz"
    values = np.arange(18 * 24 * 2, dtype=np.float32).reshape(18, 24, 2)
    nib.save(nib.Nifti1Image(values, affine), image)
    nib.save(nib.Nifti1Image(np.zeros(values.shape, dtype=np.uint8), affine), mask)
    split = tmp_path / "test.txt"; split.write_text(f"{image},{mask}\n")
    return image, mask, split, affine


def test_dataset_keeps_background_slices_and_checks_geometry(tmp_path):
    image, mask, split, affine = setup_case(tmp_path)
    dataset = HeartDataset(split, image_size=16)
    assert len(dataset) == 2
    assert dataset[0]["mask"].sum() == 0
    nib.save(nib.Nifti1Image(np.zeros((18, 24, 2), dtype=np.uint8), np.eye(4)), mask)
    with pytest.raises(ValueError, match="affine"):
        HeartDataset(split)


def test_inference_preserves_original_shape_and_affine(tmp_path):
    torch.set_num_threads(1)
    image, mask, split, affine = setup_case(tmp_path)
    cfg = {"model": {"features": [4, 8], "dropout": 0.0}, "training": {"batch_size": 1}, "data": {"image_size": 16, "splits_dir": str(tmp_path)}, "logging": {}}
    config = tmp_path / "config.yaml"; config.write_text(yaml.safe_dump(cfg))
    checkpoint = tmp_path / "model.pt"; torch.save(UNet(features=[4, 8], dropout=0).state_dict(), checkpoint)
    result = nib.load(infer(str(image), str(tmp_path / "out"), str(checkpoint), str(config)))
    assert result.shape == (18, 24, 2)
    np.testing.assert_allclose(result.affine, affine)
    assert result.get_data_dtype() == np.dtype("uint8")
    output = evaluate(str(checkpoint), "test", str(config), str(tmp_path / "metrics.csv"))
    assert "test,2," in output.read_text()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -0.1, 1.1])
def test_invalid_probabilities_cannot_score_as_perfect_background(value):
    with pytest.raises(ValueError):
        per_sample_overlap(torch.full((1, 1, 2, 2), value), torch.zeros(1, 1, 2, 2))


def test_targets_must_be_binary():
    with pytest.raises(ValueError, match="binary"):
        per_sample_overlap(torch.zeros(1, 1, 2, 2), torch.full((1, 1, 2, 2), 0.5))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_inference_rejects_invalid_probabilities(tmp_path, monkeypatch, value):
    class InvalidModel(torch.nn.Module):
        def forward(self, image):
            return torch.full_like(image, value)

    monkeypatch.setattr("scripts.infer.build_model", lambda cfg: InvalidModel())
    image, _, _, _ = setup_case(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"model": {}, "training": {},
                                      "data": {"image_size": 16}, "logging": {}}))
    checkpoint = tmp_path / "model.pt"
    torch.save({}, checkpoint)
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="probabilities"):
        infer(str(image), str(output), str(checkpoint), str(config))
    assert not list(output.glob("*_mask.nii.gz"))
