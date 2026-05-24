import torch

from models.unet import UNet
from scripts.metrics import dice_bce_loss, dice_score
from scripts.utils import load_config


def test_unet_forward_shape():
    model = UNet(features=(8, 16), dropout=0.0)
    output = model(torch.rand(1, 1, 64, 64))

    assert output.shape == (1, 1, 64, 64)
    assert torch.all(output >= 0)
    assert torch.all(output <= 1)


def test_metrics_are_bounded():
    prediction = torch.ones(1, 1, 8, 8)
    target = torch.ones(1, 1, 8, 8)

    assert dice_score(prediction, target).item() == 1.0
    assert dice_bce_loss(prediction, target).item() >= 0.0


def test_config_loads():
    config = load_config("configs/unet.yaml")

    assert config["model"]["in_channels"] == 1
    assert config["training"]["loss"] == "dice_bce"
