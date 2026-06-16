import tempfile
import unittest
from pathlib import Path

import torch

from data.dataset import HeartDataset
from models.unet import UNet
from scripts.metrics import dice_bce_loss, dice_score
from scripts.utils import load_config


class UNetTests(unittest.TestCase):
    def test_unet_forward_shape(self):
        model = UNet(features=(8, 16), dropout=0.0)
        output = model(torch.rand(1, 1, 64, 64))

        self.assertEqual(output.shape, (1, 1, 64, 64))
        self.assertTrue(torch.all(output >= 0).item())
        self.assertTrue(torch.all(output <= 1).item())

    def test_metrics_are_bounded(self):
        prediction = torch.ones(1, 1, 8, 8)
        target = torch.ones(1, 1, 8, 8)

        self.assertEqual(dice_score(prediction, target).item(), 1.0)
        self.assertGreaterEqual(dice_bce_loss(prediction, target).item(), 0.0)

    def test_config_loads(self):
        config = load_config("configs/unet.yaml")

        self.assertEqual(config["model"]["in_channels"], 1)
        self.assertEqual(config["training"]["loss"], "dice_bce")

    def test_normalize_constant_slice_returns_zeros(self):
        image = torch.ones(4, 4).numpy()

        normalized = HeartDataset._normalize(image)

        self.assertEqual(normalized.shape, image.shape)
        self.assertEqual(normalized.max(), 0.0)
        self.assertEqual(normalized.min(), 0.0)

    def test_load_config_rejects_missing_sections(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Path(temp_dir) / "broken.yaml"
            config.write_text("model:\n  in_channels: 1\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing sections"):
                load_config(config)


if __name__ == "__main__":
    unittest.main()
