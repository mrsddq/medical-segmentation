from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PortfolioContractTests(unittest.TestCase):
    def test_required_portfolio_docs_exist(self):
        for relative_path in [
            "docs/PORTFOLIO_EVIDENCE.md",
            "docs/RESULTS.md",
            "docs/MODEL_CARD.md",
            "docs/REPRODUCIBILITY.md",
            "docs/DEPLOYMENT_NOTES.md",
        ]:
            with self.subTest(path=relative_path):
                self.assertTrue((ROOT / relative_path).is_file())

    def test_readme_links_evidence_without_overclaiming(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("Portfolio Evidence Plan", readme)
        self.assertIn("No verified public metrics are committed yet", readme)

    def test_training_config_has_reproducibility_contract(self):
        config = (ROOT / "configs/unet.yaml").read_text(encoding="utf-8")
        for needle in ["seed:", "model:", "training:", "data:", "logging:"]:
            with self.subTest(needle=needle):
                self.assertIn(needle, config)


if __name__ == "__main__":
    unittest.main()
