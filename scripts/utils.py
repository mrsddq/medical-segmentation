from pathlib import Path

import yaml


REQUIRED_CONFIG_SECTIONS = {"model", "training", "data", "logging"}


def load_config(path):
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")

    missing = sorted(REQUIRED_CONFIG_SECTIONS - set(config))
    if missing:
        raise ValueError(f"Config is missing sections: {', '.join(missing)}")

    return config
