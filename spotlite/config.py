import json
from pathlib import Path

_CONFIG_CACHE = None


def get_config():
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "configs.json"
    example_path = root / "configs.example.json"

    if cfg_path.exists():
        _CONFIG_CACHE = json.load(cfg_path.open("r", encoding="utf-8"))
    elif example_path.exists():
        _CONFIG_CACHE = json.load(example_path.open("r", encoding="utf-8"))
    else:
        _CONFIG_CACHE = {}

    return _CONFIG_CACHE
