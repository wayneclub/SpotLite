import json
from pathlib import Path

# Directory containing config files
CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parent

# cache for arbitrary config files
_CONFIG_CACHE: dict[str, dict] = {}


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_config(name: str) -> dict:
    """Load a JSON config from the configs/ directory.

    Examples:
        load_config("configs.json")
        load_config("crawler.json")
        load_config("aspect_seeds.json")
    """
    global _CONFIG_CACHE

    if name in _CONFIG_CACHE:
        return _CONFIG_CACHE[name]

    cfg_path = CONFIG_DIR / name
    example_path = CONFIG_DIR / f"{name}.example"

    data = {}
    if cfg_path.exists():
        data = _load_json(cfg_path)
    elif example_path.exists():
        data = _load_json(example_path)

    _CONFIG_CACHE[name] = data
    return data


def get_config() -> dict:
    """Backward-compatible helper for older code.

    Returns the content of configs/configs.json.
    Newer code should call load_config("configs.json")
    or define more specific helpers.
    """
    return load_config("configs.json")
