from pathlib import Path
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"

def load_config(path=None):
    if path is None:
        path = DEFAULT_CONFIG

    with open(path, "r") as f:
        return yaml.safe_load(f)
