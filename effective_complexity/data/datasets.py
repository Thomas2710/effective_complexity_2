# effective_complexity/data/datasets.py

from effective_complexity.data.synthetic import build_synthetic


DATASET_REGISTRY = {
    "synthetic": build_synthetic
}


def get_dataloaders(cfg):
    dataset_type = cfg.get("type", "synthetic")

    if dataset_type not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return DATASET_REGISTRY[dataset_type](cfg)
