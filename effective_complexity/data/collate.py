# effective_complexity/data/collate.py

import torch


def collate_dict(batch):
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
    }
