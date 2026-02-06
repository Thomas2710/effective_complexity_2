# effective_complexity/training/optim.py

import torch.optim as optim


OPTIMIZERS = {
    "adam": lambda params, lr: optim.Adam(params, lr=lr),
    "sgd": lambda params, lr: optim.SGD(params, lr=lr, momentum=0.9),
    "adamw": lambda params, lr: optim.AdamW(params, lr=lr),
}


def build_optimizer(cfg, model):
    name = cfg.get("optimizer", "adam")
    lr = cfg.get("lr", 1e-3)

    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {name}")

    return OPTIMIZERS[name](model.parameters(), lr)
