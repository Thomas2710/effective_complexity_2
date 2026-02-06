# effective_complexity/models/factory.py

from effective_complexity.models.mlp import MLP
from effective_complexity.models.resmlp import ResMLP


def build_model(cfg: dict):
    arch = cfg.get("arch_type", "mlp")

    if arch == "mlp":
        return MLP(
            input_size=cfg["input_size"],
            hidden_sizes=cfg["hidden_sizes"],
            embedding_size=cfg["embedding_size"],
            num_classes=cfg["num_classes"],
        )

    elif arch == "resmlp":
        return ResMLP(
            input_size=cfg["input_size"],
            hidden_sizes=cfg["hidden_sizes"],
            embedding_size=cfg["embedding_size"],
            num_classes=cfg["num_classes"],
        )

    else:
        raise ValueError(f"Unknown architecture type: {arch}")
