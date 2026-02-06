# effective_complexity/data/synthetic.py

import math
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from effective_complexity.models.factory import build_model
from effective_complexity.models.init import initialize_weights
from effective_complexity.utils.seed import set_seed


class SyntheticDataset(Dataset):
    """
    Dataset returning:
      x      : input vector
      label  : teacher probability distribution
    """

    def __init__(self, x, label, f_x=None):
        self.x = x
        self.label = label
        self.f_x = f_x  # optional (teacher embeddings)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return {
            "x": self.x[idx],
            "label": self.label[idx],
        }


def _make_full_rank_orthogonal_matrix(d, C, scale=3.0):
    A = torch.randn(d, C)
    Q, _ = torch.linalg.qr(A)
    return scale * Q[:, :C]


def generate_synthetic_data(cfg):
    """
    Generates synthetic dataset using a teacher MLP.

    Returns:
        train_ds, val_ds, test_ds
    """

    set_seed(cfg["seed"])

    # -------------------------
    # Parameters
    # -------------------------
    d = cfg["d"]
    C = cfg["num_classes"]
    N = cfg["num_samples"]
    mu = cfg.get("mu", 0.0)
    cov = cfg.get("cov", 10.0)
    splits = cfg.get("splits", (0.7, 0.1, 0.2))
    train_p, val_p, test_p = splits

    # -------------------------
    # Sample inputs
    # -------------------------
    mean = np.full(d, mu)
    cov_mat = np.eye(d) * cov

    x = np.random.multivariate_normal(mean, cov_mat, N)
    x = torch.from_numpy(x).float()

    # -------------------------
    # Teacher model
    # -------------------------
    teacher_cfg = cfg.copy()
    teacher_cfg["embedding_size"] = d
    teacher_cfg["arch_type"] = "mlp"

    teacher = build_model(teacher_cfg)
    teacher.apply(initialize_weights)
    teacher.eval()

    with torch.no_grad():
        f_x = teacher(x)

    # -------------------------
    # Teacher linear head
    # -------------------------
    W = _make_full_rank_orthogonal_matrix(d, C)
    logits = f_x @ W
    probs = torch.softmax(logits, dim=-1)

    # -------------------------
    # Split
    # -------------------------
    n_train = int(train_p * N)
    n_val = int(val_p * N)

    train_ds = SyntheticDataset(
        x[:n_train],
        probs[:n_train]
    )

    val_ds = SyntheticDataset(
        x[n_train:n_train + n_val],
        probs[n_train:n_train + n_val]
    )

    test_ds = SyntheticDataset(
        x[n_train + n_val:],
        probs[n_train + n_val:],
        f_x=f_x[n_train + n_val:]
    )


    return train_ds, val_ds, test_ds


def build_synthetic(cfg):
    """
    Convenience wrapper returning PyTorch DataLoaders.
    """

    train_ds, val_ds, test_ds = generate_synthetic_data(cfg)

    return {
        "train": DataLoader(
            train_ds,
            batch_size=cfg["batch_size"],
            shuffle=True,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=cfg["batch_size"],
            shuffle=False,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=cfg["batch_size"],
            shuffle=False,
        ),
    }
