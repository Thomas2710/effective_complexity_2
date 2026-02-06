# effective_complexity/models/mlp.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Plain MLP producing embeddings f(x) and
    linear unembedding head W.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        embedding_size: int,
        num_classes: int,
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        in_features = input_size
        for h in hidden_sizes:
            self.layers.append(nn.Linear(in_features, h))
            in_features = h

        self.fx = nn.Linear(in_features, embedding_size)
        self.W = nn.Linear(embedding_size, num_classes, bias=False)

    # ------------------------------------------------

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        f = self.fx(x)
        return f

    # ------------------------------------------------
    # Compatibility helpers (match old script)
    # ------------------------------------------------

    def get_fx(self, f):
        return f

    def get_W(self):
        return self.W.weight

    def get_unembeddings(self, y):
        return torch.matmul(self.W.weight.t(), y)
