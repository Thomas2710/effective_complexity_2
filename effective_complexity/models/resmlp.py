# effective_complexity/models/resmlp.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResMLP(nn.Module):
    """
    Residual MLP version.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        embedding_size: int,
        num_classes: int,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        in_features = input_size

        for h in hidden_sizes:
            block = nn.ModuleDict({
                "fc1": nn.Linear(in_features, h),
                "fc2": nn.Linear(h, h),
            })
            self.blocks.append(block)
            in_features = h

        self.fx = nn.Linear(in_features, embedding_size)
        self.W = nn.Linear(embedding_size, num_classes, bias=False)

    # ------------------------------------------------

    def forward(self, x):
        h = x

        for block in self.blocks:
            residual = h

            h = F.relu(block["fc1"](h))
            h = block["fc2"](h)

            if residual.shape[-1] != h.shape[-1]:
                residual = F.linear(
                    residual,
                    torch.eye(
                        h.shape[-1],
                        residual.shape[-1],
                        device=h.device
                    )
                )

            h = F.relu(h + residual)

        f = self.fx(h)
        return f

    # ------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------

    def get_fx(self, f):
        return f

    def get_W(self):
        return self.W.weight

    def get_unembeddings(self, y):
        return torch.matmul(self.W.weight.t(), y)
