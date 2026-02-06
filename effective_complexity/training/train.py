# effective_complexity/training/train.py

import torch
import torch.nn as nn
from effective_complexity.training.regularization import (
    elastic_net_regularization
)


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    cfg,
    device="cpu",
):
    model.train()

    logsoftmax = nn.LogSoftmax(dim=-1)
    total_loss = 0.0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["label"].to(device)

        optimizer.zero_grad()

        f = model(x)
        f_x = model.get_fx(f)
        W = model.get_W().to(device)

        logits = f_x @ W.T
        log_probs = logsoftmax(logits)

        loss = criterion(log_probs, y)

        # 🔹 add regularization
        loss = loss + elastic_net_regularization(model, cfg)

        loss.backward()
        optimizer.step()


        total_loss += loss.item()

    return total_loss / len(loader)
