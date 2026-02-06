# effective_complexity/training/validate.py

import torch
import torch.nn as nn


def validate(
    model,
    loader,
    criterion,
    device="cpu",
):
    model.eval()

    logsoftmax = nn.LogSoftmax(dim=-1)
    total_kl = 0.0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["label"].to(device)

            f = model(x)
            f_x = model.get_fx(f)
            W = model.get_W().to(device)

            logits = f_x @ W.T
            log_probs = logsoftmax(logits)

            kl = criterion(log_probs, y)
            total_kl += kl.item()

    return total_kl / len(loader)
