# effective_complexity/training/losses.py

import torch.nn as nn


def kl_divergence():
    """
    KL(teacher || student) using log-softmax student outputs.
    """
    return nn.KLDivLoss(reduction="batchmean")
