# effective_complexity/evaluation/diversity.py

import numpy as np


def check_diversity(model, embeddings, eps=1e-6):

    W = model.get_W().detach().cpu().numpy()
    _, S_W, _ = np.linalg.svd(W)
    rank_W = np.sum(S_W > eps)

    F = embeddings.detach().cpu().numpy()
    F = F - F.mean(axis=0, keepdims=True)
    cov = np.cov(F, rowvar=False)

    eigvals = np.linalg.eigvalsh(cov)
    rank_cov = np.sum(eigvals > eps)

    return {
        "rank_W": rank_W,
        "singular_values_W": S_W,
        "rank_embedding_cov": rank_cov,
        "embedding_cov_eigvals": eigvals
    }
