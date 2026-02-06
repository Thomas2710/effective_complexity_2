# effective_complexity/evaluation/pca.py

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def apply_pca(data, num_components):
    pca = PCA(n_components=num_components)
    reduced = pca.fit_transform(data)
    reconstructed = pca.inverse_transform(reduced)
    variance = np.cumsum(pca.explained_variance_ratio_)
    return reduced, reconstructed, variance


def apply_tsne(data, num_components=2, perplexity=30, random_state=42):
    tsne = TSNE(
        n_components=num_components,
        perplexity=perplexity,
        random_state=random_state
    )
    return tsne.fit_transform(data)
