# effective_complexity/evaluation/plots.py

import os
import numpy as np
import matplotlib.pyplot as plt


def find_distribution_limits(a, b):
    min_limit = np.minimum(a.min(axis=0), b.min(axis=0))
    max_limit = np.maximum(a.max(axis=0), b.max(axis=0))
    return min_limit, max_limit


def show_distrib(distrib, ax, predicted=True, method="PCA", limit=None):

    exp = "predicted" if predicted else "reference"
    dim = distrib.shape[1]

    if dim == 2:
        ax.scatter(distrib[:, 0], distrib[:, 1], s=5, alpha=0.6)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")

    elif dim == 3:
        ax.scatter(
            distrib[:, 0],
            distrib[:, 1],
            distrib[:, 2],
            s=5,
            alpha=0.6
        )
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_zlabel("Dim 3")

    ax.set_title(f"{method} {exp}")

    if limit is not None:
        mn, mx = limit
        ax.set_xlim(mn[0], mx[0])
        ax.set_ylim(mn[1], mx[1])
        if dim == 3:
            ax.set_zlim(mn[2], mx[2])


def plot_cca_scatter(U, V, i, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    x = U[:, i]
    y = V[:, i]
    a, b = np.polyfit(x, y, 1)

    plt.figure(figsize=(4, 4))
    plt.scatter(x, y, s=5, alpha=0.3)
    plt.plot(x, a * x + b, color="red")
    plt.grid()
    plt.axis("equal")

    fname = os.path.join(save_dir, f"cca_dim_{i+1}.png")
    plt.savefig(fname, dpi=200)
    plt.close()

    return fname
