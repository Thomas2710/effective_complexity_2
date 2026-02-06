import os
import matplotlib.pyplot as plt
import numpy as np


def plot_pca_curves(variance, errors, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    x = np.arange(1, len(variance) + 1)

    # ---- Variance ----
    plt.figure()
    plt.plot(x, variance, marker="o")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Variance Explained")
    plt.grid()
    plt.savefig(os.path.join(out_dir, "pca_variance.png"))
    plt.close()

    # ---- Reconstruction Error ----
    plt.figure()
    plt.plot(x, errors, marker="o")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Reconstruction Error (MSE)")
    plt.grid()
    plt.savefig(os.path.join(out_dir, "pca_reconstruction_error.png"))
    plt.close()
    
    with open(os.path.join(out_dir, "pca_variance.json"), "w") as f:
    json.dump(variance.tolist(), f)

    with open(os.path.join(out_dir, "pca_reconstruction_error.json"), "w") as f:
        json.dump(errors.tolist(), f)

