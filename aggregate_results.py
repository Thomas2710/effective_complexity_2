import os
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = "experiments_out"
OUTPUT = "aggregated_figures"

os.makedirs(OUTPUT, exist_ok=True)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_pca_curve(run_dir, name):
    path = os.path.join(run_dir, "plots", f"{name}.json")
    if not os.path.exists(path):
        return None
    return np.array(load_json(path))


def load_residual_cca(run_dir):
    path = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(path):
        return None
    return np.array(load_json(path)["residual_cca_corrs"])


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

for exp in os.listdir(ROOT):

    exp_path = os.path.join(ROOT, exp)
    if not os.path.isdir(exp_path):
        continue

    # ---------------- PCA variance ----------------
    plt.figure()

    for arch in os.listdir(exp_path):

        arch_path = os.path.join(exp_path, arch)
        if not os.path.isdir(arch_path):
            continue

        curves = []

        for seed in os.listdir(arch_path):
            run_path = os.path.join(arch_path, seed)
            curve = load_pca_curve(run_path, "pca_variance")
            if curve is not None:
                curves.append(curve)

        if len(curves) == 0:
            continue

        curves = np.stack(curves)
        plt.plot(curves.mean(axis=0), label=arch)

    plt.xlabel("Number of components")
    plt.ylabel("Cumulative variance explained")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(OUTPUT, f"{exp}_pca_variance.png"))
    plt.close()

    # ---------------- PCA reconstruction error ----------------
    plt.figure()

    for arch in os.listdir(exp_path):

        arch_path = os.path.join(exp_path, arch)
        if not os.path.isdir(arch_path):
            continue

        curves = []

        for seed in os.listdir(arch_path):
            run_path = os.path.join(arch_path, seed)
            curve = load_pca_curve(run_path, "pca_reconstruction_error")
            if curve is not None:
                curves.append(curve)

        if len(curves) == 0:
            continue

        curves = np.stack(curves)
        plt.plot(curves.mean(axis=0), label=arch)

    plt.xlabel("Number of components")
    plt.ylabel("Reconstruction error")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(OUTPUT, f"{exp}_pca_reconstruction.png"))
    plt.close()

    # ---------------- Residual CCA ----------------
    plt.figure()

    for arch in os.listdir(exp_path):

        arch_path = os.path.join(exp_path, arch)
        if not os.path.isdir(arch_path):
            continue

        curves = []

        for seed in os.listdir(arch_path):
            run_path = os.path.join(arch_path, seed)
            curve = load_residual_cca(run_path)
            if curve is not None:
                curves.append(curve)

        if len(curves) == 0:
            continue

        curves = np.stack(curves)
        plt.plot(curves.mean(axis=0), label=arch)

    plt.xlabel("CCA component")
    plt.ylabel("Residual correlation")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(OUTPUT, f"{exp}_residual_cca.png"))
    plt.close()
