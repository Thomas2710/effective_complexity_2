# effective_complexity/experiments/single_run.py

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from effective_complexity.models import build_model, initialize_weights
from effective_complexity.training import (
    kl_divergence,
    build_optimizer,
    EarlyStopping,
    train_one_epoch,
    validate,
)
from effective_complexity.utils.seed import set_seed

# ---------- Evaluation ----------
from effective_complexity.evaluation.cca import (
    run_cca,
    cca_with_directions,
    get_canonical_variables,
)
from effective_complexity.evaluation.alignment import (
    linear_alignment_error_projected,
)
from effective_complexity.evaluation.diversity import check_diversity
from effective_complexity.evaluation.pca import apply_pca, apply_tsne
from effective_complexity.evaluation.plots import (
    show_distrib,
    find_distribution_limits,
    plot_cca_scatter,
)

from effective_complexity.evaluation.residual_cca import residual_cca
from effective_complexity.evaluation.pca_spectrum import pca_spectrum
from effective_complexity.evaluation.pca_plots import plot_pca_curves


# -----------------------------------------------------------
# Main entry
# -----------------------------------------------------------

def run_single_experiment(cfg, loaders, run_dir):

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    model.apply(initialize_weights)

    criterion = kl_divergence()
    optimizer = build_optimizer(cfg, model)

    stopper = EarlyStopping(
        patience=cfg.get("patience", 15),
        delta=float(cfg.get("delta", 1e-6)) ,
    )

    check_interval = cfg.get("check_interval", 50)

    os.makedirs(run_dir, exist_ok=True)
    plot_root = os.path.join(run_dir, "plots")
    os.makedirs(plot_root, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    best_path = os.path.join(run_dir, "best_model.pt")

    # ===========================================================
    # Training loop
    # ===========================================================
    print(
    f"\n[RUN] "
    f"Teacher d = {cfg['input_size']} | "
    f"Student d = {cfg['embedding_size']} | "
    f"Classes = {cfg['num_classes']} | "
    f"Arch = {cfg['architecture']}\n"
)
    for epoch in tqdm(range(cfg["epochs"])):

        train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            criterion,
            cfg,
            device
        )

        val_loss = validate(
            model,
            loaders["val"],
            criterion,
            device
        )


        stop, improved = stopper.step(val_loss, epoch)

        if improved:
            torch.save(model.state_dict(), best_path)

        # ---------------- Dynamic evaluation ----------------
        if epoch % check_interval == 0:

            epoch_plot_dir = os.path.join(plot_root, f"epoch_{epoch}")
            os.makedirs(epoch_plot_dir, exist_ok=True)

            # Collect embeddings
            emb_tmp = []
            with torch.no_grad():
                for batch in loaders["test"]:
                    x = batch["x"].to(device)
                    f = model(x)
                    emb_tmp.append(model.get_fx(f))

            F_student_tmp = torch.cat(emb_tmp)
            F_teacher = loaders["test"].dataset.f_x

            Fs = F_student_tmp.cpu().numpy()

            # ----- PCA spectrum -----
            variance, errors = pca_spectrum(Fs)
            plot_pca_curves(variance, errors, epoch_plot_dir)

            # ----- CCA -----
            cca_corrs_epoch = run_cca(F_teacher, F_student_tmp)

            plt.figure()
            plt.plot(cca_corrs_epoch, marker="o")
            plt.xlabel("CCA component")
            plt.ylabel("Correlation")
            plt.grid()
            plt.savefig(os.path.join(epoch_plot_dir, "cca.png"))
            plt.close()

        if stop:
            break

    # Load best model
    model.load_state_dict(torch.load(best_path, map_location=device))

    # ===========================================================
    # Final embeddings
    # ===========================================================

    embeddings = []
    with torch.no_grad():
        for batch in loaders["test"]:
            x = batch["x"].to(device)
            f = model(x)
            embeddings.append(model.get_fx(f))

    F_student = torch.cat(embeddings)
    F_teacher = loaders["test"].dataset.f_x

    Fs = F_student.cpu().numpy()
    Ft = F_teacher.cpu().numpy()

    # ===========================================================
    # Final evaluation
    # ===========================================================

    cca_corrs = run_cca(F_teacher, F_student)

    teacher_d = F_teacher.shape[1]
    k = min(teacher_d, cfg["num_classes"] - 1)

    corrs, U_t, U_s = cca_with_directions(
        F_teacher, F_student, k
    )

    residual_corrs = residual_cca(
        F_teacher, F_student, k
    )

    align_err = linear_alignment_error_projected(
        F_teacher, F_student, U_t, U_s
    )

    diversity = check_diversity(model, F_student)

    # ===========================================================
    # Final plots
    # ===========================================================

    variance, errors = pca_spectrum(Fs)
    plot_pca_curves(variance, errors, plot_root)

    dim = min(3, Fs.shape[1], Ft.shape[1])

    pca_s, _, _ = apply_pca(Fs, dim)
    pca_t, _, _ = apply_pca(Ft, dim)

    tsne_s = apply_tsne(Fs, dim)
    tsne_t = apply_tsne(Ft, dim)

    projection = "3d" if dim == 3 else None

    # PCA scatter
    fig, axs = plt.subplots(
        1, 2, figsize=(10,5),
        subplot_kw={"projection": projection} if projection else {}
    )

    limit = find_distribution_limits(pca_t, pca_s)

    show_distrib(pca_t, axs[0], predicted=False, method="PCA", limit=limit)
    show_distrib(pca_s, axs[1], predicted=True, method="PCA", limit=limit)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_root, "pca.png"))
    plt.close()

    # t-SNE scatter
    fig, axs = plt.subplots(
        1, 2, figsize=(10,5),
        subplot_kw={"projection": projection} if projection else {}
    )

    limit = find_distribution_limits(tsne_t, tsne_s)

    show_distrib(tsne_t, axs[0], predicted=False, method="TSNE", limit=limit)
    show_distrib(tsne_s, axs[1], predicted=True, method="TSNE", limit=limit)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_root, "tsne.png"))
    plt.close()

    # CCA scatter
    U, V = get_canonical_variables(F_teacher, F_student, k)
    for i in range(k):
        plot_cca_scatter(U, V, i, plot_root)

    # ===========================================================
    # Return results
    # ===========================================================

    return {
        "architecture": cfg["architecture"],
        "seed": int(cfg["seed"]),
        "cca_corrs": [float(x) for x in cca_corrs],
        "residual_cca_corrs": [float(x) for x in residual_corrs],
        "align_err_proj": float(align_err),
        "rank_W": int(diversity["rank_W"]),
        "rank_embedding_cov": int(diversity["rank_embedding_cov"]),
    }

