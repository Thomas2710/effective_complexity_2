import matplotlib.pyplot as plt
import numpy as np
#from effective_complexity.model import MLP, initialize_weights
import torch.nn as nn
import torch
from tqdm import tqdm
import torch.optim as optim
import os
from datetime import datetime

from torch.utils.data import DataLoader
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision import transforms, utils
from effective_complexity.models.models import DistributionComponents
import torch.nn.functional as F
import math
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import wandb
from sklearn.cross_decomposition import CCA
import pandas as pd

from collections import defaultdict
import json
import numpy as np



def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


hyperparams = {
    "COV": 10,
    "MU" : 0,
    "d" : 3,
    "input_size" : 3,
    "hidden_sizes" : [64,32,64],
    "embedding_size" : 10,
    "num_samples" : 10000,
    "BATCH_SIZE" : 256,
    "flatten_input" : False,
    "num_classes" : 3,
    "epochs" : 500,
    "lr" : 1e-2,
    "check_interval" : 50,
    "num_classes" : 3,
    "l1_lambda" : 1e-4,
    "l2_lambda" : 1e-4,
    "l1_weight" : 0,
    "l2_weight" : 0,
    "optimizer": "sgd",
}

#Plot distribution in input, in plt ax in input
def show_distrib(distrib, method = 'NO', predicted = True, ax = None, limit=None):
    min_limit, max_limit = limit
    dim = distrib.shape[1]
    if predicted:
        exp = 'predicted'
    else:
        exp = 'reference'

    if dim == 2:
        ax.scatter(distrib[:, 0], distrib[:, 1], c='blue', alpha=0.6)
        ax.set_title(f"{method.upper()} Reduced Visualization of {exp} Data (2D)")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_xlim(min_limit[0],max_limit[0])
        ax.set_ylim(min_limit[1],max_limit[1])
    elif dim == 3:
        ax.scatter(distrib[:, 0], distrib[:, 1], distrib[:, 2], c='blue', alpha=0.6)
        ax.set_title(f"{method.upper()} Reduced Visualization of {exp} Data (3D)")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_zlabel("Dimension 3")
        ax.set_xlim(min_limit[0],max_limit[0])
        ax.set_ylim(min_limit[1],max_limit[1])
        ax.set_zlim(min_limit[2],max_limit[2])
    else:
        raise ValueError("target_dim must be 2 or 3.")

#Find the range of two distributions in each dimension
def find_distribution_limits(distrib1, distrib2):
    # Find min and max per dimension (i.e., column)
    min_per_dimension = np.min(distrib1, axis=0)
    max_per_dimension = np.max(distrib2, axis=0)

    min_per_pred_dimension = np.min(distrib1, axis=0)
    max_per_pred_dimension = np.max(distrib2, axis=0)

    min_limit = np.minimum(min_per_dimension, min_per_pred_dimension)
    max_limit = np.maximum(max_per_dimension, max_per_pred_dimension)
    limit = (min_limit, max_limit)
    return limit

#Apply PCA dim reduction to distribution
def apply_pca(data, num_components=2):
    # Apply PCA to reduce to 2 dimensions
    pca = PCA(n_components=num_components)
    pca_result = pca.fit_transform(data)
    data_reconstructed = pca.inverse_transform(pca_result)
    variance_explained = np.cumsum(pca.explained_variance_ratio_)

    return pca_result, data_reconstructed, variance_explained

#Apply t-SNE dim reduction to distribution
def apply_tsne(data, num_components=2, perplexity=30, random_state=42):
    # Apply t-SNE to reduce to 2 dimensions for visualization
    tsne = TSNE(n_components=num_components, perplexity=perplexity, random_state=random_state)
    tsne_result = tsne.fit_transform(data)
    return tsne_result

def run_cca(F_teacher, F_student, max_components=None):
    """
    Runs CCA and returns canonical correlations.
    """

    Ft = F_teacher.cpu().numpy()
    Fs = F_student.cpu().numpy()

    if max_components is None:
        max_components = min(Ft.shape[1], Fs.shape[1])

    cca = CCA(n_components=max_components)
    cca.fit(Ft, Fs)
    Xc, Yc = cca.transform(Ft, Fs)

    corrs = []
    for i in range(max_components):
        corr = np.corrcoef(Xc[:, i], Yc[:, i])[0, 1]
        corrs.append(corr)

    return np.array(corrs)


def analyze_embeddings(
    predicted_batches,
    predicted_distrib_batches,
    reference_embeddings,
    plots_folder_path,
    embedding_size,
    num_classes,
    epoch_for_logging=0
):
    """
    Analyze embeddings: PCA reconstruction, variance explained, and t-SNE / PCA plots.

    Args:
        predicted_batches (list of tensors): Model embeddings per batch
        predicted_distrib_batches (list of tensors): Model soft-label distributions per batch
        reference_embeddings (tensor): Ground-truth f_x embeddings from dataset
        plots_folder_path (str): Folder to save plots
        embedding_size (int): Dimensionality of model embeddings
        num_classes (int): Number of output classes
        epoch_for_logging (int): Optional, for log labeling
    """

    # Merge batches
    embeddings = torch.cat(predicted_batches)
    predicted_distrib = torch.cat(predicted_distrib_batches)

    #-----------------------------
    # Scale embeddings independently
    scaler_pred = MinMaxScaler()
    embeddings_scaled = scaler_pred.fit_transform(embeddings)

    scaler_ref = MinMaxScaler()
    reference_embeddings_scaled = scaler_ref.fit_transform(reference_embeddings)

    #-----------------------------
    # Compute reconstruction error / variance via PCA
    true_dim = embeddings.shape[1]   # <-- real dimension
    num_components = np.arange(1, true_dim + 1)

    errors = []
    variance = []

    for n in num_components:
        pcareduced, reconstructed, variance_explained = apply_pca(
            torch.from_numpy(embeddings_scaled).float(), num_components=n
        )
        reconstruction_error = np.mean((embeddings_scaled - reconstructed)**2)
        errors.append(reconstruction_error)
        variance.append(variance_explained[-1].item())

        # Optional: log to wandb per component
        wandb.define_metric(f"variance_explained/{n}_component", step_metric="epoch_mod_interval")
        wandb.define_metric(f"reconstruction_error/{n}_component", step_metric="epoch_mod_interval")
        wandb.log({
            "epoch_mod_interval": epoch_for_logging,
            f"variance_explained/{n}_component": variance_explained[-1].item(),
            f"reconstruction_error/{n}_component": reconstruction_error
        })

    # Plot reconstruction error vs components
    plt.figure(figsize=(8, 5))
    plt.plot(num_components, errors, color='red', marker='o')
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Reconstruction Error (MSE)")
    plt.title("PCA Reconstruction Error vs. Number of Components")
    plt.grid(True)
    plt.savefig(os.path.join(plots_folder_path, f'epoch_{epoch_for_logging}_PCA_MSE.png'))
    plt.close()

    # Plot variance explained
    plt.figure(figsize=(8, 5))
    plt.plot(num_components, variance, color='green', marker='o')
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Variance Explained")
    plt.title("PCA Variance Explained vs. Number of Components")
    plt.grid(True)
    plt.savefig(os.path.join(plots_folder_path, f'epoch_{epoch_for_logging}_PCA_VAR.png'))
    plt.close()

    #-----------------------------
    # Visualize embeddings in PCA / t-SNE space
    dim_student = embeddings.shape[1]
    dim_teacher = reference_embeddings_scaled.shape[1]

    min_components = min(3, dim_student, dim_teacher)

    pcareduced_pred, _, _ = apply_pca(torch.from_numpy(embeddings_scaled).float(), num_components=min_components)
    pcareduced_ref, _, _ = apply_pca(torch.from_numpy(reference_embeddings_scaled).float(), num_components=min_components)

    tsnereduced_pred = apply_tsne(embeddings_scaled, num_components=min_components)
    tsnereduced_ref = apply_tsne(reference_embeddings_scaled, num_components=min_components)

    # PCA plots
    projection = '3d' if min_components == 3 else None
    pca_fig, pca_axs = plt.subplots(1, 2, figsize=(12, 6),
                                    subplot_kw={'projection': projection} if projection else {})
    limit = find_distribution_limits(pcareduced_ref, pcareduced_pred)
    show_distrib(pcareduced_ref, method='PCA', predicted=False, ax=pca_axs[0], limit=limit)
    show_distrib(pcareduced_pred, method='PCA', predicted=True, ax=pca_axs[1], limit=limit)
    pca_fig.tight_layout()
    plt.savefig(os.path.join(plots_folder_path, f'epoch_{epoch_for_logging}_PCA_embeddings.png'))
    plt.close()

    # t-SNE plots
    tsne_fig, tsne_axs = plt.subplots(1, 2, figsize=(12, 6),
                                      subplot_kw={'projection': projection} if projection else {})
    limit = find_distribution_limits(tsnereduced_ref, tsnereduced_pred)
    show_distrib(tsnereduced_ref, method='TSNE', predicted=False, ax=tsne_axs[0], limit=limit)
    show_distrib(tsnereduced_pred, method='TSNE', predicted=True, ax=tsne_axs[1], limit=limit)
    tsne_fig.tight_layout()
    plt.savefig(os.path.join(plots_folder_path, f'epoch_{epoch_for_logging}_TSNE_embeddings.png'))
    plt.close()
    return variance, errors

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_model(hyperparams):
    arch = hyperparams.get("arch_type", "mlp")

    if arch == "mlp":
        return model_mlp(hyperparams)

    elif arch == "resmlp":
        return model_resmlp(hyperparams)

    else:
        raise ValueError(f"Unknown architecture type: {arch}")

def check_diversity(model, embeddings, eps=1e-6):
    """
    Checks diversity conditions numerically.
    
    Args:
        model: trained student or teacher model
        embeddings: tensor [N, d] of f(x)
    """

    # ---- Check rank of W ----
    W = model.get_W().detach().cpu().numpy()
    _, S_W, _ = np.linalg.svd(W)

    rank_W = np.sum(S_W > eps)

    # ---- Check rank of embedding covariance ----
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

def get_canonical_variables(F_teacher, F_student, k):
    Ft = F_teacher.cpu().numpy()
    Fs = F_student.cpu().numpy()

    cca = CCA(n_components=k)
    U, V = cca.fit_transform(Ft, Fs)   # shapes: (N, k)

    return U, V



def cca_with_directions(F_teacher, F_student, k):
    """
    Runs CCA and returns canonical correlations and projection directions.
    """
    Ft = F_teacher.cpu().numpy()
    Fs = F_student.cpu().numpy()

    cca = CCA(n_components=k)
    Xc, Yc = cca.fit_transform(Ft, Fs)

    corrs = [
        np.corrcoef(Xc[:, i], Yc[:, i])[0, 1]
        for i in range(k)
    ]

    # canonical directions (columns)
    U_t = cca.x_weights_   # shape (d, k)
    U_s = cca.y_weights_   # shape (d, k)

    return np.array(corrs), U_t, U_s

def project_out(F, U):
    """
    Removes components of F along columns of U.
    """
    U = torch.from_numpy(U).to(F.device).float()
    return F - (F @ U) @ U.T

def residual_cca_test(F_teacher, F_student, k):
    corrs, U_t, U_s = cca_with_directions(F_teacher, F_student, k)

    F_teacher_res = project_out(F_teacher, U_t)
    F_student_res = project_out(F_student, U_s)

    # run CCA again on residuals
    cca_res = run_cca(F_teacher_res, F_student_res)

    print("\n=== Residual CCA ===")
    for i, c in enumerate(cca_res):
        print(f"Residual CCA dim {i+1}: {c:.4f}")

    return cca_res


#------------------ALIGNMENT TESTS------------------

def linear_alignment_error(F_teacher, F_student):
    """
    Computes min_A || A F_teacher - F_student ||^2
    """
    A = torch.linalg.lstsq(F_teacher, F_student).solution
    pred = F_teacher @ A
    return torch.mean((pred - F_student) ** 2)

def project_to_subspace(F, U):
    U = torch.from_numpy(U).to(F.device).float()
    return F @ U

def linear_alignment_error_projected(F_teacher, F_student, U_t, U_s):
    """
    Computes min_A || A (P_M F_teacher) - (P_M F_student) ||^2
    """
    Ft_proj = project_to_subspace(F_teacher, U_t)
    Fs_proj = project_to_subspace(F_student, U_s)

    A = torch.linalg.lstsq(Ft_proj, Fs_proj).solution
    pred = Ft_proj @ A

    mse = torch.mean((pred - Fs_proj) ** 2)
    return mse

# ------------------END ALIGNMENT TESTS------------------


def plot_cca_scatter(U, V, dims=(0, 1, 2), max_points=2000):
    n = min(len(U), max_points)
    idx = np.random.choice(len(U), n, replace=False)

    for i in dims:
        plt.figure(figsize=(4, 4))
        plt.scatter(U[idx, i], V[idx, i], s=5, alpha=0.5)
        plt.xlabel(f"Teacher CCA dim {i+1}")
        plt.ylabel(f"Student CCA dim {i+1}")
        plt.title(f"CCA dim {i+1}")
        plt.grid()
        plt.axis("equal")
        plt.show()

def plot_cca_with_fit_and_save(
    U, V, i,
    save_dir="cca_plots",
    prefix="cca_fit"
):
    os.makedirs(save_dir, exist_ok=True)

    x = U[:, i]
    y = V[:, i]

    a, b = np.polyfit(x, y, 1)

    plt.figure(figsize=(4, 4))
    plt.scatter(x, y, s=5, alpha=0.3)
    plt.plot(x, a * x + b, color="red", linewidth=2)
    plt.xlabel(f"Teacher CCA dim {i+1}")
    plt.ylabel(f"Student CCA dim {i+1}")
    plt.title(f"CCA dim {i+1} (linear fit)")
    plt.grid()
    plt.axis("equal")

    fname = os.path.join(save_dir, f"{prefix}_dim_{i+1}.png")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved: {fname}")
    return fname

def plot_averaged_metrics(
    averaged_results,
    save_dir="FIGURES",
    max_components=None
):
    """
    Generates paper-ready plots from averaged (mean ± std) metrics.

    Args:
        averaged_results (dict): output of average_results_across_seeds
        save_dir (str): directory where figures are saved
        max_components (int or None): optionally truncate component axis
    """
    os.makedirs(save_dir, exist_ok=True)

    # -------------------------
    # Helper for shaded plots
    # -------------------------
    def plot_mean_std(x, mean, std, label):
        mean = np.array(mean)
        std = np.array(std)
        plt.plot(x, mean, label=label)
        plt.fill_between(x, mean - std, mean + std, alpha=0.25)

    # ==========================================================
    # 1) CCA correlations (mean ± std)
    # ==========================================================
    plt.figure(figsize=(7, 5))

    for arch, res in averaged_results.items():
        mean = res["cca_corrs_mean"]
        std = res["cca_corrs_std"]

        if max_components is not None:
            mean = mean[:max_components]
            std = std[:max_components]

        x = np.arange(1, len(mean) + 1)
        plot_mean_std(x, mean, std, label=arch)

    plt.xlabel("CCA component")
    plt.ylabel("Correlation")
    plt.title("CCA correlations (mean ± std across seeds)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "CCA_mean_std.png"), dpi=300)
    plt.close()

    # ==========================================================
    # 2) Residual CCA correlations
    # ==========================================================
    plt.figure(figsize=(7, 5))

    for arch, res in averaged_results.items():
        mean = res["residual_cca_corrs_mean"]
        std = res["residual_cca_corrs_std"]

        if max_components is not None:
            mean = mean[:max_components]
            std = std[:max_components]

        x = np.arange(1, len(mean) + 1)
        plot_mean_std(x, mean, std, label=arch)

    plt.xlabel("CCA component")
    plt.ylabel("Residual correlation")
    plt.title("Residual CCA (mean ± std across seeds)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Residual_CCA_mean_std.png"), dpi=300)
    plt.close()

    # ==========================================================
    # 3) PCA cumulative variance explained
    # ==========================================================
    plt.figure(figsize=(7, 5))

    for arch, res in averaged_results.items():
        mean = res["variance_explained_mean"]
        std = res["variance_explained_std"]

        if max_components is not None:
            mean = mean[:max_components]
            std = std[:max_components]

        x = np.arange(1, len(mean) + 1)
        plot_mean_std(x, mean, std, label=arch)

    plt.xlabel("Number of PCA components")
    plt.ylabel("Cumulative variance explained")
    plt.title("PCA variance explained (mean ± std)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "PCA_variance_mean_std.png"), dpi=300)
    plt.close()

    # ==========================================================
    # 4) PCA reconstruction error
    # ==========================================================
    plt.figure(figsize=(7, 5))

    for arch, res in averaged_results.items():
        mean = res["errors_mean"]
        std = res["errors_std"]

        if max_components is not None:
            mean = mean[:max_components]
            std = std[:max_components]

        x = np.arange(1, len(mean) + 1)
        plot_mean_std(x, mean, std, label=arch)

    plt.xlabel("Number of PCA components")
    plt.ylabel("Reconstruction error (MSE)")
    plt.title("PCA reconstruction error (mean ± std)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "PCA_reconstruction_error_mean_std.png"), dpi=300)
    plt.close()






def model_mlp(hyperparams):
    class MLP(nn.Module, DistributionComponents):
        def __init__(self, input_size, hidden_sizes, embedding_size=3, output_size=3, flatten_input=False):
            super(MLP, self).__init__()
            # Define layers
            self.layers = nn.ModuleList()
            in_features = input_size
            for hidden_size in hidden_sizes:
                self.layers.append(nn.Linear(in_features, hidden_size))
                in_features = hidden_size
            self.fx = nn.Linear(in_features, embedding_size)
            self.W = nn.Linear(embedding_size, output_size, bias=False)

        def forward(self, x):
            # Pass input through each layer with ReLU activation
            for layer in self.layers:
                x = F.relu(layer(x))
            # Output layer
            x = self.fx(x)
            return x
        
        def get_fx(self, x):
            return x
        
        def get_unembeddings(self, y):
            return torch.matmul(self.W.weight.t(),y)
        
        def get_W(self):
            return self.W.weight

    flatten_input = hyperparams['flatten_input']
    input_size = hyperparams['input_size']
    hidden_sizes = hyperparams['hidden_sizes']
    embedding_size = hyperparams['embedding_size']
    num_classes = hyperparams['num_classes']
    model = MLP(input_size, hidden_sizes,embedding_size, num_classes, flatten_input)
    return model



def model_resmlp(hyperparams):
    class ResMLP(nn.Module):
        def __init__(self, input_size, hidden_sizes, embedding_size=3, output_size=3):
            super().__init__()

            layers = []
            in_features = input_size

            # Build residual blocks
            self.blocks = nn.ModuleList()
            for hidden_size in hidden_sizes:
                block = nn.ModuleDict({
                    "fc1": nn.Linear(in_features, hidden_size),
                    "fc2": nn.Linear(hidden_size, hidden_size)
                })
                self.blocks.append(block)
                in_features = hidden_size

            # Projection to embedding
            self.fx = nn.Linear(in_features, embedding_size)

            # Linear unembedding
            self.W = nn.Linear(embedding_size, output_size, bias=False)

        def forward(self, x):
            h = x
            for block in self.blocks:
                residual = h
                h = F.relu(block["fc1"](h))
                h = block["fc2"](h)

                # match shapes for skip if needed
                if residual.shape[-1] != h.shape[-1]:
                    residual = F.linear(residual, torch.eye(h.shape[-1], residual.shape[-1]).to(h.device))

                h = F.relu(h + residual)

            f = self.fx(h)
            return f

        def get_fx(self, x):
            return x

        def get_unembeddings(self, y):
            return torch.matmul(self.W.weight.t(), y)

        def get_W(self):
            return self.W.weight

    return ResMLP(
        input_size=hyperparams["input_size"],
        hidden_sizes=hyperparams["hidden_sizes"],
        embedding_size=hyperparams["embedding_size"],
        output_size=hyperparams["num_classes"]
    )

eps = 10e-4

def collate_fn(batch):
    """ Custom function to collate dictionary-based data. """
    inputs = torch.stack([item["x"] for item in batch])  # Stack 3D tensors
    labels = torch.stack([item["label"] for item in batch])  # Convert labels to tensor
    return {"x": inputs, "label": labels}


def dataset_synthetic(hyperparams):

    def make_full_rank_orthogonal_matrix(d, C, scale=3.0):
        A = torch.randn(d, C)
        Q, _ = torch.linalg.qr(A)   # orthonormal columns
        return scale * Q[:, :C]    # ensure shape (d, C)


    def build_W_effective_d(d, C, scale=3.0):
        #assert C >= d + 1

        # basis for d-dim subspace
        B = torch.randn(d, d)

        # random class coefficients
        A = torch.randn(d, C)

        W = scale * (B @ A)   # shape (d, C)

        return W

    def assert_orthogonal(W, tol=1e-5):
        D = W.shape[1]
        G = W.T @ W                     # Gram matrix
        target = torch.eye(D) * (W[:,0].norm()**2)

        assert torch.allclose(G, target, atol=tol), \
            "W columns are NOT orthogonal!"
        
    def assert_full_rank(W):
        assert torch.linalg.matrix_rank(W) == W.shape[0], \
        "W is not full rank!"

    def assert_effective_complexity(W, tol=1e-6):
        d, C = W.shape

        Wc = W - W[:, :1]          # subtract reference column
        r = torch.linalg.matrix_rank(Wc, tol=tol)

        expected = min(d, C-1)

        assert r == expected, \
            f"Effective complexity {r}, expected {expected}"



    class SYNTHETIC(Dataset):

        def __init__(self):
            """
            Custom PyTorch Dataset that stores an array of dictionaries.

            Args:
                data (list of dicts): Each dictionary represents a data point with keys as feature names.
            """
            self.data = []
            self.f_x = None

        def __len__(self):
            """Returns the number of samples in the dataset."""
            return len(self.data)

        def __getitem__(self, idx):
            """
            Retrieves a single sample from the dataset.

            Args:
                idx (int): Index of the sample.

            Returns:
                dict: A dictionary containing the features of the indexed sample.
            """
            return self.data[idx]
        
        def add_item(self, item):
            """
            Adds a new item (dictionary) to the dataset.

            Args:
                item (dict): A dictionary containing the new data sample.
            """
            if not isinstance(item, dict):
                raise ValueError("Item must be a dictionary.")
            self.data.append(item)
        
        def add_items(self, items):
            """
            Adds tensor as dataset.

            Args:
                items (dict): A dict containing data samples as tensors.
            """

            if not isinstance(items, dict):
                raise ValueError("Item must be a dict.")
            self.data= [dict(zip(items.keys(), values)) for values in zip(*items.values())]
        
        def add_embeddings(self, embeddings):
            """
            Adds embeddings to the dataset.

            Args:
                embeddings (tensor): A tensor containing the embeddings.
            """
            self.f_x  = embeddings



    
    train_percent=0.7
    test_percent = 0.2
    val_percent = 0.1

    # Number of points to sample
    num_samples = hyperparams['num_samples']
    COV = hyperparams['COV']
    MU = hyperparams['MU']
    BATCH_SIZE = hyperparams['BATCH_SIZE']
    num_classes = hyperparams['num_classes']
    d = hyperparams['d']
    input_size = hyperparams['input_size']
    
    
    one_hots = torch.eye(num_classes).float()

    train_dataset = SYNTHETIC()
    test_dataset = SYNTHETIC()
    val_dataset = SYNTHETIC()

    with torch.no_grad():
        # Create a random tensor of 3 dimensions with Gaussian distribution
        if not isinstance(d, int):
            random_tensors = []
            for _ in range(num_samples):
                tensor = torch.randn(d, variance=torch.eye(d)*10)
                random_tensors.append(tensor)
            samples = torch.stack(random_tensors)
        else:
            #Generate data and W
            mu_sample = np.full(input_size,MU)
            covariance_sample = np.diag(np.full(input_size ,COV))

            # Sample points
            samples = np.random.multivariate_normal(mu_sample, covariance_sample, num_samples)
            samples = torch.from_numpy(samples).float()
        '''
        #Define orthogonal vectors
        w1 = 3*torch.tensor([1,1,1])
        w2 = 3*torch.tensor([1,-1,0])
        w3 = 3*torch.tensor([1,1,-2])/math.sqrt(6)
        W = torch.stack([w1,w2,w3], dim = 0).t().float()
        ''' 
        W = make_full_rank_orthogonal_matrix(d, num_classes, scale=3.0)
        #assert_full_rank(W)
        assert_effective_complexity(W)



        #generate f_x
        # Instantiate the MLP (f(x))
        teacher_params = hyperparams.copy()
        teacher_params['embedding_size'] = d
        mlp_model = model_mlp(teacher_params)
        mlp_model.apply(initialize_weights)
        mlp_model.eval()


        #mlp_model.apply(initialize_weights)
        


        embedding_outputs = mlp_model(samples)
        f_x = mlp_model.get_fx(embedding_outputs)
        logits = f_x @ W

        print(f"[DATASET SYNTHETIC] num classes : {num_classes}, d: {d}, W shape: {W.shape}, f_x shape: {f_x.shape}, logits shape: {logits.shape}")
        # Define the softmax layer
        softmax = torch.nn.Softmax(dim=-1)
        distrib = softmax(logits)

        sum = torch.sum(distrib, dim=1)
        assert torch.all(sum > 1 - eps) and torch.all(sum < 1 + eps)

        #Shuffle sample tensor
        #torch.randperm?

        #Add to the dataset
        train_end_index = math.floor(num_samples*train_percent)
        train_dataset.add_items({'x':samples[:train_end_index, :] , 'label':distrib[:train_end_index, :]})

        val_start_index = train_end_index
        val_end_index = math.floor(num_samples*(train_percent+val_percent))
        val_dataset.add_items({'x':samples[val_start_index:val_end_index, :], 'label': distrib[val_start_index:val_end_index, :]})
        
        test_start_index = val_end_index
        test_dataset.add_items({'x': samples[test_start_index:, :], 'label':distrib[test_start_index:, :]})
        test_dataset.add_embeddings(f_x[test_start_index:, :])

        teacher_stats = check_diversity(mlp_model, f_x)

        print("[DATASET SYNTHETIC] \n=== Teacher diversity ===")
        print("[DATASET SYNTHETIC] Rank(W*):", teacher_stats["rank_W"])
        print("[DATASET SYNTHETIC] Rank(Cov(f*)):", teacher_stats["rank_embedding_cov"])


    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size = 128, shuffle = False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader

eps = 1e-6
l1_lambda = 1e-4
l2_lambda = 1e-4


def elastic_net_regularization(model, config):
    l1 = config["l1_weight"]
    l2 = config["l2_weight"]
    return (
        l1 * l1_lambda * sum(p.abs().sum() for p in model.parameters()) +
        l2 * l2_lambda * sum((p ** 2).sum() for p in model.parameters())
    )



def train_loop(train_loader, model, criterion, optimizer, device = 'cpu', config=None):
    model.train()
    logsoftmax = nn.LogSoftmax(dim=-1)
    total_loss = 0
    accuracy = 0
    total = 0
    correct = 0

    num_classes = model.get_W().shape[0]
    one_hots = torch.eye(num_classes).float()


    for batch in train_loader:
        optimizer.zero_grad()
        labels = batch['label'].to(device)
        inputs = batch['x'].to(device)
        embedding_output = model(inputs)
        #Compute f_x from output
        f_x = model.get_fx(embedding_output)
        #Compute unenmbed from output
        #unembedding = model.get_unembeddings(one_hots)
        #logits = torch.matmul(f_x, unembedding)
        W = model.get_W().to(device)
        logits = f_x @ W.T      # (N, d) @ (d, C) = (N, C)



        outputs = logsoftmax(logits)
        #MINIMIZE KL DIVERGENCE
        loss = criterion(outputs, labels)
        #loss += elastic_net_regularization(model, config)

        total_loss += loss.item()
        #Accuracy computation
        #_, predicted = torch.max(outputs, 1)
        #correct += (predicted == labels).sum().item()
        #total += labels.size(0)

        # Backward pass
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    #accuracy = correct / total
    return avg_loss, accuracy



def val_loop(
    val_loader,
    model,
    criterion,
    device='cpu'
):
    model.eval()
    total_kl = 0.0
    logsoftmax = nn.LogSoftmax(dim=-1)

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['x'].to(device)
            labels = batch['label'].to(device)

            outputs = model(inputs)

            f_x = model.get_fx(outputs)
            W = model.get_W().to(device)
            logits = f_x @ W.T

            log_probs = logsoftmax(logits)

            # PURE KL (no regularization)
            kl = criterion(log_probs, labels)
            total_kl += kl.item()

    avg_kl = total_kl / len(val_loader)
    return avg_kl



def test_loop(
    test_loader,
    model,
    criterion,
    device='cpu',
    kl_threshold=5e-4
):
    """
    Computes PURE KL divergence on test set (no regularization)
    and checks convergence.

    Returns:
        avg_kl: float
        converged: bool
        embeddings: list[tensor]
        predicted_distrib: list[tensor]
    """

    model.eval()
    total_kl = 0.0
    logsoftmax = nn.LogSoftmax(dim=-1)
    softmax = nn.Softmax(dim=-1)

    outputs_to_return = []
    total_fx = []

    with torch.no_grad():

        for batch in test_loader:
            inputs = batch['x'].to(device)
            labels = batch['label'].to(device)

            # forward
            outputs = model(inputs)

            # embeddings
            f_x = model.get_fx(outputs)
            total_fx.append(f_x)

            # logits
            W = model.get_W().to(device)
            logits = f_x @ W.T

            # probabilities
            log_probs = logsoftmax(logits)
            probs = softmax(logits)
            outputs_to_return.append(probs)

            # PURE KL (NO REGULARIZATION)
            kl = criterion(log_probs, labels)
            total_kl += kl.item()

    avg_kl = total_kl / len(test_loader)

    # ---- CONVERGENCE CHECK ----
    converged = avg_kl < kl_threshold

    return avg_kl, converged, total_fx, outputs_to_return

            

ARCHITECTURES = {
    "mlp_small":  {
        "arch_type": "mlp",
        "hidden_sizes": [64,32,64],
    },
    "mlp_medium": {
        "arch_type": "mlp",
        "hidden_sizes": [256, 128, 128, 256],
    },
    "mlp_large":  {
        "arch_type": "mlp",
        "hidden_sizes": [512,1024, 1024, 512],
    },
    "res_small":  {
        "arch_type": "resmlp",
        "hidden_sizes": [256,64,256],
    },
    "res_medium": {
        "arch_type": "resmlp",
        "hidden_sizes": [256, 128, 128, 256],
    },
    "res_large":  {
        "arch_type": "resmlp",
        "hidden_sizes": [512,1024, 1024, 512],
    },

}

OPTIMIZERS = {
    "adam": lambda params, lr: optim.Adam(params, lr=lr),
    "sgd":  lambda params, lr: optim.SGD(params, lr=lr, momentum=0.9),
    "adamw": lambda params, lr: optim.AdamW(params, lr=lr)
}

TEACHER_D_VALUES = [2, 3, 5, 10]
STUDENT_EMBEDDING_VALUES = [2, 3, 5, 10]

REGULARIZATION_CONFIGS = [
    {"l1_weight": 0, "l2_weight": 0},           # none
    {"l1_weight": 1, "l2_weight": 0},           # L1
    {"l1_weight": 0, "l2_weight": 1},           # L2
    {"l1_weight": 1, "l2_weight": 1},           # elastic net
]


def run_single_experiment(config, train_loader, val_loader, test_loader, run_timestamp=None):
    
    set_seed(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # ---------- MODEL ----------
    model = test_model = build_model(config).to(device)
    model.apply(initialize_weights)

    criterion = nn.KLDivLoss(reduction='batchmean')
    opt_name = config["optimizer"]
    optimizer = OPTIMIZERS[opt_name](model.parameters(), config['lr'])


    epochs = config['epochs']
    check_interval = config['check_interval']
    embedding_size = config['embedding_size']

    # ---------- FOLDERS ----------
    run_name = f"{config['architecture']}_SEED{config['seed']}"

    plots_folder_path = os.path.join(
        "PLOTS",
        f"d{config['d']}_C{config['num_classes']}",
        run_timestamp,
        run_name
    )

    checkpoints_folder_path = os.path.join(
        "CHECKPOINTS",
        f"d{config['d']}_C{config['num_classes']}",
        run_timestamp,
        run_name
    )


    os.makedirs(plots_folder_path, exist_ok=False)
    os.makedirs(checkpoints_folder_path, exist_ok=False)

    # ---------- WANDB ----------
    wandb.init(
    project="effective-complexity-v2",
    name=run_name,
    group=f"d{config['d']}_C{config['num_classes']}_{run_timestamp}",
    config=config
)

    with open(os.path.join(plots_folder_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


    best_val_loss = float('inf')
    best_epoch = 0
    train_loss_over_epochs = []
    val_loss_over_epochs = []
    pca_logs = []
    KL_THRESHOLD = 1e-4
    patience = 15
    wait = 0
    delta = 1e-6


    print("\nTraining...")
    pbar = tqdm(range(epochs))

    for epoch in pbar:

        train_loss, _ = train_loop(train_loader, model, criterion, optimizer, device, config)
        
        # ---- VALIDATION KL ----
        val_kl = val_loop(
            val_loader, model, criterion, device
        )

        pbar.set_postfix({
            "val_KL": f"{val_kl:.2e}"
        })

        # ---- EARLY STOPPING (plateau) ----
        if best_val_loss - val_kl > delta:
            best_val_loss = val_kl
            best_epoch = epoch
            wait = 0

            torch.save(
                model.state_dict(),
                os.path.join(checkpoints_folder_path, "best_model.pth")
            )

        else:
            wait += 1

        if wait >= patience:
            print(
                f"[EARLY STOPPING] "
                f"epoch={epoch} "
                f"val_loss={val_kl:.6e}"
            )
            break


        train_loss_over_epochs.append(train_loss)
        val_loss_over_epochs.append(val_kl)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_kl
        })

        if epoch % check_interval == 0:
            test_loss, converged, embeddings, predicted_distrib = test_loop(
                test_loader,
                model,
                criterion,
                device,
            )

            variance, errors = analyze_embeddings(
                embeddings,
                predicted_distrib,
                test_loader.dataset.f_x,
                plots_folder_path,
                embedding_size,
                config['num_classes'],
                epoch_for_logging=epoch // check_interval
            )
            # 🔵 LOG ALL PCA VARIANCE VALUES
            for k, v in enumerate(variance, start=1):
                pca_logs.append({
                    "architecture": config["architecture"],
                    "seed": config["seed"],
                    "epoch": epoch,
                    "components": k,
                    "variance_explained": float(v)
                })

  

    model.load_state_dict(
        torch.load(os.path.join(checkpoints_folder_path, "best_model.pth"))
    )

    # ---------- FINAL TEST ----------
    test_loss, _, embeddings, predicted_distrib = test_loop(test_loader, model, criterion, device)

    # ---- DISTRIBUTION EQUALITY CHECK ----
    student_probs = torch.cat(predicted_distrib)
    teacher_probs = test_loader.dataset.data

    # ===== SANITY CHECK =====
    F = test_loader.dataset.f_x.detach()

    cca_self = run_cca(F, F)

    print("\n=== SANITY CHECK: teacher vs teacher ===")
    for i, c in enumerate(cca_self):
        print(f"CCA dim {i+1}: {c:.6f}")

    align_self = linear_alignment_error(F, F)
    print("Self-alignment MSE:", align_self.item())
    # ===== END SANITY CHECK =====


    # rebuild teacher distribution tensor
    teacher_probs = torch.stack([d["label"] for d in test_loader.dataset])
    logsoftmax = nn.LogSoftmax(dim=-1)
    log_student_probs = logsoftmax(student_probs)
    kl_test = criterion(log_student_probs, teacher_probs)

    print("\n=== Distribution equality check ===")
    print("Test KL(teacher || student):", kl_test.item())


    #assert kl_test < 1, "Model not converged!"

    wandb.log({
        "test_KL": kl_test.item()
    })


    # ---- DIVERSITY CHECK ----
    emb = torch.cat(embeddings)

    div_stats = check_diversity(model, emb)

    print("\n=== Diversity diagnostics ===")
    print("Rank(W):", div_stats["rank_W"])
    print("Singular values W:", div_stats["singular_values_W"])
    print("Rank(Cov(f(x))):", div_stats["rank_embedding_cov"])
    print("Eigenvalues Cov(f(x)):", div_stats["embedding_cov_eigvals"])

    wandb.log({
        "rank_W": div_stats["rank_W"],
        "rank_embedding_cov": div_stats["rank_embedding_cov"]
    })

    # ---- CASE 3: LINEAR ALIGNMENT TEST ----

    F_student = torch.cat(embeddings).detach()
    F_teacher = test_loader.dataset.f_x.detach()

    # ---- CCA ANALYSIS ----
    cca_corrs = run_cca(F_teacher, F_student)

    print("\n=== CCA results ===")
    for i, c in enumerate(cca_corrs):
        print(f"CCA dim {i+1}: {c:.4f}")

    # log to wandb
    for i, c in enumerate(cca_corrs):
        wandb.log({f"CCA/corr_dim_{i+1}": c})

    '''
    plt.figure()
    plt.plot(cca_corrs, marker='o')
    plt.xlabel("CCA component")
    plt.ylabel("Correlation")
    plt.title("Canonical Correlations (Teacher vs Student)")
    plt.grid()
    plt.savefig(os.path.join(plots_folder_path, "CCA_correlations.png"))
    plt.close()
    '''


    # ---- RESIDUAL CCA TEST ----
    k = 3  # expected effective complexity

    residual_corrs = residual_cca_test(
        F_teacher,
        F_student,
        k
    )

    for i, c in enumerate(residual_corrs):
        wandb.log({f"Residual_CCA/corr_dim_{i+1}": c})

    # ---- PROJECTED LINEAR ALIGNMENT (∼EL IDENTIFIABILITY) ----
    corrs, U_t, U_s = cca_with_directions(
        F_teacher,
        F_student,
        k
    )

    #----------Plot CCA directions----------
    if config['seed'] == 0:
        U, V = get_canonical_variables(F_teacher, F_student, k)
        for i in range(k):
            fname = plot_cca_with_fit_and_save(
                U, V, i,
                save_dir=plots_folder_path,
                prefix="cca_fit"
            )
            wandb.log({
                f"CCA/scatter_dim_{i+1}": wandb.Image(fname)
            })

    #--------------------------------------

    align_err_proj = linear_alignment_error_projected(
        F_teacher,
        F_student,
        U_t,
        U_s
    )

    print("\n=== Projected linear alignment test ===")
    print("Projected alignment MSE:", align_err_proj.item())

    wandb.log({
        "projected_linear_alignment_MSE": align_err_proj.item()
    })



    print(f"\nFinal test loss: {test_loss:.4f}")

    plt.plot(train_loss_over_epochs, label="Train Loss")
    plt.plot(val_loss_over_epochs, label="Val Loss")
    plt.legend(); plt.grid()
    plt.savefig(os.path.join(plots_folder_path, "Loss_over_epochs.png"))
    plt.close()

    ''''
    variance, errors = analyze_embeddings(
        embeddings,
        predicted_distrib,
        test_loader.dataset.f_x,
        plots_folder_path,
        embedding_size,
        config['num_classes']
    )
    # 🔵 LOG ALL PCA VARIANCE VALUES
    for k, v in enumerate(variance, start=1):
        pca_logs.append({
            "architecture": config["architecture"],
            "seed": config["seed"],
            "epoch": epoch,
            "components": k,
            "variance_explained": float(v)
        })
        wandb.log({
            "epoch": epoch,
            f"pca_variance/components_{k}": v
        })
    '''

    wandb.finish()


    df = pd.DataFrame(pca_logs)
    df.to_csv(os.path.join(plots_folder_path, "pca_variance_log.csv"), index=False)


    return {
        "architecture": config['architecture'],
        "seed": config['seed'],
        "test_loss": test_loss,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "plots_folder": plots_folder_path,
        "cca_corrs": cca_corrs,
        "residual_cca_corrs": residual_corrs,
        "align_err_proj": align_err_proj.item(),
        "variance_explained": variance,
        "errors": errors,
        "rank_W": div_stats["rank_W"],
        "rank_embedding_cov": div_stats["rank_embedding_cov"]
    }


def run_multiple_experiments(base_hyperparams, seeds, architectures,
                             train_loader, val_loader, test_loader, run_timestamp):

    results = []

    for arch_name, arch_params in architectures.items():
        for seed in seeds:

            print(f"\n=== RUNNING {arch_name} — seed={seed} ===")

            config = base_hyperparams.copy()

            # student architecture params (hidden sizes, arch_type, etc)
            config.update(arch_params)

            # add THESE TWO
            config["architecture"] = arch_name   
            config["seed"] = seed                

            result = run_single_experiment(
                config,
                train_loader,
                val_loader,
                test_loader,
                run_timestamp=run_timestamp,
            )

            results.append(result)

    return results


def run_ablation_experiments(base_hyperparams, seeds, architectures):

    results = []

    for arch_name, arch_params in architectures.items():
        for seed in seeds:
            for d in [5]:
                for C in [5]:

                    config = base_hyperparams.copy()
                    config.update(arch_params)

                    config.update({
                        "architecture": arch_name,
                        "seed": seed,
                        "d": d,
                        "input_size": d,
                        "num_classes": C
                    })


                    # IMPORTANT: regenerate dataset
                    train_loader, val_loader, test_loader = \
                        dataset_synthetic(config)


                    result = run_single_experiment(
                        config,
                        train_loader,
                        val_loader,
                        test_loader,
                        run_timestamp=None,
                    )

                    results.append(result)

    return results



def average_results_across_seeds(results, save_dir="RESULTS"):
    os.makedirs(save_dir, exist_ok=True)

    by_arch = defaultdict(list)
    for r in results:
        by_arch[r["architecture"]].append(r)

    averaged = {}

    for arch, runs in by_arch.items():
        metrics = defaultdict(list)

        for r in runs:
            for k, v in r.items():
                if k not in ["architecture", "seed", "plots_folder"]:
                    metrics[k].append(v)

        avg = {
            "architecture": arch,
            "num_seeds": len(runs),
        }

        for k, v in metrics.items():
            if k in ["cca_corrs","residual_cca_corrs","errors","variance_explained"]:
                # list of lists
                v_stacked = np.stack(v, axis=0)   # shape (num_seeds, dim)
                mean_vals = np.mean(v_stacked, axis=0).tolist()
                std_vals = np.std(v_stacked, axis=0).tolist()

                avg[f"{k}_mean"] = mean_vals
                avg[f"{k}_std"] = std_vals
            else:
                avg[f"{k}_mean"] = float(np.mean(v))
                avg[f"{k}_std"]  = float(np.std(v))

        averaged[arch] = avg

        with open(os.path.join(save_dir, f"{arch}_avg.json"), "w") as f:
            json.dump(avg, f, indent=2)

    return averaged

'''
# 1) set seed so dataset is reproducible
set_seed(0)

# 2) generate dataset ONCE
train_loader, val_loader, test_loader = dataset_synthetic(hyperparams)

# 3) run multiple experiments
seeds = [0,1,2,3,4]

results = run_multiple_experiments(
    base_hyperparams = hyperparams,
    seeds = seeds,
    architectures = ARCHITECTURES,
    train_loader = train_loader,f
    val_loader = val_loader,
    test_loader = test_loader
)

print(results)


results = run_ablation_experiments(
    base_hyperparams=hyperparams,
    seeds=[0],
    architectures=ARCHITECTURES
)
'''
TEACHER_REPRESENTATIONS_D =[5,10]
NUM_CLASSES = [5]
DATASET_SEED = 0
for d in TEACHER_REPRESENTATIONS_D:
    for C in NUM_CLASSES:
        config = hyperparams.copy()
        config.update({"d": d, "input_size": d, "num_classes": C})
        set_seed(DATASET_SEED)
        dataset_tag = f"d{d}_C{C}"
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


        train_loader, val_loader, test_loader = dataset_synthetic(config)

        SEEDS = [0, 1, 2, 3, 4]

        results = run_multiple_experiments(
            base_hyperparams=config,
            seeds=SEEDS,
            architectures=ARCHITECTURES,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            run_timestamp=run_timestamp,
        )

        averaged_results = average_results_across_seeds(results, save_dir=os.path.join("RESULTS", dataset_tag,datetime.now().strftime("%Y%m%d_%H%M%S")))
        plot_averaged_metrics(
            averaged_results,
            save_dir=os.path.join("FIGURES", dataset_tag, datetime.now().strftime("%Y%m%d_%H%M%S")),
            max_components=10   # or None
        )

        print("\n=== AVERAGED RESULTS ===")
        for arch, res in averaged_results.items():
            print(arch, res)

