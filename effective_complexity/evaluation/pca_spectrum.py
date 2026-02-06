import numpy as np
from effective_complexity.evaluation.pca import apply_pca


def pca_spectrum(data):
    """
    Returns:
        variance_explained[k]
        reconstruction_error[k]
    """

    d = data.shape[1]

    variances = []
    errors = []

    for k in range(1, d + 1):
        _, reconstructed, var = apply_pca(data, k)
        mse = np.mean((data - reconstructed) ** 2)

        variances.append(var[-1])
        errors.append(mse)

    return np.array(variances), np.array(errors)
