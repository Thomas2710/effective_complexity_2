# effective_complexity/evaluation/cca.py

import numpy as np
import torch
from sklearn.cross_decomposition import CCA


def run_cca(F_teacher, F_student, max_components=None):
    Ft = F_teacher.cpu().numpy()
    Fs = F_student.cpu().numpy()

    if max_components is None:
        max_components = min(Ft.shape[1], Fs.shape[1])

    cca = CCA(n_components=max_components)
    Xc, Yc = cca.fit_transform(Ft, Fs)

    corrs = [
        np.corrcoef(Xc[:, i], Yc[:, i])[0, 1]
        for i in range(max_components)
    ]

    return np.array(corrs)


def cca_with_directions(F_teacher, F_student, k):
    Ft = F_teacher.cpu().numpy()
    Fs = F_student.cpu().numpy()

    cca = CCA(n_components=k)
    Xc, Yc = cca.fit_transform(Ft, Fs)

    corrs = [
        np.corrcoef(Xc[:, i], Yc[:, i])[0, 1]
        for i in range(k)
    ]

    return np.array(corrs), cca.x_weights_, cca.y_weights_


def get_canonical_variables(F_teacher, F_student, k):
    Ft = F_teacher.cpu().numpy()
    Fs = F_student.cpu().numpy()

    cca = CCA(n_components=k)
    U, V = cca.fit_transform(Ft, Fs)

    return U, V
