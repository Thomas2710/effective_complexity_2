# effective_complexity/evaluation/alignment.py

import torch
import numpy as np


def linear_alignment_error(F_teacher, F_student):
    A = torch.linalg.lstsq(F_teacher, F_student).solution
    pred = F_teacher @ A
    return torch.mean((pred - F_student) ** 2)


def project_out(F, U):
    U = torch.from_numpy(U).to(F.device).float()
    return F - (F @ U) @ U.T


def project_to_subspace(F, U):
    U = torch.from_numpy(U).to(F.device).float()
    return F @ U


def linear_alignment_error_projected(F_teacher, F_student, U_t, U_s):
    Ft = project_to_subspace(F_teacher, U_t)
    Fs = project_to_subspace(F_student, U_s)

    A = torch.linalg.lstsq(Ft, Fs).solution
    pred = Ft @ A
    return torch.mean((pred - Fs) ** 2)
