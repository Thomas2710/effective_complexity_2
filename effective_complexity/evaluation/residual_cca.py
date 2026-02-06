import numpy as np
from effective_complexity.evaluation.cca import (
    run_cca,
    cca_with_directions,
)
from effective_complexity.evaluation.alignment import project_out


def residual_cca(F_teacher, F_student, k):
    """
    Run CCA, remove first k canonical directions,
    then run CCA again on residuals.
    """

    _, U_t, U_s = cca_with_directions(F_teacher, F_student, k)

    Ft_res = project_out(F_teacher, U_t)
    Fs_res = project_out(F_student, U_s)

    return run_cca(Ft_res, Fs_res)
