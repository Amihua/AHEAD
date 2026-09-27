
from __future__ import annotations

import copy

import numpy as np
import scipy.sparse as sp

from .problem import QuadraticProgrammingProblem, ScaledQpProblem



def validate(p: QuadraticProgrammingProblem) -> None:
    n = p.num_variables
    m = p.num_constraints
    errors = []

    if len(p.variable_upper_bound) != n:
        errors.append("variable_upper_bound size mismatch")
    if p.constraint_matrix.shape[1] != n:
        errors.append("constraint_matrix columns != num_variables")
    if len(p.right_hand_side) != m:
        errors.append("right_hand_side size mismatch")
    if np.any(p.variable_lower_bound == np.inf):
        errors.append("variable_lower_bound contains +Inf")
    if np.any(p.variable_upper_bound == -np.inf):
        errors.append("variable_upper_bound contains -Inf")
    if np.any(np.isnan(p.variable_lower_bound)) or np.any(np.isnan(p.variable_upper_bound)):
        errors.append("NaN found in variable bounds")
    if not np.all(np.isfinite(p.right_hand_side)):
        errors.append("NaN/Inf found in right_hand_side")
    if not np.all(np.isfinite(p.objective_vector)):
        errors.append("NaN/Inf found in objective_vector")

    A_data = p.constraint_matrix.data
    if len(A_data) > 0 and not np.all(np.isfinite(A_data)):
        errors.append("NaN/Inf found in constraint_matrix")

    if errors:
        raise ValueError("Invalid QuadraticProgrammingProblem:\n  " + "\n  ".join(errors))



def scale_problem(
    problem: QuadraticProgrammingProblem,
    constraint_rescaling: np.ndarray,
    variable_rescaling: np.ndarray,
) -> None:
    assert np.all(constraint_rescaling > 0)
    assert np.all(variable_rescaling > 0)

    D_inv = 1.0 / variable_rescaling
    E_inv = 1.0 / constraint_rescaling

    problem.objective_vector   *= D_inv
    problem.variable_lower_bound *= variable_rescaling
    problem.variable_upper_bound *= variable_rescaling
    problem.right_hand_side    *= E_inv
    problem.constraint_matrix   = (
        sp.diags(E_inv) @ problem.constraint_matrix @ sp.diags(D_inv)
    ).tocsr()



def _sparse_abs(A: sp.csr_matrix) -> sp.csr_matrix:
    B = A.copy()
    B.data = np.abs(B.data)
    return B


def _l2_norm_rows(A: sp.csr_matrix) -> np.ndarray:
    Aabs = _sparse_abs(A)
    scale = np.asarray(Aabs.max(axis=1).todense()).ravel()
    scale[scale == 0] = 1.0
    scaled = sp.diags(1.0 / scale) @ A
    return scale * np.sqrt(np.asarray(scaled.power(2).sum(axis=1)).ravel())


def _l2_norm_cols(A: sp.csr_matrix) -> np.ndarray:
    Aabs = _sparse_abs(A)
    scale = np.asarray(Aabs.max(axis=0).todense()).ravel()
    scale[scale == 0] = 1.0
    scaled = A @ sp.diags(1.0 / scale)
    return scale * np.sqrt(np.asarray(scaled.power(2).sum(axis=0)).ravel())



def ruiz_rescaling(
    problem: QuadraticProgrammingProblem,
    num_iterations: int,
    p: float = np.inf,
) -> tuple[np.ndarray, np.ndarray]:
    m, n = problem.constraint_matrix.shape
    cum_con = np.ones(m)
    cum_var = np.ones(n)

    for _ in range(num_iterations):
        A = problem.constraint_matrix

        if p == np.inf:
            Aabs = _sparse_abs(A)
            var_rescale = np.sqrt(np.asarray(Aabs.max(axis=0).todense()).ravel())
            con_rescale = np.sqrt(np.asarray(Aabs.max(axis=1).todense()).ravel()) if m > 0 else np.ones(0)
        else:
            assert p == 2
            col_norms = _l2_norm_cols(A)
            row_norms = _l2_norm_rows(A)
            target_row_norm = np.sqrt(n / m) if m > 0 else 1.0
            var_rescale = np.sqrt(np.sqrt(col_norms ** 2))
            con_rescale = np.sqrt(row_norms / target_row_norm)

        var_rescale[var_rescale == 0] = 1.0
        if m > 0:
            con_rescale[con_rescale == 0] = 1.0

        scale_problem(problem, con_rescale, var_rescale)
        cum_con *= con_rescale
        cum_var *= var_rescale

    return cum_con, cum_var



def l2_norm_rescaling(
    problem: QuadraticProgrammingProblem,
) -> tuple[np.ndarray, np.ndarray]:
    A = problem.constraint_matrix
    row_norms = _l2_norm_rows(A)
    col_norms = _l2_norm_cols(A)
    row_norms[row_norms == 0] = 1.0
    col_norms[col_norms == 0] = 1.0
    row_rescale = np.sqrt(row_norms)
    col_rescale = np.sqrt(col_norms)
    scale_problem(problem, row_rescale, col_rescale)
    return row_rescale, col_rescale



def pock_chambolle_rescaling(
    problem: QuadraticProgrammingProblem,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    assert 0 <= alpha <= 2
    A = problem.constraint_matrix

    A_abs = A.copy()
    A_abs.data = np.abs(A_abs.data)

    A_var = A_abs.copy()
    A_var.data = A_var.data ** (2 - alpha)
    var_rescale = np.sqrt(np.asarray(A_var.sum(axis=0)).ravel())

    A_con = A_abs.copy()
    A_con.data = A_con.data ** alpha
    con_rescale = np.sqrt(np.asarray(A_con.sum(axis=1)).ravel())

    var_rescale[var_rescale == 0] = 1.0
    con_rescale[con_rescale == 0] = 1.0

    scale_problem(problem, con_rescale, var_rescale)
    return con_rescale, var_rescale



def rescale_problem(
    l_inf_ruiz_iterations: int,
    l2_norm_rescaling_flag: bool,
    pock_chambolle_alpha,
    verbosity: int,
    original_problem: QuadraticProgrammingProblem,
) -> ScaledQpProblem:
    problem = original_problem.copy()
    m, n = problem.constraint_matrix.shape
    constraint_rescaling = np.ones(m)
    variable_rescaling   = np.ones(n)

    if l_inf_ruiz_iterations > 0:
        con, var = ruiz_rescaling(problem, l_inf_ruiz_iterations, np.inf)
        constraint_rescaling *= con
        variable_rescaling   *= var

    if l2_norm_rescaling_flag:
        con, var = l2_norm_rescaling(problem)
        constraint_rescaling *= con
        variable_rescaling   *= var

    if pock_chambolle_alpha is not None:
        con, var = pock_chambolle_rescaling(problem, float(pock_chambolle_alpha))
        constraint_rescaling *= con
        variable_rescaling   *= var

    if verbosity >= 3:
        A = problem.constraint_matrix
        print(f"  After rescaling: ||A||_F={np.linalg.norm(A.data):.4g}  "
              f"||c||_2={np.linalg.norm(problem.objective_vector):.4g}  "
              f"||b||_2={np.linalg.norm(problem.right_hand_side):.4g}")

    return ScaledQpProblem(
        original_qp=original_problem,
        scaled_qp=problem,
        constraint_rescaling=constraint_rescaling,
        variable_rescaling=variable_rescaling,
    )
