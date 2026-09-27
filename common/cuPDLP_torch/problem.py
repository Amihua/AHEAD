
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch



@dataclass
class QuadraticProgrammingProblem:
    variable_lower_bound: np.ndarray
    variable_upper_bound: np.ndarray
    objective_vector: np.ndarray
    objective_constant: float
    constraint_matrix: sp.csr_matrix
    right_hand_side: np.ndarray
    num_equalities: int

    @property
    def num_variables(self) -> int:
        return int(self.objective_vector.shape[0])

    @property
    def num_constraints(self) -> int:
        return int(self.constraint_matrix.shape[0])

    def copy(self) -> "QuadraticProgrammingProblem":
        return copy.deepcopy(self)


@dataclass
class ScaledQpProblem:
    original_qp: QuadraticProgrammingProblem
    scaled_qp: QuadraticProgrammingProblem
    constraint_rescaling: np.ndarray
    variable_rescaling: np.ndarray



@dataclass
class GpuLinearProgrammingProblem:
    num_variables: int
    num_constraints: int
    num_equalities: int
    variable_lower_bound: torch.Tensor
    variable_upper_bound: torch.Tensor
    isfinite_variable_lower_bound: torch.Tensor
    isfinite_variable_upper_bound: torch.Tensor
    objective_vector: torch.Tensor
    objective_constant: float
    constraint_matrix: torch.Tensor
    constraint_matrix_t: torch.Tensor
    right_hand_side: torch.Tensor


@dataclass
class ScaledGpuProblem:
    original_gpu_problem: GpuLinearProgrammingProblem
    scaled_gpu_problem: GpuLinearProgrammingProblem
    constraint_rescaling: torch.Tensor
    variable_rescaling: torch.Tensor



def _scipy_csr_to_torch_csr(
    A: sp.csr_matrix,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    A = A.tocsr().astype(np.float64)
    crow = torch.tensor(A.indptr, dtype=torch.int32, device=device)
    col  = torch.tensor(A.indices, dtype=torch.int32, device=device)
    val  = torch.tensor(A.data, dtype=dtype, device=device)
    return torch.sparse_csr_tensor(crow, col, val, size=tuple(A.shape),
                                   dtype=dtype, device=device)


def qp_to_gpu(
    problem: QuadraticProgrammingProblem,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> GpuLinearProgrammingProblem:
    A = problem.constraint_matrix
    A_t = A.T.tocsr()

    return GpuLinearProgrammingProblem(
        num_variables=problem.num_variables,
        num_constraints=problem.num_constraints,
        num_equalities=problem.num_equalities,
        variable_lower_bound=torch.tensor(
            problem.variable_lower_bound, dtype=dtype, device=device),
        variable_upper_bound=torch.tensor(
            problem.variable_upper_bound, dtype=dtype, device=device),
        isfinite_variable_lower_bound=torch.tensor(
            np.isfinite(problem.variable_lower_bound), dtype=torch.bool, device=device),
        isfinite_variable_upper_bound=torch.tensor(
            np.isfinite(problem.variable_upper_bound), dtype=torch.bool, device=device),
        objective_vector=torch.tensor(
            problem.objective_vector, dtype=dtype, device=device),
        objective_constant=float(problem.objective_constant),
        constraint_matrix=_scipy_csr_to_torch_csr(A, device, dtype),
        constraint_matrix_t=_scipy_csr_to_torch_csr(A_t, device, dtype),
        right_hand_side=torch.tensor(
            problem.right_hand_side, dtype=dtype, device=device),
    )


def scaled_qp_to_gpu(
    scaled: ScaledQpProblem,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> ScaledGpuProblem:
    return ScaledGpuProblem(
        original_gpu_problem=qp_to_gpu(scaled.original_qp, device, dtype),
        scaled_gpu_problem=qp_to_gpu(scaled.scaled_qp, device, dtype),
        constraint_rescaling=torch.tensor(
            scaled.constraint_rescaling, dtype=dtype, device=device),
        variable_rescaling=torch.tensor(
            scaled.variable_rescaling, dtype=dtype, device=device),
    )



def spmv(A_csr: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.mv(A_csr, x)
