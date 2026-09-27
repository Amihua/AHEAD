
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from .problem import QuadraticProgrammingProblem
from .solve_log import (
    ConvergenceInformation,
    IterationStats,
    TerminationReason,
)


class OptimalityNorm(Enum):
    L_INF = 1
    L2 = 2


@dataclass
class TerminationCriteria:
    optimality_norm: OptimalityNorm = OptimalityNorm.L2
    eps_optimal_absolute: float = 1e-6
    eps_optimal_relative: float = 1e-6
    eps_primal_infeasible: float = 1e-8
    eps_dual_infeasible: float = 1e-8
    time_sec_limit: float = float("inf")
    iteration_limit: int = 10_000_000
    kkt_matrix_pass_limit: float = float("inf")


@dataclass
class CachedQuadraticProgramInfo:
    l_inf_norm_c: float
    l2_norm_c: float
    l_inf_norm_b: float
    l2_norm_b: float


def cached_quadratic_program_info(
    problem: QuadraticProgrammingProblem,
) -> CachedQuadraticProgramInfo:
    c = problem.objective_vector
    b = problem.right_hand_side
    return CachedQuadraticProgramInfo(
        l_inf_norm_c=float(np.max(np.abs(c))) if len(c) > 0 else 0.0,
        l2_norm_c=float(np.linalg.norm(c)),
        l_inf_norm_b=float(np.max(np.abs(b))) if len(b) > 0 else 0.0,
        l2_norm_b=float(np.linalg.norm(b)),
    )



def _optimality_criteria_met(
    criteria: TerminationCriteria,
    qp_cache: CachedQuadraticProgramInfo,
    conv_info: ConvergenceInformation,
) -> bool:
    eps_abs = criteria.eps_optimal_absolute
    eps_rel = criteria.eps_optimal_relative

    if criteria.optimality_norm == OptimalityNorm.L_INF:
        primal_tol = eps_abs + eps_rel * qp_cache.l_inf_norm_b
        dual_tol   = eps_abs + eps_rel * qp_cache.l_inf_norm_c
        primal_ok  = conv_info.l_inf_primal_residual <= primal_tol
        dual_ok    = conv_info.l_inf_dual_residual   <= dual_tol
    else:
        primal_tol = eps_abs + eps_rel * qp_cache.l2_norm_b
        dual_tol   = eps_abs + eps_rel * qp_cache.l2_norm_c
        primal_ok  = conv_info.l2_primal_residual <= primal_tol
        dual_ok    = conv_info.l2_dual_residual   <= dual_tol

    p_obj = conv_info.primal_objective
    d_obj = conv_info.dual_objective
    abs_obj = abs(p_obj) + abs(d_obj)
    if not math.isfinite(abs_obj):
        gap_ok = False
    else:
        gap     = abs(p_obj - d_obj)
        gap_ok  = gap <= eps_abs + eps_rel * abs_obj

    return primal_ok and dual_ok and gap_ok


def _primal_infeasibility_criteria_met(
    criteria: TerminationCriteria,
    conv_info: ConvergenceInformation,
) -> bool:
    return False


def _dual_infeasibility_criteria_met(
    criteria: TerminationCriteria,
    conv_info: ConvergenceInformation,
) -> bool:
    return False



def check_termination_criteria(
    criteria: TerminationCriteria,
    qp_cache: CachedQuadraticProgramInfo,
    iteration_stats: IterationStats,
) -> Optional[TerminationReason]:
    for conv_info in iteration_stats.convergence_information:
        if _optimality_criteria_met(criteria, qp_cache, conv_info):
            return TerminationReason.OPTIMAL

    if iteration_stats.iteration_number > criteria.iteration_limit:
        return TerminationReason.ITERATION_LIMIT
    if iteration_stats.cumulative_kkt_matrix_passes > criteria.kkt_matrix_pass_limit:
        return TerminationReason.KKT_MATRIX_PASS_LIMIT
    if iteration_stats.cumulative_time_sec > criteria.time_sec_limit:
        return TerminationReason.TIME_LIMIT

    return None
