
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class TerminationReason(Enum):
    UNSPECIFIED = 0
    OPTIMAL = 1
    PRIMAL_INFEASIBLE = 2
    DUAL_INFEASIBLE = 3
    TIME_LIMIT = 4
    ITERATION_LIMIT = 5
    KKT_MATRIX_PASS_LIMIT = 6
    NUMERICAL_ERROR = 7
    INVALID = 8
    OTHER = 9


class PointType(Enum):
    POINT_TYPE_UNSPECIFIED = 0
    POINT_TYPE_CURRENT_ITERATE = 1
    POINT_TYPE_AVERAGE_ITERATE = 2
    POINT_TYPE_ITERATE_DIFFERENCE = 3


class RestartChoice(Enum):
    RESTART_CHOICE_UNSPECIFIED = 0
    RESTART_CHOICE_NO_RESTART = 1
    RESTART_CHOICE_WEIGHTED_AVERAGE_RESET = 2
    RESTART_CHOICE_RESTART_TO_AVERAGE = 3


@dataclass
class ConvergenceInformation:
    candidate_type: PointType = PointType.POINT_TYPE_UNSPECIFIED
    primal_objective: float = 0.0
    dual_objective: float = 0.0
    corrected_dual_objective: float = 0.0
    l_inf_primal_residual: float = 0.0
    l2_primal_residual: float = 0.0
    l_inf_dual_residual: float = 0.0
    l2_dual_residual: float = 0.0
    relative_l_inf_primal_residual: float = 0.0
    relative_l2_primal_residual: float = 0.0
    relative_l_inf_dual_residual: float = 0.0
    relative_l2_dual_residual: float = 0.0
    relative_optimality_gap: float = 0.0
    l_inf_primal_variable: float = 0.0
    l2_primal_variable: float = 0.0
    l_inf_dual_variable: float = 0.0
    l2_dual_variable: float = 0.0


@dataclass
class InfeasibilityInformation:
    candidate_type: PointType = PointType.POINT_TYPE_UNSPECIFIED
    max_primal_ray_infeasibility: float = 0.0
    primal_ray_linear_objective: float = 0.0
    max_dual_ray_infeasibility: float = 0.0
    dual_ray_objective: float = 0.0


@dataclass
class IterationStats:
    iteration_number: int = 0
    convergence_information: List[ConvergenceInformation] = field(default_factory=list)
    infeasibility_information: List[InfeasibilityInformation] = field(default_factory=list)
    cumulative_kkt_matrix_passes: float = 0.0
    cumulative_rejected_steps: int = 0
    cumulative_time_sec: float = 0.0
    restart_used: RestartChoice = RestartChoice.RESTART_CHOICE_NO_RESTART
    step_size: float = 0.0
    primal_weight: float = 0.0
    method_specific_stats: Dict = field(default_factory=dict)


@dataclass
class SaddlePointOutput:
    primal_solution: "np.ndarray"
    dual_solution: "np.ndarray"
    termination_reason: TerminationReason
    termination_string: str
    iteration_count: int
    iteration_stats: List[IterationStats]
    trajectory: List[Dict] = field(default_factory=list)
    step_size: float = 0.0
    primal_weight: float = 0.0
    checkpoint: "Optional[Dict]" = None


def termination_reason_to_string(reason: TerminationReason) -> str:
    mapping = {
        TerminationReason.OPTIMAL: "TERMINATION_REASON_OPTIMAL",
        TerminationReason.PRIMAL_INFEASIBLE: "TERMINATION_REASON_PRIMAL_INFEASIBLE",
        TerminationReason.DUAL_INFEASIBLE: "TERMINATION_REASON_DUAL_INFEASIBLE",
        TerminationReason.TIME_LIMIT: "TERMINATION_REASON_TIME_LIMIT",
        TerminationReason.ITERATION_LIMIT: "TERMINATION_REASON_ITERATION_LIMIT",
        TerminationReason.KKT_MATRIX_PASS_LIMIT: "TERMINATION_REASON_KKT_MATRIX_PASS_LIMIT",
        TerminationReason.NUMERICAL_ERROR: "TERMINATION_REASON_NUMERICAL_ERROR",
    }
    return mapping.get(reason, "TERMINATION_REASON_UNSPECIFIED")
