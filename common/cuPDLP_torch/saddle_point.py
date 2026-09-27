
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch

from .problem import GpuLinearProgrammingProblem, spmv
from .iteration_stats import (
    compute_primal_residual,
    compute_dual_stats,
    primal_obj,
)
from .solve_log import RestartChoice



class RestartScheme(Enum):
    NO_RESTARTS = 0
    FIXED_FREQUENCY = 1
    ADAPTIVE_KKT = 2
    ADAPTIVE_HEURISTIC = 3


class RestartToCurrentMetric(Enum):
    NO_RESTART_TO_CURRENT = 0
    KKT_GREEDY = 1



@dataclass
class RestartParameters:
    restart_scheme: RestartScheme = RestartScheme.ADAPTIVE_HEURISTIC
    restart_to_current_metric: RestartToCurrentMetric = RestartToCurrentMetric.KKT_GREEDY
    restart_frequency_if_fixed: int = 1000
    artificial_restart_threshold: float = 0.36
    sufficient_reduction_for_restart: float = 0.1
    necessary_reduction_for_restart: float = 0.9
    primal_weight_update_smoothing: float = 0.5
    dual_decay_on_restart: float = 1.0
    major_iteration_frequency: int = 64


def construct_restart_parameters(
    restart_scheme: RestartScheme,
    restart_to_current_metric: RestartToCurrentMetric,
    restart_frequency_if_fixed: int,
    artificial_restart_threshold: float,
    sufficient_reduction_for_restart: float,
    necessary_reduction_for_restart: float,
    primal_weight_update_smoothing: float,
) -> RestartParameters:
    assert restart_frequency_if_fixed > 1
    assert 0.0 < artificial_restart_threshold <= 1.0
    assert 0.0 < sufficient_reduction_for_restart <= necessary_reduction_for_restart <= 1.0
    assert 0.0 <= primal_weight_update_smoothing <= 1.0
    return RestartParameters(
        restart_scheme=restart_scheme,
        restart_to_current_metric=restart_to_current_metric,
        restart_frequency_if_fixed=restart_frequency_if_fixed,
        artificial_restart_threshold=artificial_restart_threshold,
        sufficient_reduction_for_restart=sufficient_reduction_for_restart,
        necessary_reduction_for_restart=necessary_reduction_for_restart,
        primal_weight_update_smoothing=primal_weight_update_smoothing,
    )



@dataclass
class SolutionWeightedAverage:
    sum_primal_solutions: torch.Tensor
    sum_dual_solutions: torch.Tensor
    sum_primal_product: torch.Tensor
    sum_dual_product: torch.Tensor
    sum_primal_solution_weights: object
    sum_dual_solution_weights: object
    sum_primal_solutions_count: int
    sum_dual_solutions_count: int
    use_gpu_scalars: bool = False


def initialize_solution_weighted_average(
    n: int, m: int, device: torch.device, dtype: torch.dtype,
    use_gpu_scalars: bool = False,
) -> SolutionWeightedAverage:
    if use_gpu_scalars:
        w0 = torch.zeros((), dtype=dtype, device=device)
        return SolutionWeightedAverage(
            sum_primal_solutions=torch.zeros(n, dtype=dtype, device=device),
            sum_dual_solutions=torch.zeros(m, dtype=dtype, device=device),
            sum_primal_product=torch.zeros(m, dtype=dtype, device=device),
            sum_dual_product=torch.zeros(n, dtype=dtype, device=device),
            sum_primal_solution_weights=w0,
            sum_dual_solution_weights=w0.clone(),
            sum_primal_solutions_count=0,
            sum_dual_solutions_count=0,
            use_gpu_scalars=True,
        )
    return SolutionWeightedAverage(
        sum_primal_solutions=torch.zeros(n, dtype=dtype, device=device),
        sum_dual_solutions=torch.zeros(m, dtype=dtype, device=device),
        sum_primal_product=torch.zeros(m, dtype=dtype, device=device),
        sum_dual_product=torch.zeros(n, dtype=dtype, device=device),
        sum_primal_solution_weights=0.0,
        sum_dual_solution_weights=0.0,
        sum_primal_solutions_count=0,
        sum_dual_solutions_count=0,
        use_gpu_scalars=False,
    )


def reset_solution_weighted_average_(avg: SolutionWeightedAverage) -> None:
    avg.sum_primal_solutions.zero_()
    avg.sum_dual_solutions.zero_()
    avg.sum_primal_product.zero_()
    avg.sum_dual_product.zero_()
    if avg.use_gpu_scalars:
        avg.sum_primal_solution_weights.zero_()
        avg.sum_dual_solution_weights.zero_()
    else:
        avg.sum_primal_solution_weights = 0.0
        avg.sum_dual_solution_weights   = 0.0
    avg.sum_primal_solutions_count = 0
    avg.sum_dual_solutions_count   = 0


def add_to_solution_weighted_average_(
    avg: SolutionWeightedAverage,
    primal: torch.Tensor,
    dual: torch.Tensor,
    weight,
    dual_product: torch.Tensor,
    primal_product: Optional[torch.Tensor] = None,
) -> None:
    if isinstance(weight, torch.Tensor):
        avg.sum_primal_solutions.add_(primal * weight)
        avg.sum_dual_solutions.add_(dual * weight)
        if primal_product is not None:
            avg.sum_primal_product.add_(primal_product * weight)
        avg.sum_dual_product.add_(dual_product * weight)
        avg.sum_primal_solution_weights.add_(weight)
        avg.sum_dual_solution_weights.add_(weight)
    else:
        avg.sum_primal_solutions.add_(primal, alpha=weight)
        avg.sum_dual_solutions.add_(dual, alpha=weight)
        if primal_product is not None:
            avg.sum_primal_product.add_(primal_product, alpha=weight)
        avg.sum_dual_product.add_(dual_product, alpha=weight)
        avg.sum_primal_solution_weights += weight
        avg.sum_dual_solution_weights   += weight
    avg.sum_primal_solutions_count += 1
    avg.sum_dual_solutions_count   += 1


@dataclass
class BufferAvgState:
    avg_primal_solution: torch.Tensor
    avg_dual_solution: torch.Tensor
    avg_primal_product: torch.Tensor
    avg_primal_gradient: torch.Tensor


def compute_average_(
    avg: SolutionWeightedAverage,
    buf: BufferAvgState,
    problem: GpuLinearProgrammingProblem,
) -> None:
    w_p = avg.sum_primal_solution_weights
    w_d = avg.sum_dual_solution_weights

    buf.avg_primal_solution.copy_(avg.sum_primal_solutions / w_p)
    buf.avg_dual_solution.copy_(avg.sum_dual_solutions / w_d)
    buf.avg_primal_product.copy_(spmv(problem.constraint_matrix, buf.avg_primal_solution))
    buf.avg_primal_gradient.copy_(
        problem.objective_vector - avg.sum_dual_product / w_d
    )



@dataclass
class RestartInfo:
    primal_solution: torch.Tensor
    dual_solution: torch.Tensor
    primal_product: torch.Tensor
    primal_gradient: torch.Tensor
    last_restart_kkt_residual: Optional[float]
    last_restart_length: int
    primal_distance_moved_last_restart_period: float
    dual_distance_moved_last_restart_period: float
    kkt_reduction_ratio_last_trial: float
    normalized_gap_at_last_restart: float
    normalized_gap_at_last_trial: float


def create_last_restart_info(
    primal_solution: torch.Tensor,
    dual_solution: torch.Tensor,
    primal_product: torch.Tensor,
    primal_gradient: torch.Tensor,
) -> RestartInfo:
    return RestartInfo(
        primal_solution=primal_solution.clone(),
        dual_solution=dual_solution.clone(),
        primal_product=primal_product.clone(),
        primal_gradient=primal_gradient.clone(),
        last_restart_kkt_residual=None,
        last_restart_length=1,
        primal_distance_moved_last_restart_period=0.0,
        dual_distance_moved_last_restart_period=0.0,
        kkt_reduction_ratio_last_trial=1.0,
        normalized_gap_at_last_restart=0.0,
        normalized_gap_at_last_trial=float('inf'),
    )



def compute_weight_kkt_residual(
    problem: GpuLinearProgrammingProblem,
    primal_iterate: torch.Tensor,
    dual_iterate: torch.Tensor,
    primal_product: torch.Tensor,
    primal_gradient: torch.Tensor,
    primal_weight: float,
) -> float:
    (cv, lv, uv) = compute_primal_residual(problem, primal_iterate, primal_product)
    (d_obj, dr, _, rcv, _) = compute_dual_stats(problem, primal_gradient, dual_iterate)

    pr_cat = torch.cat([cv, lv, uv])
    dr_cat = torch.cat([dr, rcv])
    p_obj_t = torch.dot(problem.objective_vector, primal_iterate)

    scalars = torch.stack([
        pr_cat.dot(pr_cat),
        dr_cat.dot(dr_cat),
        p_obj_t,
    ]).cpu()
    l2_pr_sq = float(scalars[0])
    l2_dr_sq = float(scalars[1])
    p_obj    = float(scalars[2]) + problem.objective_constant

    return math.sqrt(
        primal_weight * l2_pr_sq
        + (1.0 / primal_weight) * l2_dr_sq
        + abs(p_obj - d_obj) ** 2
    )



def _trust_region_bound_gap(
    g_x: torch.Tensor,
    g_y: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    lb_x: torch.Tensor,
    ub_x: torch.Tensor,
    num_eq: int,
    radius: float,
    primal_weight: float,
) -> float:
    pw = primal_weight

    Δx_min = lb_x - x
    Δx_max = ub_x - x
    y_ineq = y[num_eq:]
    g_y_eq = g_y[:num_eq]
    g_y_ineq = g_y[num_eq:]

    def radius_sq_at(delta: float) -> float:
        Δx = torch.clamp(-2.0 * delta * g_x / pw, Δx_min, Δx_max)
        Δy_eq = 2.0 * pw * delta * g_y_eq
        Δy_ineq = torch.minimum(2.0 * pw * delta * g_y_ineq, -y_ineq)
        return (0.5 * pw * float(Δx.dot(Δx))
                + 0.5 / pw * (float(Δy_eq.dot(Δy_eq)) + float(Δy_ineq.dot(Δy_ineq))))

    target = radius * radius

    if target <= 0.0:
        return 0.0

    delta_hi = 1.0
    while radius_sq_at(delta_hi) < target:
        delta_hi *= 4.0
        if delta_hi > 1e20:
            break

    delta_lo = 0.0
    for _ in range(64):
        mid = (delta_lo + delta_hi) * 0.5
        if radius_sq_at(mid) < target:
            delta_lo = mid
        else:
            delta_hi = mid

    delta_opt = (delta_lo + delta_hi) * 0.5
    Δx = torch.clamp(-2.0 * delta_opt * g_x / pw, Δx_min, Δx_max)
    Δy_eq = 2.0 * pw * delta_opt * g_y_eq
    Δy_ineq = torch.minimum(2.0 * pw * delta_opt * g_y_ineq, -y_ineq)

    primal_delta = float(g_x.dot(Δx))
    dual_delta = float(g_y_eq.dot(Δy_eq)) + float(g_y_ineq.dot(Δy_ineq))
    return dual_delta - primal_delta


def compute_normalized_gap(
    x: torch.Tensor,
    y: torch.Tensor,
    Ax: torch.Tensor,
    ATy: torch.Tensor,
    x_start: torch.Tensor,
    y_start: torch.Tensor,
    problem: "GpuLinearProgrammingProblem",
    primal_weight: float,
) -> tuple[float, float]:
    pw = primal_weight
    dx = x - x_start
    dy = y - y_start
    scalars = torch.stack([dx.dot(dx), dy.dot(dy)]).cpu()
    radius_sq = 0.5 * pw * float(scalars[0]) + 0.5 / pw * float(scalars[1])
    radius = math.sqrt(radius_sq)

    if radius <= 0.0:
        return float('inf'), 0.0

    g_x = problem.objective_vector - ATy
    g_y = problem.right_hand_side - Ax

    bound_gap = _trust_region_bound_gap(
        g_x, g_y, x, y,
        problem.variable_lower_bound,
        problem.variable_upper_bound,
        problem.num_equalities,
        radius, pw,
    )
    return bound_gap / radius, radius


def _avg_has_better_potential(
    norm_gap_avg: float, radius_avg: float,
    norm_gap_cur: float, radius_cur: float,
) -> bool:
    if radius_avg <= 0.0:
        return False
    if radius_cur <= 0.0:
        return True
    return norm_gap_avg / radius_avg < norm_gap_cur / radius_cur



def _should_reset_to_average(
    current_kkt: float,
    average_kkt: float,
    metric: RestartToCurrentMetric,
) -> bool:
    if metric == RestartToCurrentMetric.KKT_GREEDY:
        return current_kkt >= average_kkt
    return True


def _should_do_adaptive_restart_heuristic(
    normalized_gap: float,
    restart_params: "RestartParameters",
    last_restart_info: "RestartInfo",
) -> bool:
    last_gap = last_restart_info.normalized_gap_at_last_restart
    if last_gap <= 0.0:
        return False
    ratio = normalized_gap / last_gap
    if ratio < restart_params.sufficient_reduction_for_restart:
        return True
    last_trial_ratio = last_restart_info.normalized_gap_at_last_trial / last_gap
    if (ratio < restart_params.necessary_reduction_for_restart
            and ratio > last_trial_ratio):
        return True
    return False


def _should_do_adaptive_restart_kkt(
    candidate_kkt: float,
    restart_params: RestartParameters,
    last_restart_info: RestartInfo,
) -> bool:
    if last_restart_info.last_restart_kkt_residual is None:
        return False
    last_kkt = last_restart_info.last_restart_kkt_residual
    if last_kkt == 0.0:
        return False
    ratio = candidate_kkt / last_kkt
    do_restart = False
    if ratio < restart_params.necessary_reduction_for_restart:
        if ratio < restart_params.sufficient_reduction_for_restart:
            do_restart = True
        elif ratio > last_restart_info.kkt_reduction_ratio_last_trial:
            do_restart = True
    last_restart_info.kkt_reduction_ratio_last_trial = ratio
    return do_restart



def run_restart_scheme(
    problem: GpuLinearProgrammingProblem,
    avg: SolutionWeightedAverage,
    current_primal_solution: torch.Tensor,
    current_dual_solution: torch.Tensor,
    current_primal_product: torch.Tensor,
    current_dual_product: torch.Tensor,
    primal_gradient: torch.Tensor,
    last_restart_info: RestartInfo,
    iterations_completed: int,
    primal_weight: float,
    restart_params: RestartParameters,
    buf_avg: BufferAvgState,
    verbosity: int = 0,
) -> RestartChoice:
    if avg.sum_primal_solutions_count == 0 or avg.sum_dual_solutions_count == 0:
        return RestartChoice.RESTART_CHOICE_NO_RESTART

    restart_length  = avg.sum_primal_solutions_count
    artificial_restart = False
    do_restart      = False

    if restart_params.restart_scheme == RestartScheme.NO_RESTARTS:
        if (iterations_completed > 0
                and iterations_completed % restart_params.major_iteration_frequency == 0):
            reset_solution_weighted_average_(avg)
            return RestartChoice.RESTART_CHOICE_WEIGHTED_AVERAGE_RESET
        return RestartChoice.RESTART_CHOICE_NO_RESTART

    if restart_params.restart_scheme == RestartScheme.ADAPTIVE_HEURISTIC:
        if restart_length >= iterations_completed * 0.5:
            do_restart = True
            artificial_restart = True

        if not do_restart:
            avg_ATy = problem.objective_vector - buf_avg.avg_primal_gradient

            norm_gap_avg, radius_avg = compute_normalized_gap(
                buf_avg.avg_primal_solution, buf_avg.avg_dual_solution,
                buf_avg.avg_primal_product, avg_ATy,
                last_restart_info.primal_solution, last_restart_info.dual_solution,
                problem, primal_weight,
            )
            norm_gap_cur, radius_cur = compute_normalized_gap(
                current_primal_solution, current_dual_solution,
                current_primal_product, current_dual_product,
                last_restart_info.primal_solution, last_restart_info.dual_solution,
                problem, primal_weight,
            )

            reset_to_average = _avg_has_better_potential(
                norm_gap_avg, radius_avg, norm_gap_cur, radius_cur
            )
            candidate_gap = norm_gap_avg if reset_to_average else norm_gap_cur

            do_restart = _should_do_adaptive_restart_heuristic(
                candidate_gap, restart_params, last_restart_info
            )
            if not do_restart:
                last_restart_info.normalized_gap_at_last_trial = candidate_gap
                return RestartChoice.RESTART_CHOICE_NO_RESTART

        if artificial_restart:
            avg_ATy = problem.objective_vector - buf_avg.avg_primal_gradient
            norm_gap_avg, radius_avg = compute_normalized_gap(
                buf_avg.avg_primal_solution, buf_avg.avg_dual_solution,
                buf_avg.avg_primal_product, avg_ATy,
                last_restart_info.primal_solution, last_restart_info.dual_solution,
                problem, primal_weight,
            )
            norm_gap_cur, radius_cur = compute_normalized_gap(
                current_primal_solution, current_dual_solution,
                current_primal_product, current_dual_product,
                last_restart_info.primal_solution, last_restart_info.dual_solution,
                problem, primal_weight,
            )
            reset_to_average = _avg_has_better_potential(
                norm_gap_avg, radius_avg, norm_gap_cur, radius_cur
            )
            candidate_gap = norm_gap_avg if reset_to_average else norm_gap_cur

        if verbosity >= 4:
            tag = "average" if reset_to_average else "current"
            art = "*" if artificial_restart else ""
            print(f"  [HEURISTIC] Restarted to {tag} after {restart_length:4d} iters{art}")

        if reset_to_average:
            current_primal_solution.copy_(buf_avg.avg_primal_solution)
            current_dual_solution.copy_(buf_avg.avg_dual_solution)
            current_primal_product.copy_(buf_avg.avg_primal_product)
            current_dual_product.copy_(
                problem.objective_vector - buf_avg.avg_primal_gradient
            )
            primal_gradient.copy_(buf_avg.avg_primal_gradient)

        alpha = restart_params.dual_decay_on_restart
        if alpha != 1.0:
            current_dual_solution.mul_(alpha)
            current_dual_product.mul_(alpha)
            primal_gradient.copy_(problem.objective_vector - current_dual_product)

        new_norm_gap, _ = compute_normalized_gap(
            current_primal_solution, current_dual_solution,
            current_primal_product, current_dual_product,
            last_restart_info.primal_solution, last_restart_info.dual_solution,
            problem, primal_weight,
        )

        _update_last_restart_info(
            last_restart_info,
            current_primal_solution=current_primal_solution,
            current_dual_solution=current_dual_solution,
            avg_primal_solution=buf_avg.avg_primal_solution,
            avg_dual_solution=buf_avg.avg_dual_solution,
            current_primal_product=current_primal_product,
            primal_gradient=primal_gradient,
            primal_weight=primal_weight,
            candidate_kkt=None,
            restart_length=restart_length,
        )
        last_restart_info.normalized_gap_at_last_restart = new_norm_gap
        last_restart_info.normalized_gap_at_last_trial = float('inf')

        reset_solution_weighted_average_(avg)

        return (RestartChoice.RESTART_CHOICE_RESTART_TO_AVERAGE if reset_to_average
                else RestartChoice.RESTART_CHOICE_WEIGHTED_AVERAGE_RESET)


    if restart_length >= restart_params.artificial_restart_threshold * iterations_completed:
        do_restart = True
        artificial_restart = True

    current_kkt = compute_weight_kkt_residual(
        problem,
        current_primal_solution, current_dual_solution,
        current_primal_product, primal_gradient,
        primal_weight,
    )
    average_kkt = compute_weight_kkt_residual(
        problem,
        buf_avg.avg_primal_solution, buf_avg.avg_dual_solution,
        buf_avg.avg_primal_product, buf_avg.avg_primal_gradient,
        primal_weight,
    )
    reset_to_average = _should_reset_to_average(
        current_kkt, average_kkt,
        restart_params.restart_to_current_metric,
    )
    candidate_kkt = average_kkt if reset_to_average else current_kkt

    if not do_restart:
        if restart_params.restart_scheme == RestartScheme.ADAPTIVE_KKT:
            do_restart = _should_do_adaptive_restart_kkt(
                candidate_kkt, restart_params, last_restart_info
            )
        elif (restart_params.restart_scheme == RestartScheme.FIXED_FREQUENCY
              and restart_params.restart_frequency_if_fixed <= restart_length):
            do_restart = True

    if not do_restart:
        return RestartChoice.RESTART_CHOICE_NO_RESTART

    if verbosity >= 4:
        tag = "average" if reset_to_average else "current"
        art = "*" if artificial_restart else ""
        print(f"  Restarted to {tag} after {restart_length:4d} iterations{art}")

    if reset_to_average:
        current_primal_solution.copy_(buf_avg.avg_primal_solution)
        current_dual_solution.copy_(buf_avg.avg_dual_solution)
        current_primal_product.copy_(buf_avg.avg_primal_product)
        current_dual_product.copy_(
            problem.objective_vector - buf_avg.avg_primal_gradient
        )
        primal_gradient.copy_(buf_avg.avg_primal_gradient)

    alpha = restart_params.dual_decay_on_restart
    if alpha != 1.0:
        current_dual_solution.mul_(alpha)
        current_dual_product.mul_(alpha)
        primal_gradient.copy_(problem.objective_vector - current_dual_product)

    _update_last_restart_info(
        last_restart_info,
        current_primal_solution=current_primal_solution,
        current_dual_solution=current_dual_solution,
        avg_primal_solution=buf_avg.avg_primal_solution,
        avg_dual_solution=buf_avg.avg_dual_solution,
        current_primal_product=current_primal_product,
        primal_gradient=primal_gradient,
        primal_weight=primal_weight,
        candidate_kkt=candidate_kkt,
        restart_length=restart_length,
    )

    reset_solution_weighted_average_(avg)

    if reset_to_average:
        return RestartChoice.RESTART_CHOICE_RESTART_TO_AVERAGE
    else:
        return RestartChoice.RESTART_CHOICE_WEIGHTED_AVERAGE_RESET



def _weighted_norm(v: torch.Tensor, w: float) -> float:
    return math.sqrt(w) * float(torch.norm(v))


def _update_last_restart_info(
    info: RestartInfo,
    current_primal_solution: torch.Tensor,
    current_dual_solution: torch.Tensor,
    avg_primal_solution: torch.Tensor,
    avg_dual_solution: torch.Tensor,
    current_primal_product: torch.Tensor,
    primal_gradient: torch.Tensor,
    primal_weight: float,
    candidate_kkt: Optional[float],
    restart_length: int,
) -> None:
    delta_p = avg_primal_solution - info.primal_solution
    delta_d = avg_dual_solution   - info.dual_solution
    dist_scalars = torch.stack([
        delta_p.dot(delta_p),
        delta_d.dot(delta_d),
    ]).cpu()
    info.primal_distance_moved_last_restart_period = float(dist_scalars[0]) ** 0.5
    info.dual_distance_moved_last_restart_period   = float(dist_scalars[1]) ** 0.5

    info.primal_solution.copy_(current_primal_solution)
    info.dual_solution.copy_(current_dual_solution)
    info.primal_product.copy_(current_primal_product)
    info.primal_gradient.copy_(primal_gradient)

    info.last_restart_length       = restart_length
    info.last_restart_kkt_residual = candidate_kkt



def compute_new_primal_weight(
    last_restart_info: RestartInfo,
    primal_weight: float,
    primal_weight_update_smoothing: float,
    verbosity: int = 0,
) -> float:
    p_dist = last_restart_info.primal_distance_moved_last_restart_period
    d_dist = last_restart_info.dual_distance_moved_last_restart_period

    if p_dist > 1e-16 and d_dist > 1e-16:
        new_estimate = d_dist / p_dist
        log_pw = (
            primal_weight_update_smoothing * math.log(new_estimate)
            + (1 - primal_weight_update_smoothing) * math.log(primal_weight)
        )
        primal_weight = math.exp(log_pw)
        if verbosity >= 4:
            print(f"  New computed primal weight is {primal_weight:.2e}")

    return primal_weight



def select_initial_primal_weight(
    problem: GpuLinearProgrammingProblem,
    primal_importance: float,
    verbosity: int = 0,
) -> float:
    norms = torch.stack([
        problem.objective_vector.dot(problem.objective_vector),
        problem.right_hand_side.dot(problem.right_hand_side),
    ]).cpu()
    obj_norm = float(norms[0]) ** 0.5
    rhs_norm = float(norms[1]) ** 0.5
    if obj_norm > 0.0 and rhs_norm > 0.0:
        pw = primal_importance * obj_norm / rhs_norm
    else:
        pw = primal_importance
    if verbosity >= 6:
        print(f"Initial primal weight = {pw:.4g}")
    return pw
