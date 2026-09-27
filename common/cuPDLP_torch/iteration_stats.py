
from __future__ import annotations

import torch

from .problem import GpuLinearProgrammingProblem, ScaledGpuProblem
from .solve_log import (
    ConvergenceInformation,
    IterationStats,
    PointType,
    RestartChoice,
)
from .termination import CachedQuadraticProgramInfo, TerminationCriteria



def compute_primal_residual(
    problem: GpuLinearProgrammingProblem,
    primal_solution: torch.Tensor,
    primal_product: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_eq = problem.num_equalities
    residual = problem.right_hand_side - primal_product

    if num_eq == problem.num_constraints:
        constraint_violation = residual
    elif num_eq == 0:
        constraint_violation = torch.clamp(-residual, min=0.0)
    else:
        eq_part  = residual[:num_eq]
        ineq_part = torch.clamp(-residual[num_eq:], min=0.0)
        constraint_violation = torch.cat([eq_part, ineq_part])

    lower_violation = torch.clamp(problem.variable_lower_bound - primal_solution, min=0.0)
    upper_violation = torch.clamp(primal_solution - problem.variable_upper_bound, min=0.0)

    return constraint_violation, lower_violation, upper_violation



def primal_obj(
    problem: GpuLinearProgrammingProblem,
    primal_solution: torch.Tensor,
) -> float:
    return float(torch.dot(problem.objective_vector, primal_solution)) + problem.objective_constant



def compute_dual_stats(
    problem: GpuLinearProgrammingProblem,
    primal_gradient: torch.Tensor,
    dual_solution: torch.Tensor,
) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    num_eq = problem.num_equalities
    g  = primal_gradient
    fl = problem.isfinite_variable_lower_bound.to(g.dtype)
    fu = problem.isfinite_variable_upper_bound.to(g.dtype)

    reduced_costs = torch.clamp(g, min=0.0) * fl + torch.clamp(g, max=0.0) * fu
    reduced_costs_violation = g - reduced_costs

    l = problem.variable_lower_bound
    u = problem.variable_upper_bound
    lower_contrib = torch.where(
        problem.isfinite_variable_lower_bound & (reduced_costs > 0),
        l * reduced_costs,
        torch.zeros_like(l),
    )
    upper_contrib = torch.where(
        problem.isfinite_variable_upper_bound & (reduced_costs < 0),
        u * reduced_costs,
        torch.zeros_like(u),
    )

    if num_eq < problem.num_constraints:
        dual_residual = torch.clamp(dual_solution[num_eq:], min=0.0)
    else:
        dual_residual = dual_solution.new_zeros(0)

    rcv_abs_max = reduced_costs_violation.abs().max()
    scalars_gpu = torch.stack([
        torch.dot(problem.right_hand_side, dual_solution),
        (lower_contrib + upper_contrib).sum(),
        dual_residual.abs().max() if dual_residual.numel() > 0 else rcv_abs_max.new_zeros(()),
        rcv_abs_max,
    ])
    scalars = scalars_gpu.cpu()
    dual_obj = float(scalars[0]) + problem.objective_constant + float(scalars[1])
    dual_res_inf = max(float(scalars[2]), float(scalars[3]))

    return dual_obj, dual_residual, reduced_costs, reduced_costs_violation, dual_res_inf



def compute_convergence_information(
    problem: GpuLinearProgrammingProblem,
    qp_cache: CachedQuadraticProgramInfo,
    primal_solution: torch.Tensor,
    dual_solution: torch.Tensor,
    primal_product: torch.Tensor,
    primal_gradient: torch.Tensor,
    eps_optimal_absolute: float,
    eps_optimal_relative: float,
    candidate_type: PointType,
) -> ConvergenceInformation:
    eps_ratio = eps_optimal_absolute / eps_optimal_relative

    (cv, lv, uv) = compute_primal_residual(problem, primal_solution, primal_product)
    all_pr = torch.cat([cv, lv, uv])

    (dual_obj, dual_res, rc, rcv, dual_res_inf) = compute_dual_stats(
        problem, primal_gradient, dual_solution
    )
    all_dr = torch.cat([dual_res, rcv])

    p_obj_t = torch.dot(problem.objective_vector, primal_solution)

    pr_has = all_pr.numel() > 0
    dr_has = all_dr.numel() > 0
    scalars_gpu = torch.stack([
        all_pr.abs().max() if pr_has else p_obj_t.new_zeros(()),
        all_pr.dot(all_pr) if pr_has else p_obj_t.new_zeros(()),
        all_dr.abs().max() if dr_has else p_obj_t.new_zeros(()),
        all_dr.dot(all_dr) if dr_has else p_obj_t.new_zeros(()),
        p_obj_t,
    ])
    s = scalars_gpu.cpu()
    l_inf_pr = float(s[0]) if pr_has else 0.0
    l2_pr    = float(s[1]) ** 0.5 if pr_has else 0.0
    l_inf_dr = float(s[2]) if dr_has else 0.0
    l2_dr    = float(s[3]) ** 0.5 if dr_has else 0.0
    p_obj    = float(s[4]) + problem.objective_constant

    corr_dual_obj = dual_obj if dual_res_inf == 0.0 else float("-inf")

    rel_l_inf_pr = l_inf_pr / (eps_ratio + qp_cache.l_inf_norm_b)
    rel_l2_pr    = l2_pr    / (eps_ratio + qp_cache.l2_norm_b)
    rel_l_inf_dr = l_inf_dr / (eps_ratio + qp_cache.l_inf_norm_c)
    rel_l2_dr    = l2_dr    / (eps_ratio + qp_cache.l2_norm_c)

    if corr_dual_obj == float("-inf"):
        rel_gap = float("inf")
    else:
        gap     = abs(p_obj - corr_dual_obj)
        gap_denom = eps_ratio + abs(p_obj) + abs(corr_dual_obj)
        rel_gap = gap / gap_denom

    pv_has = primal_solution.numel() > 0
    dv_has = dual_solution.numel() > 0
    var_scalars = torch.stack([
        primal_solution.abs().max() if pv_has else p_obj_t.new_zeros(()),
        primal_solution.dot(primal_solution) if pv_has else p_obj_t.new_zeros(()),
        dual_solution.abs().max() if dv_has else p_obj_t.new_zeros(()),
        dual_solution.dot(dual_solution) if dv_has else p_obj_t.new_zeros(()),
    ]).cpu()
    l_inf_pv = float(var_scalars[0]) if pv_has else 0.0
    l2_pv    = float(var_scalars[1]) ** 0.5 if pv_has else 0.0
    l_inf_dv = float(var_scalars[2]) if dv_has else 0.0
    l2_dv    = float(var_scalars[3]) ** 0.5 if dv_has else 0.0

    return ConvergenceInformation(
        candidate_type=candidate_type,
        primal_objective=p_obj,
        dual_objective=dual_obj,
        corrected_dual_objective=corr_dual_obj,
        l_inf_primal_residual=l_inf_pr,
        l2_primal_residual=l2_pr,
        l_inf_dual_residual=l_inf_dr,
        l2_dual_residual=l2_dr,
        relative_l_inf_primal_residual=rel_l_inf_pr,
        relative_l2_primal_residual=rel_l2_pr,
        relative_l_inf_dual_residual=rel_l_inf_dr,
        relative_l2_dual_residual=rel_l2_dr,
        relative_optimality_gap=rel_gap,
        l_inf_primal_variable=l_inf_pv,
        l2_primal_variable=l2_pv,
        l_inf_dual_variable=l_inf_dv,
        l2_dual_variable=l2_dv,
    )



def evaluate_unscaled_iteration_stats(
    scaled_problem: ScaledGpuProblem,
    qp_cache: CachedQuadraticProgramInfo,
    termination_criteria: TerminationCriteria,
    record_iteration_stats: bool,
    avg_primal_solution: torch.Tensor,
    avg_dual_solution: torch.Tensor,
    avg_primal_product: torch.Tensor,
    avg_primal_gradient: torch.Tensor,
    iteration: int,
    elapsed_time: float,
    cumulative_kkt_passes: float,
    step_size: float,
    primal_weight: float,
    candidate_type: PointType,
) -> IterationStats:
    D = scaled_problem.variable_rescaling
    E = scaled_problem.constraint_rescaling

    orig_primal          = avg_primal_solution  / D
    orig_dual            = avg_dual_solution    / E
    orig_primal_product  = avg_primal_product   * E
    orig_primal_gradient = avg_primal_gradient  * D

    conv_info = compute_convergence_information(
        problem=scaled_problem.original_gpu_problem,
        qp_cache=qp_cache,
        primal_solution=orig_primal,
        dual_solution=orig_dual,
        primal_product=orig_primal_product,
        primal_gradient=orig_primal_gradient,
        eps_optimal_absolute=termination_criteria.eps_optimal_absolute,
        eps_optimal_relative=termination_criteria.eps_optimal_relative,
        candidate_type=candidate_type,
    )

    return IterationStats(
        iteration_number=iteration,
        convergence_information=[conv_info],
        infeasibility_information=[],
        cumulative_kkt_matrix_passes=cumulative_kkt_passes,
        cumulative_time_sec=elapsed_time,
        step_size=step_size,
        primal_weight=primal_weight,
        method_specific_stats={},
    )
