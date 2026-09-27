
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np
import torch

from .problem import (
    GpuLinearProgrammingProblem,
    QuadraticProgrammingProblem,
    ScaledGpuProblem,
    ScaledQpProblem,
    scaled_qp_to_gpu,
    spmv,
)
from .preprocess import rescale_problem, validate
from .termination import (
    CachedQuadraticProgramInfo,
    TerminationCriteria,
    cached_quadratic_program_info,
    check_termination_criteria,
)
from .iteration_stats import evaluate_unscaled_iteration_stats
from .saddle_point import (
    BufferAvgState,
    RestartParameters,
    RestartScheme,
    SolutionWeightedAverage,
    _update_last_restart_info,
    add_to_solution_weighted_average_,
    compute_average_,
    compute_new_primal_weight,
    construct_restart_parameters,
    create_last_restart_info,
    initialize_solution_weighted_average,
    reset_solution_weighted_average_,
    run_restart_scheme,
    select_initial_primal_weight,
)
from .solve_log import (
    IterationStats,
    PointType,
    RestartChoice,
    SaddlePointOutput,
    TerminationReason,
    termination_reason_to_string,
)


KKT_PASSES_PER_TERMINATION_EVALUATION = 2.0

_COMPILE_AVAILABLE = (
    hasattr(torch, "compile")
    and torch.cuda.is_available()
)


def _maybe_compile(fn):
    if _COMPILE_AVAILABLE:
        return torch.compile(fn, fullgraph=False, dynamic=True)
    return fn



@dataclass
class AdaptiveStepsizeParams:
    reduction_exponent: float = 0.3
    growth_exponent: float = 0.6
    check_frequency: int = 1


@dataclass
class ConstantStepsizeParams:
    pass



@dataclass
class PdhgParameters:
    l_inf_ruiz_iterations: int = 5
    l2_norm_rescaling: bool = True
    pock_chambolle_alpha: Optional[float] = None
    primal_importance: float = 1.0
    scale_invariant_initial_primal_weight: bool = True
    verbosity: int = 2
    record_iteration_stats: bool = False
    termination_evaluation_frequency: int = 64
    termination_criteria: TerminationCriteria = field(
        default_factory=TerminationCriteria)
    restart_params: RestartParameters = field(
        default_factory=RestartParameters)
    step_size_policy_params: Union[AdaptiveStepsizeParams, ConstantStepsizeParams] = field(
        default_factory=AdaptiveStepsizeParams)



@dataclass
class PdhgSolverState:
    current_primal_solution: torch.Tensor
    current_dual_solution: torch.Tensor
    current_primal_product: torch.Tensor
    current_dual_product: torch.Tensor
    step_size: float
    primal_weight: float
    numerical_error: bool
    cumulative_kkt_passes: float
    total_number_iterations: int
    last_movement: float = 0.0
    last_nonlinearity: float = 0.0
    last_step_size_limit: float = math.inf


@dataclass
class BufferState:
    delta_primal: torch.Tensor
    delta_dual: torch.Tensor
    delta_primal_product: torch.Tensor



def define_norms(step_size: float, primal_weight: float):
    return (1.0 / step_size) * primal_weight, (1.0 / step_size) / primal_weight



def _power_method_failure_probability(
    dimension: int, epsilon: float, k: int
) -> float:
    if k < 2 or epsilon <= 0.0:
        return 1.0
    return (
        min(0.824, 0.354 / math.sqrt(epsilon * (k - 1)))
        * math.sqrt(dimension)
        * (1.0 - epsilon) ** (k - 0.5)
    )


def estimate_maximum_singular_value(
    A,
    probability_of_failure: float = 0.01,
    desired_relative_error: float = 0.1,
    seed: int = 1,
) -> tuple[float, int]:
    import numpy as np

    epsilon = 1.0 - (1.0 - desired_relative_error) ** 2
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(A.shape[1])

    n_iter = 0
    while _power_method_failure_probability(A.shape[1], epsilon, n_iter) > probability_of_failure:
        x = x / np.linalg.norm(x)
        x = A.T @ (A @ x)
        n_iter += 1

    Ax = A @ x
    rayleigh = np.dot(x, A.T @ Ax) / (np.linalg.norm(x) ** 2)
    return float(np.sqrt(max(rayleigh, 0.0))), n_iter



def _primal_step_impl(
    obj: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
    A,
    current_primal: torch.Tensor,
    current_dual_product: torch.Tensor,
    step_size: torch.Tensor,
    primal_weight: torch.Tensor,
    delta_primal: torch.Tensor,
    delta_primal_product: torch.Tensor,
) -> None:
    scale = step_size / primal_weight
    x_temp = torch.clamp(
        current_primal - scale * (obj - current_dual_product), lb, ub
    )
    delta_primal.copy_(x_temp - current_primal)
    delta_primal_product.copy_(spmv(A, delta_primal))


def compute_next_primal_solution_(
    problem: GpuLinearProgrammingProblem,
    current_primal: torch.Tensor,
    current_dual_product: torch.Tensor,
    step_size: float,
    primal_weight: float,
    delta_primal: torch.Tensor,
    delta_primal_product: torch.Tensor,
) -> None:
    dtype, dev = current_primal.dtype, current_primal.device
    _primal_step_impl(
        problem.objective_vector,
        problem.variable_lower_bound,
        problem.variable_upper_bound,
        problem.constraint_matrix,
        current_primal, current_dual_product,
        torch.tensor(step_size,     dtype=dtype, device=dev),
        torch.tensor(primal_weight, dtype=dtype, device=dev),
        delta_primal, delta_primal_product,
    )



def _dual_step_impl(
    rhs: torch.Tensor,
    current_dual: torch.Tensor,
    current_primal_product: torch.Tensor,
    delta_primal_product: torch.Tensor,
    step_size: torch.Tensor,
    primal_weight: torch.Tensor,
    num_eq: int,
    num_constraints: int,
    delta_dual: torch.Tensor,
    gamma: float = 1.0,
) -> None:
    sigma = primal_weight * step_size
    increment = sigma * (
        rhs - (1.0 + gamma) * delta_primal_product - gamma * current_primal_product
    )
    y_new = current_dual + increment
    if num_eq < num_constraints:
        y_new[num_eq:] = torch.clamp(y_new[num_eq:], max=0.0)
    delta_dual.copy_(y_new - current_dual)


def compute_next_dual_solution_(
    problem: GpuLinearProgrammingProblem,
    current_dual: torch.Tensor,
    current_primal_product: torch.Tensor,
    delta_primal_product: torch.Tensor,
    step_size: float,
    primal_weight: float,
    delta_dual: torch.Tensor,
    extrapolation_coefficient: float = 1.0,
) -> None:
    dtype, dev = current_dual.dtype, current_dual.device
    _dual_step_impl(
        problem.right_hand_side,
        current_dual, current_primal_product, delta_primal_product,
        torch.tensor(step_size,     dtype=dtype, device=dev),
        torch.tensor(primal_weight, dtype=dtype, device=dev),
        problem.num_equalities, problem.num_constraints,
        delta_dual, extrapolation_coefficient,
    )



def _combined_pdhg_step_impl(
    obj: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
    A,
    AT,
    rhs: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    Ax: torch.Tensor,
    ATy: torch.Tensor,
    ss: torch.Tensor,
    pw: torch.Tensor,
    num_eq: int,
    num_constraints: int,
    dx: torch.Tensor,
    dy: torch.Tensor,
    dAx: torch.Tensor,
    atynew: torch.Tensor,
) -> torch.Tensor:
    scale = ss / pw
    x_new = torch.clamp(x - scale * (obj - ATy), lb, ub)
    dx.copy_(x_new - x)

    dAx.copy_(spmv(A, dx))
    sigma = pw * ss
    y_new = y + sigma * (rhs - 2.0 * dAx - Ax)
    if num_eq < num_constraints:
        y_new[num_eq:] = torch.clamp(y_new[num_eq:], max=0.0)
    dy.copy_(y_new - y)

    atynew.copy_(spmv(AT, y_new))

    return torch.stack([
        -dx.dot(atynew - ATy),
        dx.dot(dx),
        dy.dot(dy),
    ])


_compiled_primal_step   = _maybe_compile(_primal_step_impl)
_compiled_dual_step     = _maybe_compile(_dual_step_impl)
_compiled_combined_step = _maybe_compile(_combined_pdhg_step_impl)



def update_solution_in_solver_state_(
    problem: GpuLinearProgrammingProblem,
    state: PdhgSolverState,
    buf: BufferState,
    avg: SolutionWeightedAverage,
) -> None:
    state.current_primal_solution.add_(buf.delta_primal)
    state.current_primal_product.copy_(
        spmv(problem.constraint_matrix, state.current_primal_solution)
    )

    state.current_dual_solution.add_(buf.delta_dual)
    state.current_dual_product.copy_(
        spmv(problem.constraint_matrix_t, state.current_dual_solution)
    )

    weight = state.step_size
    add_to_solution_weighted_average_(
        avg,
        state.current_primal_solution,
        state.current_dual_solution,
        weight,
        state.current_dual_product,
    )



STEP_ATTEMPT_LOG: Optional[list] = None



try:
    from . import eigen_spmv as _eigen
except ImportError:
    _eigen = None

_KDIVERGENT_MOVEMENT = 1.0e100


class _OrtExactContext:

    def __init__(self, original_problem: QuadraticProgrammingProblem,
                 l_inf_ruiz_iterations: int, l2_norm_rescaling: bool) -> None:
        A = original_problem.constraint_matrix.tocsr()
        A.sort_indices()
        self.m, self.n = A.shape
        crow64 = A.indptr.astype(np.int64)
        col64 = A.indices.astype(np.int64)
        val0 = A.data.astype(np.float64)
        self.val, self.row_scaling, self.col_scaling = _eigen.ort_preprocess(
            crow64, col64, val0, self.m, self.n,
            l_inf_ruiz_iterations, l2_norm_rescaling)
        self.crow = crow64.astype(np.int32)
        self.col = col64.astype(np.int32)
        self.c = np.asarray(original_problem.objective_vector,
                            dtype=np.float64) * self.col_scaling
        self.lb = np.asarray(original_problem.variable_lower_bound,
                             dtype=np.float64) / self.col_scaling
        self.ub = np.asarray(original_problem.variable_upper_bound,
                             dtype=np.float64) / self.col_scaling
        b = np.asarray(original_problem.right_hand_side, dtype=np.float64)
        ne = original_problem.num_equalities
        self.cub = b * self.row_scaling
        self.clb = np.full(self.m, -np.inf)
        self.clb[:ne] = self.cub[:ne]
        self.b_scaled = self.cub
        import scipy.sparse as _sps
        A_s = _sps.csr_matrix((self.val, self.col, self.crow),
                              shape=(self.m, self.n)).tocsc()
        A_s.sort_indices()
        self.ccol = A_s.indptr.astype(np.int32)
        self.crow_idx = A_s.indices.astype(np.int32)
        self.val_csc = A_s.data

    def matvec(self, v: np.ndarray) -> np.ndarray:
        return _eigen.csr_matvec(self.crow, self.col, self.val,
                                 self.m, self.n, v)

    def matvec_scaled(self, scale: float, v: np.ndarray) -> np.ndarray:
        return _eigen.csr_matvec_scaled(self.crow, self.col, self.val,
                                        self.m, self.n, scale, v)

    def rmatvec(self, v: np.ndarray) -> np.ndarray:
        return _eigen.csr_matvec(self.ccol, self.crow_idx, self.val_csc,
                                 self.n, self.m, v)


def take_step_adaptive_ort_exact_(
    step_params: AdaptiveStepsizeParams,
    ctx: _OrtExactContext,
    state: PdhgSolverState,
    avg: SolutionWeightedAverage,
) -> None:
    step_size = state.step_size
    pw = state.primal_weight
    red_exp = step_params.reduction_exponent
    grw_exp = step_params.growth_exponent

    x = state.current_primal_solution.numpy()
    y = state.current_dual_solution.numpy()
    ATy = state.current_dual_product.numpy()

    inner = 0
    while True:
        if inner >= 60:
            state.numerical_error = True
            break
        state.total_number_iterations += 1

        primal_step_size = step_size / pw
        dual_step_size = step_size * pw

        x_new = np.maximum(
            np.minimum(x - primal_step_size * (ctx.c - ATy), ctx.ub), ctx.lb)
        dx = x_new - x

        temp = y - ctx.matvec_scaled(dual_step_size, x_new + dx)
        y_new = np.maximum(
            np.minimum(0.0, temp + dual_step_size * ctx.cub),
            temp + dual_step_size * ctx.clb)
        dy = y_new - y

        movement = (0.5 * pw * _eigen.squared_norm(dx)
                    + (0.5 / pw) * _eigen.squared_norm(dy))
        state.cumulative_kkt_passes += 1.0

        if movement == 0.0 or movement > _KDIVERGENT_MOVEMENT:
            state.numerical_error = True
            break

        atynew = ctx.rmatvec(y_new)
        nonlinearity = -_eigen.vdot(dx, atynew - ATy)

        if nonlinearity > 0:
            step_size_limit = movement / nonlinearity
        else:
            step_size_limit = math.inf

        accepted = step_size <= step_size_limit
        if accepted:
            state.current_primal_solution.copy_(torch.from_numpy(x_new))
            state.current_dual_solution.copy_(torch.from_numpy(y_new))
            state.current_dual_product.copy_(torch.from_numpy(atynew))
            state.current_primal_product.copy_(
                torch.from_numpy(ctx.matvec(x_new)))
            add_to_solution_weighted_average_(
                avg, state.current_primal_solution,
                state.current_dual_solution, step_size,
                state.current_dual_product)
            state.last_movement = movement
            state.last_nonlinearity = nonlinearity
            state.last_step_size_limit = step_size_limit

        t = float(state.total_number_iterations)
        first_term = (step_size_limit if math.isinf(step_size_limit)
                      else (1.0 - (t + 1.0) ** -red_exp) * step_size_limit)
        second_term = (1.0 + (t + 1.0) ** -grw_exp) * step_size
        step_size_tried = step_size
        step_size = min(first_term, second_term)

        if STEP_ATTEMPT_LOG is not None:
            STEP_ATTEMPT_LOG.append({
                "step_size_tried":       step_size_tried,
                "movement":              movement,
                "nonlinearity":          nonlinearity,
                "step_size_limit":       step_size_limit,
                "accepted":              accepted,
                "total_steps_attempted": t,
                "next_step_size":        step_size,
            })

        if accepted:
            break
        inner += 1

    state.step_size = step_size


def take_step_adaptive_(
    step_params: AdaptiveStepsizeParams,
    problem: GpuLinearProgrammingProblem,
    state: PdhgSolverState,
    buf: BufferState,
    avg: SolutionWeightedAverage,
    _ss: torch.Tensor,
    _pw: torch.Tensor,
) -> None:
    step_size = state.step_size
    freq      = step_params.check_frequency

    if freq > 1 and (state.total_number_iterations % freq) != 0:
        state.total_number_iterations += 1
        _ss.fill_(step_size)
        _compiled_primal_step(
            problem.objective_vector, problem.variable_lower_bound,
            problem.variable_upper_bound, problem.constraint_matrix,
            state.current_primal_solution, state.current_dual_product,
            _ss, _pw, buf.delta_primal, buf.delta_primal_product,
        )
        _compiled_dual_step(
            problem.right_hand_side, state.current_dual_solution,
            state.current_primal_product, buf.delta_primal_product,
            _ss, _pw, problem.num_equalities, problem.num_constraints,
            buf.delta_dual,
        )
        state.cumulative_kkt_passes += 1.0
        update_solution_in_solver_state_(problem, state, buf, avg)
        return

    ne  = problem.num_equalities
    nc  = problem.num_constraints
    obj = problem.objective_vector
    lb  = problem.variable_lower_bound
    ub  = problem.variable_upper_bound
    A   = problem.constraint_matrix
    AT  = problem.constraint_matrix_t
    rhs = problem.right_hand_side
    x   = state.current_primal_solution
    y   = state.current_dual_solution
    Ax  = state.current_primal_product
    ATy = state.current_dual_product
    dx  = buf.delta_primal
    dy  = buf.delta_dual
    dAx = buf.delta_primal_product

    _atynew = torch.empty_like(ATy)

    primal_weight = state.primal_weight
    _kkt = state.cumulative_kkt_passes
    red_exp = step_params.reduction_exponent
    grw_exp = step_params.growth_exponent

    done = False
    while not done:
        state.total_number_iterations += 1
        _ss.fill_(step_size)

        scalars = _compiled_combined_step(
            obj, lb, ub, A, AT, rhs,
            x, y, Ax, ATy,
            _ss, _pw,
            ne, nc,
            dx, dy, dAx,
            _atynew,
        ).cpu()
        nonlinearity = float(scalars[0])
        movement = (0.5 * primal_weight * float(scalars[1])
                    + 0.5 / primal_weight * float(scalars[2]))

        _kkt += 1.0

        if movement == 0.0:
            state.numerical_error = True
            break
        if nonlinearity > 0.0:
            step_size_limit = movement / nonlinearity
        else:
            step_size_limit = math.inf

        if step_size <= step_size_limit:
            x.add_(dx)
            y.add_(dy)
            Ax.add_(dAx)
            ATy.copy_(_atynew)
            add_to_solution_weighted_average_(avg, x, y, step_size, ATy)
            state.last_movement         = movement
            state.last_nonlinearity     = nonlinearity
            state.last_step_size_limit  = step_size_limit
            done = True

        t = state.total_number_iterations
        first_term  = (step_size_limit if math.isinf(step_size_limit)
                       else (1.0 - 1.0 / (t + 1) ** red_exp) * step_size_limit)
        second_term = (1.0 + 1.0 / (t + 1) ** grw_exp) * step_size
        step_size_tried = step_size
        step_size = min(first_term, second_term)

        if STEP_ATTEMPT_LOG is not None:
            STEP_ATTEMPT_LOG.append({
                "step_size_tried":       step_size_tried,
                "movement":              movement,
                "nonlinearity":          nonlinearity,
                "step_size_limit":       step_size_limit,
                "accepted":              done,
                "total_steps_attempted": float(t),
                "next_step_size":        step_size,
            })

    state.step_size = step_size
    state.cumulative_kkt_passes = _kkt



_INF_F64 = torch.tensor(float("inf"), dtype=torch.float64)
_INF_F32 = torch.tensor(float("inf"), dtype=torch.float32)


def take_step_adaptive_gpu_(
    step_params: AdaptiveStepsizeParams,
    problem: GpuLinearProgrammingProblem,
    state: PdhgSolverState,
    buf: BufferState,
    avg: SolutionWeightedAverage,
    step_size_t: torch.Tensor,
    primal_weight_t: torch.Tensor,
    inf_t: torch.Tensor,
) -> None:
    state.total_number_iterations += 1
    t = state.total_number_iterations

    scale = step_size_t / primal_weight_t
    x_temp = torch.clamp(
        state.current_primal_solution - scale * (problem.objective_vector - state.current_dual_product),
        problem.variable_lower_bound,
        problem.variable_upper_bound,
    )
    buf.delta_primal.copy_(x_temp - state.current_primal_solution)
    buf.delta_primal_product.copy_(spmv(problem.constraint_matrix, buf.delta_primal))

    sigma = primal_weight_t * step_size_t
    y_new = state.current_dual_solution + sigma * (
        problem.right_hand_side
        - 2.0 * buf.delta_primal_product
        - state.current_primal_product
    )
    if problem.num_equalities < problem.num_constraints:
        y_new[problem.num_equalities:] = torch.clamp(
            y_new[problem.num_equalities:], max=0.0
        )
    buf.delta_dual.copy_(y_new - state.current_dual_solution)

    nonlinearity = -torch.dot(buf.delta_primal_product, buf.delta_dual)
    norm_dx_sq   = buf.delta_primal.dot(buf.delta_primal)
    norm_dy_sq   = buf.delta_dual.dot(buf.delta_dual)
    movement     = 0.5 * primal_weight_t * norm_dx_sq + 0.5 / primal_weight_t * norm_dy_sq

    limit = torch.where(nonlinearity > 0.0, movement / nonlinearity, inf_t)

    factor_r = 1.0 - 1.0 / (t + 1) ** step_params.reduction_exponent
    factor_g  = 1.0 + 1.0 / (t + 1) ** step_params.growth_exponent

    new_step = torch.minimum(factor_r * limit, factor_g * step_size_t)
    step_size_t.copy_(new_step)

    state.cumulative_kkt_passes += 1.0

    state.current_primal_solution.add_(buf.delta_primal)
    state.current_primal_product.add_(buf.delta_primal_product)
    state.current_dual_solution.add_(buf.delta_dual)
    state.current_dual_product.copy_(
        spmv(problem.constraint_matrix_t, state.current_dual_solution)
    )

    add_to_solution_weighted_average_(
        avg,
        state.current_primal_solution,
        state.current_dual_solution,
        step_size_t,
        state.current_dual_product,
    )




def take_step_constant_(
    problem: GpuLinearProgrammingProblem,
    state: PdhgSolverState,
    buf: BufferState,
    avg: SolutionWeightedAverage,
    _ss: torch.Tensor,
    _pw: torch.Tensor,
) -> None:
    _ss.fill_(state.step_size)
    _compiled_primal_step(
        problem.objective_vector,
        problem.variable_lower_bound,
        problem.variable_upper_bound,
        problem.constraint_matrix,
        state.current_primal_solution,
        state.current_dual_product,
        _ss, _pw,
        buf.delta_primal, buf.delta_primal_product,
    )
    _compiled_dual_step(
        problem.right_hand_side,
        state.current_dual_solution,
        state.current_primal_product,
        buf.delta_primal_product,
        _ss, _pw,
        problem.num_equalities, problem.num_constraints,
        buf.delta_dual,
    )
    state.cumulative_kkt_passes += 1.0
    update_solution_in_solver_state_(problem, state, buf, avg)



def _unscaled_saddle_point_output(
    scaled_problem,
    avg_primal: np.ndarray,
    avg_dual: np.ndarray,
    termination_reason: TerminationReason,
    iterations_completed: int,
    iteration_stats: List[IterationStats],
    step_size: float = 0.0,
    primal_weight: float = 0.0,
) -> SaddlePointOutput:
    orig_primal = avg_primal / scaled_problem.variable_rescaling
    orig_dual   = avg_dual   / scaled_problem.constraint_rescaling
    return SaddlePointOutput(
        primal_solution=orig_primal,
        dual_solution=orig_dual,
        termination_reason=termination_reason,
        termination_string=termination_reason_to_string(termination_reason),
        iteration_count=iterations_completed,
        iteration_stats=iteration_stats,
        step_size=step_size,
        primal_weight=primal_weight,
    )



def _display_header(verbosity: int) -> None:
    if verbosity < 2:
        return
    print(
        f"{'Iter':>6}  {'KKT passes':>10}  {'Time(s)':>8}  "
        f"{'p_obj':>14}  {'d_obj':>14}  "
        f"{'pr_res':>10}  {'du_res':>10}  {'step':>10}  {'pw':>10}"
    )
    print("-" * 110)


def _display_iteration(stats: IterationStats, verbosity: int) -> None:
    if verbosity < 2 or not stats.convergence_information:
        return
    ci = stats.convergence_information[0]
    print(
        f"{stats.iteration_number:>6}  {stats.cumulative_kkt_matrix_passes:>10.1f}  "
        f"{stats.cumulative_time_sec:>8.2f}  "
        f"{ci.primal_objective:>14.6g}  {ci.corrected_dual_objective:>14.6g}  "
        f"{ci.l2_primal_residual:>10.3g}  {ci.l2_dual_residual:>10.3g}  "
        f"{stats.step_size:>10.3g}  {stats.primal_weight:>10.3g}"
    )


def _print_to_screen_this_iteration(
    termination_reason: Optional[TerminationReason],
    iteration: int,
    verbosity: int,
    termination_evaluation_frequency: int,
) -> bool:
    if verbosity < 2:
        return False
    if termination_reason is not None:
        return True
    if iteration <= 10:
        return True
    return iteration % max(1, termination_evaluation_frequency * 10) == 0



def _make_checkpoint(
    state: "PdhgSolverState",
    avg: "SolutionWeightedAverage",
    outer_iteration: int,
    variable_rescaling: "np.ndarray | None" = None,
    constraint_rescaling: "np.ndarray | None" = None,
    last_restart_info: "Optional[object]" = None,
) -> dict:
    spw = avg.sum_primal_solution_weights
    sdw = avg.sum_dual_solution_weights
    ckpt = {
        "x_current":              state.current_primal_solution.cpu().numpy().copy(),
        "y_current":              state.current_dual_solution.cpu().numpy().copy(),
        "Ax_current":             state.current_primal_product.cpu().numpy().copy(),
        "step_size":              state.step_size,
        "primal_weight":          state.primal_weight,
        "kkt_passes":             state.cumulative_kkt_passes,
        "total_iters":            state.total_number_iterations,
        "last_movement":          state.last_movement,
        "last_nonlinearity":      state.last_nonlinearity,
        "last_step_size_limit":   state.last_step_size_limit,
        "outer_iteration":        outer_iteration,
        "avg_sum_primal":         avg.sum_primal_solutions.cpu().numpy().copy(),
        "avg_sum_dual":           avg.sum_dual_solutions.cpu().numpy().copy(),
        "avg_sum_primal_product": avg.sum_primal_product.cpu().numpy().copy(),
        "avg_sum_dual_product":   avg.sum_dual_product.cpu().numpy().copy(),
        "avg_sum_primal_weights": float(spw.item() if hasattr(spw, "item") else spw),
        "avg_sum_dual_weights":   float(sdw.item() if hasattr(sdw, "item") else sdw),
        "avg_sum_primal_count":   avg.sum_primal_solutions_count,
        "avg_sum_dual_count":     avg.sum_dual_solutions_count,
        "variable_rescaling":     variable_rescaling.copy() if variable_rescaling is not None else None,
        "constraint_rescaling":   constraint_rescaling.copy() if constraint_rescaling is not None else None,
    }
    if last_restart_info is not None:
        ckpt["ri_primal_solution"]   = last_restart_info.primal_solution.cpu().numpy().copy()
        ckpt["ri_dual_solution"]     = last_restart_info.dual_solution.cpu().numpy().copy()
        ckpt["ri_primal_product"]    = last_restart_info.primal_product.cpu().numpy().copy()
        ckpt["ri_primal_gradient"]   = last_restart_info.primal_gradient.cpu().numpy().copy()
        ckpt["ri_kkt_residual"]      = last_restart_info.last_restart_kkt_residual
        ckpt["ri_restart_length"]    = last_restart_info.last_restart_length
        ckpt["ri_primal_dist"]       = last_restart_info.primal_distance_moved_last_restart_period
        ckpt["ri_dual_dist"]         = last_restart_info.dual_distance_moved_last_restart_period
        ckpt["ri_kkt_ratio"]         = last_restart_info.kkt_reduction_ratio_last_trial
        ckpt["ri_norm_gap_restart"]  = last_restart_info.normalized_gap_at_last_restart
        ckpt["ri_norm_gap_trial"]    = last_restart_info.normalized_gap_at_last_trial
    return ckpt


def optimize(
    params: PdhgParameters,
    original_problem: QuadraticProgrammingProblem,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    x_init: Optional[np.ndarray] = None,
    y_init: Optional[np.ndarray] = None,
    record_every: int = 0,
    step_size_init: Optional[float] = None,
    primal_weight_init: Optional[float] = None,
    checkpoint: Optional[dict] = None,
    save_checkpoint_at_k: int = 0,
    init_avg: bool = False,
    ort_exact: bool = False,
) -> SaddlePointOutput:
    if device is None:
        device = torch.device("cpu")

    validate(original_problem)
    qp_cache: CachedQuadraticProgramInfo = cached_quadratic_program_info(original_problem)

    start_rescaling = time.perf_counter()
    _ort_ctx: Optional[_OrtExactContext] = None
    if ort_exact:
        if _eigen is None:
            raise RuntimeError("ort_exact=True requires the eigen_spmv extension")
        if (device is not None and device.type != "cpu") or dtype is not torch.float64:
            raise ValueError("ort_exact=True requires CPU device and float64")
        if params.pock_chambolle_alpha is not None:
            raise ValueError("ort_exact=True does not support pock_chambolle_alpha")
        _ort_ctx = _OrtExactContext(
            original_problem,
            params.l_inf_ruiz_iterations, params.l2_norm_rescaling)
        import copy as _copy
        import scipy.sparse as _sps
        _sqp = _copy.deepcopy(original_problem)
        _sqp.constraint_matrix = _sps.csr_matrix(
            (_ort_ctx.val, _ort_ctx.col, _ort_ctx.crow),
            shape=(_ort_ctx.m, _ort_ctx.n))
        _sqp.objective_vector = _ort_ctx.c.copy()
        _sqp.variable_lower_bound = _ort_ctx.lb.copy()
        _sqp.variable_upper_bound = _ort_ctx.ub.copy()
        _sqp.right_hand_side = _ort_ctx.b_scaled.copy()
        scaled_problem_cpu = ScaledQpProblem(
            original_qp=original_problem,
            scaled_qp=_sqp,
            constraint_rescaling=1.0 / _ort_ctx.row_scaling,
            variable_rescaling=1.0 / _ort_ctx.col_scaling,
        )
    else:
        scaled_problem_cpu = rescale_problem(
            params.l_inf_ruiz_iterations,
            params.l2_norm_rescaling,
            params.pock_chambolle_alpha,
            params.verbosity,
            original_problem,
        )
    if params.verbosity >= 1:
        print(f"Preconditioning time: {time.perf_counter() - start_rescaling:.2e}s")

    n  = original_problem.num_variables
    m  = original_problem.num_constraints
    ne = original_problem.num_equalities

    d_scaled = scaled_qp_to_gpu(scaled_problem_cpu, device, dtype)
    d_problem = d_scaled.scaled_gpu_problem

    if _ort_ctx is not None and (x_init is not None or y_init is not None):
        if x_init is not None:
            _x0 = np.asarray(x_init, dtype=np.float64).copy()
            _lb0 = np.asarray(original_problem.variable_lower_bound, dtype=np.float64)
            _ub0 = np.asarray(original_problem.variable_upper_bound, dtype=np.float64)
            _x0 = np.maximum(np.minimum(_x0, _ub0), _lb0)
            _x0 = _x0 / _ort_ctx.col_scaling
        else:
            _x0 = np.zeros(n)
        if y_init is not None:
            _y0 = np.asarray(y_init, dtype=np.float64).copy()
            _b0 = np.asarray(original_problem.right_hand_side, dtype=np.float64)
            _clb0 = np.full(m, -np.inf); _clb0[:ne] = _b0[:ne]
            _cub0 = _b0
            _inf_ub = ~np.isfinite(_cub0)
            _inf_lb = ~np.isfinite(_clb0)
            _y0 = np.where(_inf_ub, np.maximum(_y0, 0.0), _y0)
            _y0 = np.where(_inf_lb, np.minimum(_y0, 0.0), _y0)
            _y0 = _y0 / _ort_ctx.row_scaling
        else:
            _y0 = np.zeros(m)
        x_sc = torch.from_numpy(_x0)
        y_sc = torch.from_numpy(_y0)
    else:
        if x_init is not None:
            x_sc = torch.tensor(
                x_init * scaled_problem_cpu.variable_rescaling, dtype=dtype, device=device
            )
            x_sc = torch.clamp(
                x_sc,
                torch.tensor(d_problem.variable_lower_bound, dtype=dtype, device=device)
                if not isinstance(d_problem.variable_lower_bound, torch.Tensor)
                else d_problem.variable_lower_bound,
                torch.tensor(d_problem.variable_upper_bound, dtype=dtype, device=device)
                if not isinstance(d_problem.variable_upper_bound, torch.Tensor)
                else d_problem.variable_upper_bound,
            )
        else:
            x_sc = torch.zeros(n, dtype=dtype, device=device)

        if y_init is not None:
            y_sc = torch.tensor(
                y_init * scaled_problem_cpu.constraint_rescaling, dtype=dtype, device=device
            )
            if ne < m:
                y_sc[ne:] = torch.clamp(y_sc[ne:], max=0.0)
        else:
            y_sc = torch.zeros(m, dtype=dtype, device=device)

    if _ort_ctx is not None:
        Ax_sc = torch.from_numpy(_ort_ctx.matvec(x_sc.numpy()))
        ATy_sc = torch.from_numpy(_ort_ctx.rmatvec(y_sc.numpy()))
    else:
        Ax_sc = spmv(d_problem.constraint_matrix,   x_sc)
        ATy_sc= spmv(d_problem.constraint_matrix_t, y_sc)

    state = PdhgSolverState(
        current_primal_solution=x_sc,
        current_dual_solution=y_sc,
        current_primal_product=Ax_sc,
        current_dual_product=ATy_sc,
        step_size=0.0,
        primal_weight=1.0,
        numerical_error=False,
        cumulative_kkt_passes=0.0,
        total_number_iterations=0,
    )

    buf = BufferState(
        delta_primal=torch.zeros(n, dtype=dtype, device=device),
        delta_dual=torch.zeros(m, dtype=dtype, device=device),
        delta_primal_product=torch.zeros(m, dtype=dtype, device=device),
    )

    gpu_native = (
        device.type == "cuda"
        and isinstance(params.step_size_policy_params, AdaptiveStepsizeParams)
        and getattr(params, "gpu_native_step", False)
    )

    avg = initialize_solution_weighted_average(
        n, m, device, dtype,
        use_gpu_scalars=(device.type == "cuda"),
    )

    if init_avg and x_init is not None:
        avg.sum_primal_solutions.copy_(x_sc)
        avg.sum_primal_product.copy_(Ax_sc)
        avg.sum_dual_solutions.copy_(y_sc)
        avg.sum_dual_product.copy_(ATy_sc)
        if hasattr(avg.sum_primal_solution_weights, "fill_"):
            avg.sum_primal_solution_weights.fill_(1.0)
            avg.sum_dual_solution_weights.fill_(1.0)
        else:
            avg.sum_primal_solution_weights = 1.0
            avg.sum_dual_solution_weights   = 1.0
        avg.sum_primal_solutions_count = 1
        avg.sum_dual_solutions_count   = 1

    buf_avg = BufferAvgState(
        avg_primal_solution=torch.zeros(n, dtype=dtype, device=device),
        avg_dual_solution=torch.zeros(m, dtype=dtype, device=device),
        avg_primal_product=torch.zeros(m, dtype=dtype, device=device),
        avg_primal_gradient=torch.zeros(n, dtype=dtype, device=device),
    )

    _ckpt_has_step = checkpoint is not None and "step_size" in checkpoint
    if checkpoint is not None:
        state.current_primal_solution.copy_(
            torch.tensor(checkpoint["x_current"], dtype=dtype, device=device))
        state.current_dual_solution.copy_(
            torch.tensor(checkpoint["y_current"], dtype=dtype, device=device))
        if "Ax_current" in checkpoint:
            state.current_primal_product.copy_(
                torch.tensor(checkpoint["Ax_current"], dtype=dtype, device=device))
        else:
            state.current_primal_product.copy_(
                spmv(d_problem.constraint_matrix, state.current_primal_solution))
        state.current_dual_product.copy_(
            spmv(d_problem.constraint_matrix_t, state.current_dual_solution))
        if "step_size" in checkpoint:
            state.step_size               = checkpoint["step_size"]
            state.primal_weight           = checkpoint["primal_weight"]
            state.cumulative_kkt_passes   = checkpoint["kkt_passes"]
            state.total_number_iterations = checkpoint["total_iters"]
        if "last_movement" in checkpoint:
            state.last_movement          = checkpoint["last_movement"]
            state.last_nonlinearity      = checkpoint["last_nonlinearity"]
            state.last_step_size_limit   = checkpoint["last_step_size_limit"]

        if "avg_sum_primal" in checkpoint:
            avg.sum_primal_solutions.copy_(
                torch.tensor(checkpoint["avg_sum_primal"], dtype=dtype, device=device))
            avg.sum_dual_solutions.copy_(
                torch.tensor(checkpoint["avg_sum_dual"], dtype=dtype, device=device))
            avg.sum_primal_product.copy_(
                torch.tensor(checkpoint["avg_sum_primal_product"], dtype=dtype, device=device))
            avg.sum_dual_product.copy_(
                torch.tensor(checkpoint["avg_sum_dual_product"], dtype=dtype, device=device))
            wp = checkpoint["avg_sum_primal_weights"]
            wd = checkpoint["avg_sum_dual_weights"]
            if hasattr(avg.sum_primal_solution_weights, "fill_"):
                avg.sum_primal_solution_weights.fill_(wp)
                avg.sum_dual_solution_weights.fill_(wd)
            else:
                avg.sum_primal_solution_weights = wp
                avg.sum_dual_solution_weights   = wd
            avg.sum_primal_solutions_count = checkpoint["avg_sum_primal_count"]
            avg.sum_dual_solutions_count   = checkpoint["avg_sum_dual_count"]
        elif init_avg:
            ATy_sc = state.current_dual_product
            avg.sum_primal_solutions.copy_(state.current_primal_solution)
            avg.sum_primal_product.copy_(state.current_primal_product)
            avg.sum_dual_solutions.copy_(state.current_dual_solution)
            avg.sum_dual_product.copy_(ATy_sc)
            if hasattr(avg.sum_primal_solution_weights, "fill_"):
                avg.sum_primal_solution_weights.fill_(1.0)
                avg.sum_dual_solution_weights.fill_(1.0)
            else:
                avg.sum_primal_solution_weights = 1.0
                avg.sum_dual_solution_weights   = 1.0
            avg.sum_primal_solutions_count = 1
            avg.sum_dual_solutions_count   = 1

    primal_gradient = d_problem.objective_vector - state.current_dual_product

    if _ckpt_has_step:
        pass
    elif step_size_init is not None:
        state.step_size = step_size_init
        state.cumulative_kkt_passes += 0.5
    elif isinstance(params.step_size_policy_params, AdaptiveStepsizeParams):
        A_cpu = scaled_problem_cpu.scaled_qp.constraint_matrix
        A_abs_max = float(np.abs(A_cpu.data).max()) if A_cpu.nnz > 0 else 1.0
        state.step_size = 1.0 / A_abs_max
        state.cumulative_kkt_passes += 0.5
    else:
        desired_rel_error = 0.2
        sigma_max, n_power = estimate_maximum_singular_value(
            scaled_problem_cpu.scaled_qp.constraint_matrix,
            probability_of_failure=0.001,
            desired_relative_error=desired_rel_error,
        )
        state.step_size = (1.0 - desired_rel_error) / sigma_max
        state.cumulative_kkt_passes += n_power

    if _ckpt_has_step:
        pass
    elif primal_weight_init is not None:
        state.primal_weight = primal_weight_init
    elif _ort_ctx is not None:
        def _ort_l2(vec):
            s = 0.0
            for v in vec:
                if v == 0.0 or math.isinf(v):
                    continue
                a = abs(v)
                s += a * a
            return math.sqrt(s)
        def _combine_bounds(ub_i, lb_i):
            mx = 0.0
            if abs(ub_i) < math.inf:
                mx = abs(ub_i)
            if abs(lb_i) < math.inf:
                mx = max(mx, abs(lb_i))
            return mx
        _obj_norm = _ort_l2(_ort_ctx.c)
        _cb = [_combine_bounds(_ort_ctx.cub[i], _ort_ctx.clb[i])
               for i in range(_ort_ctx.m)]
        _cb_norm = _ort_l2(_cb)
        if _obj_norm > 0.0 and _cb_norm > 0.0:
            state.primal_weight = _obj_norm / _cb_norm
        else:
            state.primal_weight = 1.0
    elif params.scale_invariant_initial_primal_weight:
        state.primal_weight = select_initial_primal_weight(
            d_problem, params.primal_importance, params.verbosity
        )
    else:
        state.primal_weight = params.primal_importance

    if params.verbosity >= 2:
        print(
            f"Initial step_size={state.step_size:.4g}  "
            f"primal_weight={state.primal_weight:.4g}"
        )

    last_restart_info = create_last_restart_info(
        state.current_primal_solution,
        state.current_dual_solution,
        state.current_primal_product,
        primal_gradient,
    )
    if checkpoint is not None and "ri_primal_solution" in checkpoint:
        last_restart_info.primal_solution.copy_(
            torch.tensor(checkpoint["ri_primal_solution"], dtype=dtype, device=device))
        last_restart_info.dual_solution.copy_(
            torch.tensor(checkpoint["ri_dual_solution"], dtype=dtype, device=device))
        last_restart_info.primal_product.copy_(
            torch.tensor(checkpoint["ri_primal_product"], dtype=dtype, device=device))
        last_restart_info.primal_gradient.copy_(
            torch.tensor(checkpoint["ri_primal_gradient"], dtype=dtype, device=device))
        last_restart_info.last_restart_kkt_residual                  = checkpoint["ri_kkt_residual"]
        last_restart_info.last_restart_length                        = checkpoint["ri_restart_length"]
        last_restart_info.primal_distance_moved_last_restart_period  = checkpoint["ri_primal_dist"]
        last_restart_info.dual_distance_moved_last_restart_period    = checkpoint["ri_dual_dist"]
        last_restart_info.kkt_reduction_ratio_last_trial             = checkpoint["ri_kkt_ratio"]
        last_restart_info.normalized_gap_at_last_restart             = checkpoint["ri_norm_gap_restart"]
        last_restart_info.normalized_gap_at_last_trial               = checkpoint["ri_norm_gap_trial"]

    primal_weight_update_smoothing = params.restart_params.primal_weight_update_smoothing
    termination_criteria = params.termination_criteria
    iteration_limit      = termination_criteria.iteration_limit
    term_eval_freq       = params.termination_evaluation_frequency

    _ss_t  = torch.tensor(state.step_size,     dtype=dtype, device=device)
    _pw_t  = torch.tensor(state.primal_weight, dtype=dtype, device=device)
    _inf_t = torch.tensor(float("inf"),        dtype=dtype, device=device)

    iteration_stats_list: List[IterationStats] = []
    trajectory_list: List = []
    _var_resc = scaled_problem_cpu.variable_rescaling
    _con_resc = scaled_problem_cpu.constraint_rescaling
    start_time = time.perf_counter()
    _saved_checkpoint: Optional[dict] = None

    _display_header(params.verbosity)

    if _ort_ctx is not None:
        _ort_last_start_x = x_sc.numpy().copy()
        _ort_last_start_y = y_sc.numpy().copy()
        _ort_major_freq = params.restart_params.major_iteration_frequency

    iteration = checkpoint["outer_iteration"] if checkpoint is not None else 0
    while True:
        iteration += 1

        run_termination_check = (
            (iteration - 1) % term_eval_freq == 0
            or iteration == iteration_limit + 1
            or (not gpu_native and iteration <= 10)
            or state.numerical_error
        )

        if run_termination_check:
            if gpu_native:
                state.step_size    = float(_ss_t.item())
                state.primal_weight = float(_pw_t.item())

            state.current_primal_product.copy_(
                spmv(d_problem.constraint_matrix, state.current_primal_solution)
            )

            state.cumulative_kkt_passes += KKT_PASSES_PER_TERMINATION_EVALUATION

            if (state.numerical_error
                    or avg.sum_primal_solutions_count == 0
                    or avg.sum_dual_solutions_count == 0):
                buf_avg.avg_primal_solution.copy_(state.current_primal_solution)
                buf_avg.avg_dual_solution.copy_(state.current_dual_solution)
                buf_avg.avg_primal_product.copy_(state.current_primal_product)
                buf_avg.avg_primal_gradient.copy_(primal_gradient)
            else:
                compute_average_(avg, buf_avg, d_problem)

            current_iter_stats = evaluate_unscaled_iteration_stats(
                scaled_problem=d_scaled,
                qp_cache=qp_cache,
                termination_criteria=termination_criteria,
                record_iteration_stats=params.record_iteration_stats,
                avg_primal_solution=buf_avg.avg_primal_solution,
                avg_dual_solution=buf_avg.avg_dual_solution,
                avg_primal_product=buf_avg.avg_primal_product,
                avg_primal_gradient=buf_avg.avg_primal_gradient,
                iteration=iteration,
                elapsed_time=time.perf_counter() - start_time,
                cumulative_kkt_passes=state.cumulative_kkt_passes,
                step_size=state.step_size,
                primal_weight=state.primal_weight,
                candidate_type=PointType.POINT_TYPE_AVERAGE_ITERATE,
            )

            termination_reason = check_termination_criteria(
                termination_criteria, qp_cache, current_iter_stats
            )
            if state.numerical_error and termination_reason is None:
                termination_reason = TerminationReason.NUMERICAL_ERROR

            if iteration < 10 and termination_reason in (
                TerminationReason.PRIMAL_INFEASIBLE,
                TerminationReason.DUAL_INFEASIBLE,
            ):
                termination_reason = None

            _use_current_output = False
            if termination_reason is None and avg.sum_primal_solutions_count > 0:
                _cur_grad = d_problem.objective_vector - state.current_dual_product
                cur_iter_stats = evaluate_unscaled_iteration_stats(
                    scaled_problem=d_scaled,
                    qp_cache=qp_cache,
                    termination_criteria=termination_criteria,
                    record_iteration_stats=params.record_iteration_stats,
                    avg_primal_solution=state.current_primal_solution,
                    avg_dual_solution=state.current_dual_solution,
                    avg_primal_product=state.current_primal_product,
                    avg_primal_gradient=_cur_grad,
                    iteration=iteration,
                    elapsed_time=time.perf_counter() - start_time,
                    cumulative_kkt_passes=state.cumulative_kkt_passes,
                    step_size=state.step_size,
                    primal_weight=state.primal_weight,
                    candidate_type=PointType.POINT_TYPE_CURRENT_ITERATE,
                )
                cur_termination_reason = check_termination_criteria(
                    termination_criteria, qp_cache, cur_iter_stats
                )
                if iteration < 10 and cur_termination_reason in (
                    TerminationReason.PRIMAL_INFEASIBLE,
                    TerminationReason.DUAL_INFEASIBLE,
                ):
                    cur_termination_reason = None
                if cur_termination_reason is not None:
                    termination_reason = cur_termination_reason
                    current_iter_stats = cur_iter_stats
                    _use_current_output = True

            if params.record_iteration_stats or termination_reason is not None:
                iteration_stats_list.append(current_iter_stats)

            if _print_to_screen_this_iteration(
                termination_reason, iteration, params.verbosity, term_eval_freq
            ):
                _display_iteration(current_iter_stats, params.verbosity)

            if termination_reason is not None:
                if _use_current_output:
                    avg_primal_cpu = state.current_primal_solution.cpu().numpy().copy()
                    avg_dual_cpu   = state.current_dual_solution.cpu().numpy().copy()
                else:
                    avg_primal_cpu = buf_avg.avg_primal_solution.cpu().numpy()
                    avg_dual_cpu   = buf_avg.avg_dual_solution.cpu().numpy()

                if params.verbosity >= 1:
                    print(
                        f"Terminated after {iteration - 1} iterations: "
                        f"{termination_reason_to_string(termination_reason)}"
                    )
                    if current_iter_stats.convergence_information:
                        ci = current_iter_stats.convergence_information[0]
                        print(
                            f"  primal_obj={ci.primal_objective:.10g}  "
                            f"dual_obj={ci.corrected_dual_objective:.10g}  "
                            f"pr_res={ci.l2_primal_residual:.3g}  "
                            f"du_res={ci.l2_dual_residual:.3g}"
                        )

                out = _unscaled_saddle_point_output(
                    scaled_problem_cpu,
                    avg_primal_cpu,
                    avg_dual_cpu,
                    termination_reason,
                    iteration - 1,
                    iteration_stats_list,
                    step_size=float(state.step_size),
                    primal_weight=float(state.primal_weight),
                )
                out.trajectory = trajectory_list
                out.checkpoint = _saved_checkpoint
                return out

            primal_gradient.copy_(
                d_problem.objective_vector - state.current_dual_product
            )

            if _ort_ctx is not None:
                restart_choice = RestartChoice.RESTART_CHOICE_NO_RESTART
                _iters_done = iteration - 1
                _has_avg = (avg.sum_primal_solutions_count > 0
                            or avg.sum_dual_solutions_count > 0)
                if (_has_avg and _iters_done > 0
                        and _iters_done % _ort_major_freq == 0):
                    _x_np = state.current_primal_solution.numpy()
                    _y_np = state.current_dual_solution.numpy()
                    _pd = math.sqrt(_eigen.squared_norm(_x_np - _ort_last_start_x))
                    _dd = math.sqrt(_eigen.squared_norm(_y_np - _ort_last_start_y))
                    _tol = 1.0e-10
                    _pw = state.primal_weight
                    if not (_pd <= _tol or _pd >= 1.0 / _tol
                            or _dd <= _tol or _dd >= 1.0 / _tol):
                        _s = primal_weight_update_smoothing
                        _pw = math.exp(_s * math.log(_dd / _pd)
                                       + (1.0 - _s) * math.log(_pw))
                    state.primal_weight = _pw
                    avg.sum_primal_solutions.zero_()
                    avg.sum_dual_solutions.zero_()
                    avg.sum_primal_product.zero_()
                    avg.sum_dual_product.zero_()
                    if hasattr(avg.sum_primal_solution_weights, "fill_"):
                        avg.sum_primal_solution_weights.fill_(0.0)
                        avg.sum_dual_solution_weights.fill_(0.0)
                    else:
                        avg.sum_primal_solution_weights = 0.0
                        avg.sum_dual_solution_weights = 0.0
                    avg.sum_primal_solutions_count = 0
                    avg.sum_dual_solutions_count = 0
                    _ort_last_start_x = _x_np.copy()
                    _ort_last_start_y = _y_np.copy()
                    restart_choice = RestartChoice.RESTART_CHOICE_WEIGHTED_AVERAGE_RESET
                current_iter_stats.restart_used = restart_choice
            else:
                restart_choice = run_restart_scheme(
                    problem=d_problem,
                    avg=avg,
                    current_primal_solution=state.current_primal_solution,
                    current_dual_solution=state.current_dual_solution,
                    current_primal_product=state.current_primal_product,
                    current_dual_product=state.current_dual_product,
                    primal_gradient=primal_gradient,
                    last_restart_info=last_restart_info,
                    iterations_completed=iteration - 1,
                    primal_weight=state.primal_weight,
                    restart_params=params.restart_params,
                    buf_avg=buf_avg,
                    verbosity=params.verbosity,
                )
                current_iter_stats.restart_used = restart_choice

                if restart_choice != RestartChoice.RESTART_CHOICE_NO_RESTART:
                    if params.restart_params.restart_scheme == RestartScheme.NO_RESTARTS:
                        _update_last_restart_info(
                            last_restart_info,
                            current_primal_solution=state.current_primal_solution,
                            current_dual_solution=state.current_dual_solution,
                            avg_primal_solution=state.current_primal_solution,
                            avg_dual_solution=state.current_dual_solution,
                            current_primal_product=state.current_primal_product,
                            primal_gradient=primal_gradient,
                            primal_weight=state.primal_weight,
                            candidate_kkt=None,
                            restart_length=avg.sum_primal_solutions_count,
                        )
                    _, _ = define_norms(state.step_size, state.primal_weight)
                    state.primal_weight = compute_new_primal_weight(
                        last_restart_info,
                        state.primal_weight,
                        primal_weight_update_smoothing,
                        params.verbosity,
                    )
                    if gpu_native:
                        _pw_t.fill_(state.primal_weight)

        _pw_t.fill_(state.primal_weight)
        if gpu_native:
            take_step_adaptive_gpu_(
                params.step_size_policy_params, d_problem, state, buf, avg,
                _ss_t, _pw_t, _inf_t,
            )
        elif isinstance(params.step_size_policy_params, AdaptiveStepsizeParams):
            if _ort_ctx is not None:
                take_step_adaptive_ort_exact_(
                    params.step_size_policy_params, _ort_ctx, state, avg,
                )
            else:
                take_step_adaptive_(
                    params.step_size_policy_params, d_problem, state, buf, avg,
                    _ss_t, _pw_t,
                )
        else:
            take_step_constant_(d_problem, state, buf, avg, _ss_t, _pw_t)

        if record_every > 0 and iteration % record_every == 0:
            x_cpu = state.current_primal_solution.cpu().numpy()
            y_cpu = state.current_dual_solution.cpu().numpy()
            if avg.sum_primal_solutions_count > 0:
                compute_average_(avg, buf_avg, d_problem)
                avg_x_cpu = buf_avg.avg_primal_solution.cpu().numpy()
                avg_y_cpu = buf_avg.avg_dual_solution.cpu().numpy()
            else:
                avg_x_cpu = x_cpu
                avg_y_cpu = y_cpu
            trajectory_list.append({
                "iter":       iteration,
                "x":          (x_cpu   / _var_resc).astype(np.float64),
                "y":          (y_cpu   / _con_resc).astype(np.float64),
                "avg_x":      (avg_x_cpu / _var_resc).astype(np.float64),
                "avg_y":      (avg_y_cpu / _con_resc).astype(np.float64),
                "step_size":  float(state.step_size),
                "primal_weight": float(state.primal_weight),
                "movement":   state.last_movement,
                "nonlinearity": state.last_nonlinearity,
                "step_size_limit": state.last_step_size_limit,
            })

        if save_checkpoint_at_k > 0 and _saved_checkpoint is None and iteration == save_checkpoint_at_k:
            _saved_checkpoint = _make_checkpoint(state, avg, iteration,
                                                 _var_resc, _con_resc,
                                                 last_restart_info=last_restart_info)
