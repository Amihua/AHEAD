#!/usr/bin/env python3
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

PKG = Path(__file__).resolve().parent.parent
COMMON = PKG.parent / "common"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(COMMON))

from cuPDLP_torch.pdhg import AdaptiveStepsizeParams, PdhgParameters, optimize
from cuPDLP_torch.problem import QuadraticProgrammingProblem
from cuPDLP_torch.saddle_point import RestartParameters, RestartScheme
from cuPDLP_torch.termination import TerminationCriteria

M_RATIO = 0.1
NNZ_PER_ROW = 100
EPS = 1e-4
SEED = 7
N_LIST = [100_000, 200_000, 300_000, 400_000, 500_000,
          600_000, 700_000, 800_000, 900_000, 1_000_000]
N_REPEAT = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def solver_params(eps=EPS):
    return PdhgParameters(
        verbosity=0, termination_evaluation_frequency=1,
        step_size_policy_params=AdaptiveStepsizeParams(check_frequency=1),
        restart_params=RestartParameters(restart_scheme=RestartScheme.NO_RESTARTS),
        termination_criteria=TerminationCriteria(
            eps_optimal_absolute=eps, eps_optimal_relative=eps,
            iteration_limit=2_000_000))


def generate_instance(rng, n, m_ratio=M_RATIO, nnz_per_row=NNZ_PER_ROW):
    m = max(2, int(round(n * m_ratio)))
    density = nnz_per_row / n
    nnz = int(round(m * n * density))
    rows = rng.integers(0, m, nnz)
    cols = rng.integers(0, n, nnz)
    vals = rng.uniform(0.5, 1.0, nnz)
    A = sp.coo_matrix((vals, (rows, cols)), shape=(m, n)).tocsr()
    A.sum_duplicates()
    counts = np.diff(A.indptr)
    empty_or_single = np.where(counts < 2)[0]
    if len(empty_or_single) > 0:
        A = A.tolil()
        for i in empty_or_single:
            extra = rng.choice(n, 2, replace=False)
            for j in extra:
                A[i, j] = rng.uniform(0.5, 1.0)
        A = A.tocsr()

    S = float(rng.uniform(0.3, 0.7))
    x0 = rng.dirichlet(np.ones(n)) * S
    b_ineq = A @ x0 + rng.uniform(0.05, 0.3, m)
    c = rng.uniform(-1.0, 2.0, n)
    return A, b_ineq, c, S, m


def build_qp(A_ineq, b_ineq, c, S):
    m, n = A_ineq.shape
    eq_row = sp.csr_matrix(np.ones((1, n)))
    A_full = sp.vstack([eq_row, A_ineq], format="csr")
    b_full = np.concatenate([[S], b_ineq])
    return QuadraticProgrammingProblem(
        variable_lower_bound=np.zeros(n),
        variable_upper_bound=np.full(n, np.inf),
        objective_vector=c.astype(np.float64),
        objective_constant=0.0,
        constraint_matrix=A_full.astype(np.float64),
        right_hand_side=b_full.astype(np.float64),
        num_equalities=1,
    )


def binding_fraction(qp, y):
    y_ineq = y[qp.num_equalities:]
    return float(np.mean(np.abs(y_ineq) > 1e-6))


def main():
    print(f"device={DEVICE}  m_ratio={M_RATIO}  nnz_per_row={NNZ_PER_ROW}  "
          f"eps={EPS}  repeat={N_REPEAT}\n")
    print(f"{'n':>10}{'m':>10}{'nnz':>12}{'cold_iters(mean)':>18}{'iters_std':>11}"
          f"{'wall(s,mean)':>14}{'wall_std':>10}{'binding%':>10}{'status':>10}", flush=True)

    rng = np.random.default_rng(SEED)
    results = []
    for n in N_LIST:
        iters_list, wall_list, bind_list, statuses = [], [], [], []
        nnz_ref = m_ref = None
        for rep in range(N_REPEAT):
            A_ineq, b_ineq, c, S, m = generate_instance(rng, n)
            qp = build_qp(A_ineq, b_ineq, c, S)
            nnz_ref = qp.constraint_matrix.nnz
            m_ref = m

            t0 = time.perf_counter()
            res = optimize(solver_params(), qp, device=DEVICE, dtype=torch.float64)
            wall = time.perf_counter() - t0

            iters_list.append(res.iteration_count)
            wall_list.append(wall)
            statuses.append(res.termination_reason.name)
            bind_list.append(binding_fraction(qp, res.dual_solution))
            print(f"    [n={n} m={m} rep={rep}] iters={res.iteration_count} "
                  f"wall={wall:.3f}s status={res.termination_reason.name} "
                  f"binding={100*bind_list[-1]:.2f}%", flush=True)

        iters_arr = np.array(iters_list, dtype=float)
        wall_arr = np.array(wall_list, dtype=float)
        bind_arr = np.array(bind_list, dtype=float)
        print(f"{n:>10}{m_ref:>10}{nnz_ref:>12}{iters_arr.mean():>18.1f}{iters_arr.std():>11.1f}"
              f"{wall_arr.mean():>14.3f}{wall_arr.std():>10.3f}"
              f"{100*bind_arr.mean():>9.2f}%{'/'.join(sorted(set(statuses))):>15}", flush=True)
        results.append(dict(n=n, m=m_ref, nnz=nnz_ref, iters_mean=iters_arr.mean(),
                             iters_std=iters_arr.std(), wall_mean=wall_arr.mean(),
                             wall_std=wall_arr.std(), binding_mean=bind_arr.mean(),
                             statuses=statuses))

    np.save(Path(__file__).parent / "scaling_n_mprop_results.npy", results, allow_pickle=True)
    print("\nsaved -> scaling_n_mprop_results.npy")


if __name__ == "__main__":
    main()
