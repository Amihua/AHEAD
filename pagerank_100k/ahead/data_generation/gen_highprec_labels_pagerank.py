#!/usr/bin/env python3
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRATCH = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRATCH))
import scaling_n_pagerank as SPR

from ortools.pdlp.python import pdlp as ort_pdlp
from ortools.pdlp import solvers_pb2
from ortools.linear_solver import linear_solver_pb2 as lp_pb2


class QP:
    def __init__(self, A_full, b_full, c, n):
        self.constraint_matrix = A_full
        self.right_hand_side = b_full
        self.objective_vector = c
        self.num_variables = n
        self.num_constraints = A_full.shape[0]
        self.num_equalities = 1
        self.variable_lower_bound = np.zeros(n)
        self.variable_upper_bound = np.full(n, np.inf)


DEGREE = 3
DAMPING = 0.85
EPS = 1e-8
ITER_LIMIT = 2_000_000
TIME_LIMIT_SEC = 1800.0


def size_tag(n):
    if n >= 1_000_000 and n % 1_000_000 == 0:
        return f"{n // 1_000_000}m"
    if n >= 1_000 and n % 1_000 == 0:
        return f"{n // 1_000}k"
    return str(n)


def qp_to_mpmodel_proto(qp):
    n = qp.num_variables
    proto = lp_pb2.MPModelProto()
    proto.maximize = False
    lb, ub, c = qp.variable_lower_bound, qp.variable_upper_bound, qp.objective_vector
    for j in range(n):
        v = proto.variable.add()
        v.lower_bound = float(lb[j]) if np.isfinite(lb[j]) else float("-inf")
        v.upper_bound = float(ub[j]) if np.isfinite(ub[j]) else float("inf")
        v.objective_coefficient = float(c[j])
    A = qp.constraint_matrix.tocsr()
    b = qp.right_hand_side
    m = qp.num_constraints
    indptr, indices, data = A.indptr, A.indices, A.data
    for i in range(m):
        con = proto.constraint.add()
        st, ed = indptr[i], indptr[i + 1]
        for j, a in zip(indices[st:ed], data[st:ed]):
            con.var_index.append(int(j)); con.coefficient.append(float(a))
        if i < qp.num_equalities:
            con.lower_bound = con.upper_bound = float(b[i])
        else:
            con.lower_bound = float("-inf"); con.upper_bound = float(b[i])
    return proto


def make_params(eps):
    p = solvers_pb2.PrimalDualHybridGradientParams()
    p.termination_criteria.simple_optimality_criteria.eps_optimal_absolute = eps
    p.termination_criteria.simple_optimality_criteria.eps_optimal_relative = eps
    p.termination_criteria.iteration_limit = ITER_LIMIT
    p.termination_criteria.time_sec_limit = TIME_LIMIT_SEC
    return p


def solve_ort(qp, eps):
    proto = qp_to_mpmodel_proto(qp)
    ort_qp = ort_pdlp.qp_from_mpmodel_proto(proto, False, False)
    params = make_params(eps)
    t0 = time.perf_counter()
    res = ort_pdlp.primal_dual_hybrid_gradient(ort_qp, params)
    dt = time.perf_counter() - t0
    sl = res.solve_log
    return {
        "x_star": np.array(res.primal_solution, dtype=np.float64).tolist(),
        "y_star": np.array(res.dual_solution, dtype=np.float64).tolist(),
        "iteration_count": int(sl.iteration_count),
        "termination_reason": int(sl.termination_reason),
        "wall_time_s": dt,
    }


def main():
    n_nodes = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    n_test = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    tag = size_tag(n_nodes)
    train_seeds = list(range(n_train))
    test_seeds = list(range(10000, 10000 + n_test))

    out_train = SCRATCH / f"highprec_labels_pagerank{tag}_train{n_train}.json"
    results = []
    t_start = time.perf_counter()
    for i, seed in enumerate(train_seeds):
        rng = np.random.default_rng(seed)
        A_ineq, b_ineq, c, S, m = SPR.generate_instance(rng, n_nodes, DEGREE, DAMPING)
        A_full, b_full, c2, n2 = SPR.build_qp(A_ineq, b_ineq, c, S)
        qp = QP(A_full, b_full, c2, n2)
        r = solve_ort(qp, EPS)
        row = {"idx": i, "seed": seed, "n": n2, "m": qp.num_constraints, f"eps_{EPS:.0e}": r}
        results.append(row)
        out_train.write_text(json.dumps(results))
        el = time.perf_counter() - t_start
        print(f"  [train {i+1}/{n_train}] n={n_nodes} seed={seed} iters={r['iteration_count']} "
              f"reason={r['termination_reason']} ({r['wall_time_s']:.1f}s)  total_elapsed={el:.0f}s", flush=True)

    print(f"\nsaved train labels -> {out_train}")

    out_test = SCRATCH / f"pagerank{tag}_test_seeds.json"
    out_test.write_text(json.dumps({"n_nodes": n_nodes, "degree": DEGREE, "damping": DAMPING, "seeds": test_seeds}))
    print(f"saved test seed list -> {out_test}")


if __name__ == "__main__":
    main()
