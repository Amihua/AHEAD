#!/usr/bin/env python3
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB

PKG = Path(__file__).resolve().parent.parent
COMMON = PKG.parent / "common"
ROOT = PKG
SCRATCH = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(SCRATCH))
sys.path.insert(0, str(COMMON))

from cuPDLP_torch.packing_data import list_packing_files, load_packing_pkl
from cuPDLP_torch.problem import QuadraticProgrammingProblem
import scaling_n_mprop as SM
from mprop_eval_split_large import MPROP_TRAIN_PAIRS

from ortools.pdlp.python import pdlp as ort_pdlp
from ortools.pdlp import solvers_pb2
from ortools.linear_solver import linear_solver_pb2 as lp_pb2

EPS = 1e-8
ITER_LIMIT = 2_000_000
TIME_LIMIT_SEC = 1800.0

_sorted_by_n = sorted(range(len(MPROP_TRAIN_PAIRS)), key=lambda i: MPROP_TRAIN_PAIRS[i][0])
_subset_idx = [_sorted_by_n[i] for i in
               [round(j) for j in np.linspace(0, len(MPROP_TRAIN_PAIRS) - 1, 20)]]
MPROP_TRAIN_INSTANCES = [MPROP_TRAIN_PAIRS[i] for i in _subset_idx]


def build_gurobi_model(qp):
    m = gp.Model()
    m.Params.OutputFlag = 0
    n = qp.num_variables
    lb = qp.variable_lower_bound
    ub = qp.variable_upper_bound
    x = m.addMVar(n, lb=lb, ub=np.where(np.isinf(ub), GRB.INFINITY, ub))
    m.setObjective(qp.objective_vector @ x, GRB.MINIMIZE)
    A = qp.constraint_matrix.tocsr()
    b = qp.right_hand_side
    ne = qp.num_equalities
    sense = np.array(["="] * ne + ["<"] * (qp.num_constraints - ne))
    m.addMConstr(A, x, sense, b)
    m.update()
    return m


def presolve_to_qp(qp):
    gmodel = build_gurobi_model(qp)
    presolved = gmodel.presolve()
    presolved.update()
    variables = presolved.getVars()
    n = len(variables)
    lb = np.array([v.LB if np.isfinite(v.LB) else -np.inf for v in variables])
    ub = np.array([v.UB if np.isfinite(v.UB) else np.inf for v in variables])
    c = np.array([v.Obj for v in variables])

    A = presolved.getA().tocsr()
    constrs = presolved.getConstrs()
    senses = np.array([con.Sense for con in constrs])
    rhs = np.array([con.RHS for con in constrs])

    eq_mask = senses == "="
    le_mask = senses == "<"
    ge_mask = senses == ">"

    A_eq, b_eq = A[eq_mask], rhs[eq_mask]
    A_le, b_le = A[le_mask], rhs[le_mask]
    A_ge, b_ge = -A[ge_mask], -rhs[ge_mask]

    A_new = sp.vstack([A_eq, A_le, A_ge]).tocsr()
    b_new = np.concatenate([b_eq, b_le, b_ge])
    ne = int(eq_mask.sum())

    return QuadraticProgrammingProblem(
        variable_lower_bound=lb, variable_upper_bound=ub,
        objective_vector=c, objective_constant=0.0,
        constraint_matrix=A_new, right_hand_side=b_new, num_equalities=ne)


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
    for i in range(m):
        con = proto.constraint.add()
        row = A.getrow(i)
        for j, a in zip(row.indices, row.data):
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


def qp_to_serializable(qp):
    A = qp.constraint_matrix.tocsr()
    return {
        "n": qp.num_variables, "m": qp.num_constraints, "num_equalities": qp.num_equalities,
        "variable_lower_bound": [float(v) if np.isfinite(v) else None for v in qp.variable_lower_bound],
        "variable_upper_bound": [float(v) if np.isfinite(v) else None for v in qp.variable_upper_bound],
        "objective_vector": qp.objective_vector.tolist(),
        "right_hand_side": qp.right_hand_side.tolist(),
        "A_indptr": A.indptr.tolist(), "A_indices": A.indices.tolist(), "A_data": A.data.tolist(),
    }


def process_pool(name, qps_orig, meta, out_path):
    results = []
    t_start = time.perf_counter()
    for i, qp_orig in enumerate(qps_orig):
        t0 = time.perf_counter()
        qp_red = presolve_to_qp(qp_orig)
        t_presolve = time.perf_counter() - t0
        n_orig = qp_orig.num_variables
        n_red = qp_red.num_variables

        row = {"idx": i, **meta[i], "n_orig": n_orig, "n_reduced": n_red,
               "m_reduced": qp_red.num_constraints, "t_presolve": t_presolve,
               "qp_reduced": qp_to_serializable(qp_red)}
        r = solve_ort(qp_red, EPS)
        row[f"eps_{EPS:.0e}"] = r
        print(f"  [{name} {i+1}/{len(qps_orig)}] n={n_orig:>7}->red={n_red:>7}  "
              f"presolve={t_presolve:.2f}s  eps={EPS:.0e} iters={r['iteration_count']:>7} "
              f"reason={r['termination_reason']} ({r['wall_time_s']:.1f}s)", flush=True)
        results.append(row)
        out_path.write_text(json.dumps(results))
        elapsed = time.perf_counter() - t_start
        print(f"    -> case {i+1}/{len(qps_orig)} done, {elapsed:.0f}s elapsed", flush=True)
    print(f"saved -> {out_path}", flush=True)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    if which in ("all", "native"):
        files = list_packing_files(ROOT / "data" / "packing_large_train")
        qps = [load_packing_pkl(f) for f in files]
        meta = [{"file": f.name} for f in files]
        process_pool("native", qps, meta,
                     SCRATCH / "highprec_labels_packing_large_train_presolved.json")

    if which in ("all", "mprop"):
        qps, meta = [], []
        for n, seed in MPROP_TRAIN_INSTANCES:
            rng = np.random.default_rng(seed)
            A, b, c, S, m = SM.generate_instance(rng, n, m_ratio=0.1, nnz_per_row=100)
            qp = SM.build_qp(A, b, c, S)
            qps.append(qp)
            meta.append({"n_requested": n, "seed": seed})
        process_pool("mprop", qps, meta,
                     SCRATCH / "highprec_labels_mprop_train20_presolved.json")


if __name__ == "__main__":
    main()
