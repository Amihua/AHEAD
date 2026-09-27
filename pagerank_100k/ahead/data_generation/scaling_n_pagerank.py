import networkx as nx
import numpy as np
import scipy.sparse as sp

DEGREE = 3
DAMPING = 0.85


def generate_instance(rng, n, degree=DEGREE, damping_factor=DAMPING):
    seed = int(rng.integers(0, 2**31 - 1))
    graph = nx.barabasi_albert_graph(n, degree, seed=seed)
    adj = nx.adjacency_matrix(graph).astype(np.float64)
    col_sums = np.asarray(adj.sum(axis=0)).ravel()
    col_sums[col_sums == 0] = 1.0
    P = adj.multiply(1.0 / col_sums)
    P = P.tocsr()
    lp_coef = (damping_factor * P).tocsr()

    A_ineq = (lp_coef - sp.eye(n, format="csr")).tocsr()
    rhs = -(1.0 - damping_factor) / n
    b_ineq = np.full(n, rhs, dtype=np.float64)
    c = np.zeros(n, dtype=np.float64)
    S = 1.0
    m = n
    return A_ineq, b_ineq, c, S, m


def build_qp(A_ineq, b_ineq, c, S):
    n = A_ineq.shape[1]
    eq_row = sp.csr_matrix(np.ones((1, n)))
    A_full = sp.vstack([eq_row, A_ineq], format="csr")
    b_full = np.concatenate([[S], b_ineq])
    return A_full.astype(np.float64), b_full.astype(np.float64), c.astype(np.float64), n


def build_qp_full(A_ineq, b_ineq, c, S):
    from cuPDLP_torch.problem import QuadraticProgrammingProblem
    n = A_ineq.shape[1]
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
