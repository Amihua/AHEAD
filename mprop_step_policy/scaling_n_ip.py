import numpy as np
import scipy.sparse as sp

CONFIGS = {
    "IP-S": dict(multiplier=5, bdim=100),
    "IP-L": dict(multiplier=15, bdim=200),
}


def _item_groups(multiplier, bdim):
    return [
        dict(n=100 * multiplier, d_min=0.5, d_max=0.9),
        dict(n=5 * multiplier, d_min=0.6, d_max=1.0),
    ], bdim, 10 * multiplier


def generate_instance_base(rng, size="IP-S"):
    cfg = CONFIGS[size]
    groups, n_r, n_b = _item_groups(cfg["multiplier"], cfg["bdim"])
    n_i = sum(g["n"] for g in groups)

    d = np.empty((n_i, n_r), dtype=np.float64)
    off = 0
    for g in groups:
        d[off:off + g["n"], :] = rng.uniform(g["d_min"], g["d_max"], size=(g["n"], n_r))
        off += g["n"]

    t = d.sum(axis=0)
    f = 2.0
    s = f * t / n_b

    n_place = n_i * n_b
    n_deficit = n_b * n_r
    n_maxdef = n_r
    n = n_place + n_deficit + n_maxdef

    lb = np.zeros(n)
    ub = np.ones(n)
    c = np.zeros(n)
    c[n_place:n_place + n_deficit] = 1.0
    c[n_place + n_deficit:] = 10.0 * n_b * n_r

    rows_list, cols_list, vals_list = [], [], []

    eq_place = sp.kron(sp.eye(n_i, format="csr"), np.ones((1, n_b))).tocsr()
    eq = sp.hstack([eq_place, sp.csr_matrix((n_i, n_deficit + n_maxdef))], format="csr")
    b_eq = np.ones(n_i)

    supply_rows, supply_cols, supply_vals = [], [], []
    def_rows, def_cols, def_vals = [], [], []
    b_idx, i_idx = np.meshgrid(np.arange(n_b), np.arange(n_i), indexing="ij")
    b_flat, i_flat = b_idx.ravel(), i_idx.ravel()
    col_place = i_flat * n_b + b_flat

    for r in range(n_r):
        row_local = b_flat * n_r + r
        d_r = d[i_flat, r]

        supply_rows.append(row_local)
        supply_cols.append(col_place)
        supply_vals.append(d_r)

        def_rows.append(row_local)
        def_cols.append(col_place)
        def_vals.append(-(d_r * n_b / t[r]))

    supply_rows = np.concatenate(supply_rows)
    supply_cols = np.concatenate(supply_cols)
    supply_vals = np.concatenate(supply_vals)
    def_rows_item = np.concatenate(def_rows)
    def_cols_item = np.concatenate(def_cols)
    def_vals_item = np.concatenate(def_vals)

    s_2d = np.broadcast_to(s, (n_b, n_r))
    b_supply = s_2d.ravel().copy()

    br_idx = np.arange(n_b * n_r)
    def_own_col = n_place + br_idx
    def_rows_own = br_idx
    def_vals_own = -np.ones(n_b * n_r)
    b_def = -np.ones(n_b * n_r)

    A_supply = sp.coo_matrix((supply_vals, (supply_rows, supply_cols)), shape=(n_b * n_r, n)).tocsr()
    A_def = sp.coo_matrix(
        (np.concatenate([def_vals_item, def_vals_own]),
         (np.concatenate([def_rows_item, def_rows_own]), np.concatenate([def_cols_item, def_own_col]))),
        shape=(n_b * n_r, n),
    ).tocsr()

    row_md = br_idx
    r_of_row = br_idx % n_r
    col_maxdef = n_place + n_deficit + r_of_row
    col_def = n_place + br_idx
    A_maxdef = sp.coo_matrix(
        (np.concatenate([-np.ones(n_b * n_r), np.ones(n_b * n_r)]),
         (np.concatenate([row_md, row_md]), np.concatenate([col_maxdef, col_def]))),
        shape=(n_b * n_r, n),
    ).tocsr()
    b_maxdef = np.zeros(n_b * n_r)

    A_ineq = sp.vstack([A_supply, A_def, A_maxdef], format="csr")
    b_ineq = np.concatenate([b_supply, b_def, b_maxdef])

    A = sp.vstack([eq, A_ineq], format="csr")
    b = np.concatenate([b_eq, b_ineq])
    return A, b, c, lb, ub, n_i


def build_qp_full(A, b, c, lb, ub, num_equalities):
    from cuPDLP_torch.problem import QuadraticProgrammingProblem
    return QuadraticProgrammingProblem(
        variable_lower_bound=lb,
        variable_upper_bound=ub,
        objective_vector=c,
        objective_constant=0.0,
        constraint_matrix=A.astype(np.float64),
        right_hand_side=b.astype(np.float64),
        num_equalities=num_equalities,
    )


if __name__ == "__main__":
    targets = {
        "IP-S": dict(vars=31350, cons=15525, nnz=5291250),
        "IP-L": dict(vars=266450, cons=91575, nnz=94826250),
    }
    rng = np.random.default_rng(0)
    for size, tgt in targets.items():
        A, b, c, lb, ub, neq = generate_instance(rng, size)
        nv, nc, nnz = A.shape[1], A.shape[0], A.nnz
        ok = (nv, nc, nnz) == (tgt["vars"], tgt["cons"], tgt["nnz"])
        print(f"{size}: vars={nv} (target {tgt['vars']}), cons={nc} (target {tgt['cons']}), "
              f"nnz={nnz} (target {tgt['nnz']})  num_eq={neq}  {'OK' if ok else 'MISMATCH'}")


def generate_instance(rng, size):
    import os
    A, b, c, lb, ub, neq = generate_instance_base(rng, size)
    if os.environ.get("IPGEN") != "compat":
        return A, b, c, lb, ub, neq
    n_b, n_r = {"IP-S": (50, 100), "IP-L": (150, 200)}[size]; n = A.shape[1]; n_place = n - n_b * n_r - n_r; n_i = n_place // n_b
    allow = rng.random((n_i, n_b)) < 0.5
    none = ~allow.any(axis=1); allow[none, rng.integers(0, n_b, none.sum())] = True
    ub = ub.copy(); ub[:n_place] = np.where(allow.ravel(), 1.0, 0.0)
    return A, b, c, lb, ub, neq
