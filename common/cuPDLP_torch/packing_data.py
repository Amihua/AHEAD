
from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import scipy.sparse as sp

from .problem import QuadraticProgrammingProblem


def _find_equality_row(A: sp.csr_matrix, n: int) -> int:
    best, best_score = 0, -1
    for r in range(A.shape[0]):
        row = A.getrow(r)
        if row.nnz < n - 2:
            continue
        vals = np.abs(row.data)
        if np.allclose(vals, 1.0, rtol=1e-5, atol=1e-6):
            score = row.nnz
            if score > best_score:
                best_score = score
                best = r
    return best


def _swap_rows(A: sp.csr_matrix, b: np.ndarray, i: int, j: int) -> tuple:
    if i == j:
        return A, b
    A = A.tolil()
    A[[i, j], :] = A[[j, i], :]
    A = A.tocsr()
    b = b.copy()
    b[i], b[j] = b[j], b[i]
    return A, b


def packing_dict_to_qp(
    data: dict,
    *,
    num_equalities: int = 1,
) -> QuadraticProgrammingProblem:
    n = int(data["c"].shape[0])
    m = int(data["b"].shape[0])
    ei = data["edge_index"]
    if hasattr(ei, "numpy"):
        ei = ei.numpy()
    ew = data["edge_weight"]
    if hasattr(ew, "numpy"):
        ew = ew.numpy()
    ei = np.asarray(ei)
    ew = np.asarray(ew, dtype=np.float64)

    A = sp.coo_matrix((ew, (ei[1], ei[0])), shape=(m, n)).tocsr()
    b = np.asarray(data["b"], dtype=np.float64).reshape(-1)
    c = np.asarray(data["c"], dtype=np.float64).reshape(-1)

    if num_equalities > 0:
        eq_row = _find_equality_row(A, n)
        if eq_row != 0:
            A, b = _swap_rows(A, b, 0, eq_row)

    if num_equalities < m:
        A_lil = A.tolil()
        A_lil[num_equalities:] = -A_lil[num_equalities:]
        A = A_lil.tocsr()
        b[num_equalities:] = -b[num_equalities:]

    lb = np.zeros(n, dtype=np.float64)
    ub = np.full(n, np.inf, dtype=np.float64)
    return QuadraticProgrammingProblem(
        variable_lower_bound=lb,
        variable_upper_bound=ub,
        objective_vector=c,
        objective_constant=0.0,
        constraint_matrix=A,
        right_hand_side=b,
        num_equalities=num_equalities,
    )


def load_packing_pkl(path: Union[str, Path]) -> QuadraticProgrammingProblem:
    with gzip.open(path, "rb") as f:
        obj = pickle.load(f)
    if "features" in obj:
        return packing_dict_to_qp(obj["features"])
    return packing_dict_to_qp(obj)


def list_packing_files(directory: Union[str, Path]) -> List[Path]:
    d = Path(directory)
    files = sorted(d.glob("packingdata*.pkl")) + sorted(d.glob("bench_*.pkl"))
    return files
