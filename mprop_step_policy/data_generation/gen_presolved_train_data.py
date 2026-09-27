#!/usr/bin/env python3
import os, sys, time
from pathlib import Path
import numpy as np, scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

OUT = ROOT / "data" / "presolved_train"
OUT.mkdir(parents=True, exist_ok=True)


def work(pair):
    n, seed = pair
    tag = f"train_n{n}_seed{seed}"
    if (OUT / f"{tag}.npz").exists():
        return tag, "skip"
    import scaling_n_mprop as SM
    from gen_highprec_labels_presolved import presolve_to_qp
    from restricted_start import restricted_start
    t0 = time.perf_counter()
    A, b, c, S, _ = SM.generate_instance(np.random.default_rng(seed), n, m_ratio=0.1, nnz_per_row=100)
    qp = presolve_to_qp(SM.build_qp(A, b, c, S))
    Ar = qp.constraint_matrix.tocsr()
    bb = np.asarray(qp.right_hand_side)
    cc = np.asarray(qp.objective_vector)
    lb = np.asarray(qp.variable_lower_bound)
    ub = np.asarray(qp.variable_upper_bound)
    neq = int(qp.num_equalities)
    x, y, info = restricted_start(Ar, bb, cc, lb, ub, neq)
    if not info["certified"]:
        return tag, "uncertified"
    np.savez(OUT / f"{tag}.npz", A_data=Ar.data, A_indices=Ar.indices, A_indptr=Ar.indptr,
             A_shape=np.array(Ar.shape), b=bb, c=cc, lb=lb, ub=ub, num_equalities=neq,
             x_star=x, y_star=y, n_orig=n)
    return tag, f"ok n_red={Ar.shape[1]} ({time.perf_counter()-t0:.0f}s)"


if __name__ == "__main__":
    import multiprocessing as mp
    from mprop_eval_split_large import MPROP_TRAIN_PAIRS
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "8"))) as p:
        for i, (tag, msg) in enumerate(p.imap_unordered(work, MPROP_TRAIN_PAIRS)):
            print(f"[{i+1}/{len(MPROP_TRAIN_PAIRS)}] {tag}: {msg}", flush=True)
    n_saved = len(list(OUT.glob("*.npz")))
    print(f"\ndone: {n_saved}/{len(MPROP_TRAIN_PAIRS)} instances certified and saved -> {OUT}")
