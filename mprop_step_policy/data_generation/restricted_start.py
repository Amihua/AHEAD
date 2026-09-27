import time
import numpy as np, scipy.sparse as sp, gurobipy as gp
from gurobipy import GRB

def restricted_start(A, b, c, lb, ub, neq, k0=20, scores=None, tol=1e-5, max_rounds=4):
    t0 = time.perf_counter(); A = A.tocsr(); m, n = A.shape; s = -c if scores is None else scores; k = min(k0, n); info = dict(rounds=0); x = None; y = None
    for rd in range(max_rounds):
        J = np.sort(np.argpartition(-s, k - 1)[:k]) if k < n else np.arange(n)
        AJ = A[:, J].tocsr(); rows = np.union1d(np.arange(neq), np.nonzero(np.diff(AJ.indptr))[0]); sub = AJ[rows]
        mod = gp.Model(); mod.Params.OutputFlag = 0; mod.Params.Method = 1; mod.Params.Presolve = 0
        xv = mod.addMVar(len(J), lb=lb[J], ub=np.where(np.isinf(ub[J]), GRB.INFINITY, ub[J])); mod.setObjective(c[J] @ xv, GRB.MINIMIZE)
        cons = mod.addMConstr(sub, xv, np.array(["="] * neq + ["<"] * (len(rows) - neq)), b[rows]); mod.optimize()
        if mod.Status == GRB.OPTIMAL:
            xJ = xv.X.copy(); y = np.zeros(m); y[rows] = np.array(cons.Pi); mod.dispose(); x = np.zeros(n); x[J] = xJ
            r = AJ @ xJ - b; pv = max(np.abs(r[:neq]).max(initial=0.0), np.maximum(r[neq:], 0).max(initial=0.0)) / (1 + np.abs(b).max())
            R = np.nonzero(y)[0]; rc = c - A[R, :].T @ y[R]; free = x <= 1e-12; dv = max(0.0, float(-rc[free].min())) / (1 + np.abs(c).max()) if free.any() else 0.0
            po, do = c @ x, b @ y; gap = abs(po - do) / (1 + abs(po) + abs(do)); info.update(rounds=rd + 1, k=int(k), pv=float(pv), dv=float(dv), gap=float(gap), supp=int((xJ > 1e-9).sum()))
            if pv < tol and dv < tol and gap < tol: info["certified"] = True; info["t"] = time.perf_counter() - t0; return x, y, info
        else: mod.dispose()
        if k >= n: break
        k = min(n, 5 * k)
    info["certified"] = False; info["t"] = time.perf_counter() - t0
    return (x if x is not None else np.zeros(n)), (y if y is not None else np.zeros(m)), info
