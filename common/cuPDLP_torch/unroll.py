from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from .pdhg import _OrtExactContext, estimate_maximum_singular_value


def _spmv(A: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return (A @ v.unsqueeze(-1)).squeeze(-1)


def _ort_l2(vec: np.ndarray) -> float:
    s = 0.0
    for v in vec:
        if v == 0.0 or math.isinf(v):
            continue
        a = abs(v)
        s += a * a
    return math.sqrt(s)


class OrtUnrollContext:

    def __init__(self, qp, device: torch.device, dtype: torch.dtype = torch.float64,
                 l_inf_ruiz_iterations: int = 5, l2_norm_rescaling: bool = True) -> None:
        ctx = _OrtExactContext(qp, l_inf_ruiz_iterations, l2_norm_rescaling)
        self.m, self.n = ctx.m, ctx.n
        self.device, self.dtype = device, dtype

        def t(a):
            return torch.as_tensor(a, dtype=dtype, device=device)

        self.A = torch.sparse_csr_tensor(
            torch.as_tensor(ctx.crow, dtype=torch.int64),
            torch.as_tensor(ctx.col, dtype=torch.int64),
            t(ctx.val), size=(ctx.m, ctx.n), device=device)
        self.AT = torch.sparse_csr_tensor(
            torch.as_tensor(ctx.ccol, dtype=torch.int64),
            torch.as_tensor(ctx.crow_idx, dtype=torch.int64),
            t(ctx.val_csc), size=(ctx.n, ctx.m), device=device)
        self.c = t(ctx.c)
        self.lb = t(ctx.lb)
        self.ub = t(ctx.ub)
        self.cub = t(ctx.cub)
        self.clb = t(ctx.clb)
        self.clb_finite = torch.isfinite(self.clb)
        self.cub_finite = torch.isfinite(self.cub)
        self.clb_safe = torch.where(self.clb_finite, self.clb,
                                    torch.zeros_like(self.clb))
        self.cub_safe = torch.where(self.cub_finite, self.cub,
                                    torch.zeros_like(self.cub))
        self.col_scaling = t(ctx.col_scaling)
        self.row_scaling = t(ctx.row_scaling)

        import scipy.sparse as sps
        A_s = sps.csr_matrix((ctx.val, ctx.col, ctx.crow), shape=(ctx.m, ctx.n))
        sigma_max, _ = estimate_maximum_singular_value(
            A_s, probability_of_failure=0.001, desired_relative_error=0.2)
        self.step_size = (1.0 - 0.2) / sigma_max

        obj_norm = _ort_l2(ctx.c)
        cb = np.zeros(ctx.m)
        for i in range(ctx.m):
            mx = 0.0
            if abs(ctx.cub[i]) < math.inf:
                mx = abs(ctx.cub[i])
            if abs(ctx.clb[i]) < math.inf:
                mx = max(mx, abs(ctx.clb[i]))
            cb[i] = mx
        cb_norm = _ort_l2(cb)
        self.primal_weight = (obj_norm / cb_norm
                              if obj_norm > 0.0 and cb_norm > 0.0 else 1.0)

        self._zero = torch.zeros((), dtype=dtype, device=device)

    def initial_state(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.zeros(self.n, dtype=self.dtype, device=self.device)
        y = torch.zeros(self.m, dtype=self.dtype, device=self.device)
        ATy = torch.zeros(self.n, dtype=self.dtype, device=self.device)
        return x, y, ATy

    def unscale(self, x: torch.Tensor, y: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        return x * self.col_scaling, y * self.row_scaling


UFn = Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor, "OrtUnrollContext"],
               Tuple[torch.Tensor, torch.Tensor]]


def pdhg_unroll(
    ctx: OrtUnrollContext,
    K: int,
    u_fn: Optional[UFn] = None,
    x0: Optional[torch.Tensor] = None,
    y0: Optional[torch.Tensor] = None,
    return_trajectory: bool = False,
    x_correct_fn=None,
):
    tau = ctx.step_size / ctx.primal_weight
    sigma = ctx.step_size * ctx.primal_weight

    if x0 is None or y0 is None:
        x, y, ATy = ctx.initial_state()
        if x0 is not None:
            x = x0
        if y0 is not None:
            y = y0
            ATy = _spmv(ctx.AT, y)
    else:
        x, y = x0, y0
        ATy = _spmv(ctx.AT, y)

    traj: List[Tuple[torch.Tensor, torch.Tensor]] = []
    one = torch.ones((), dtype=ctx.dtype, device=ctx.device)

    for k in range(K):
        u2, u8 = (one, one) if u_fn is None else u_fn(k, x, y, ATy, ctx)

        x_new = torch.clamp(x - (u2 * tau) * (ctx.c - ATy), ctx.lb, ctx.ub)
        if x_correct_fn is not None:
            x_new = x_correct_fn(k, x_new, x, y, ATy, ctx)
        dx = x_new - x

        Ax_bar = _spmv(ctx.A, x_new + dx)
        sig = u8 * sigma
        temp = y - sig * Ax_bar
        upper = torch.where(
            ctx.cub_finite,
            torch.minimum(ctx._zero, temp + sig * ctx.cub_safe),
            ctx._zero.expand_as(temp))
        lower = temp + sig * ctx.clb_safe
        y_new = torch.where(ctx.clb_finite, torch.maximum(upper, lower), upper)

        ATy = _spmv(ctx.AT, y_new)
        x, y = x_new, y_new
        if return_trajectory:
            traj.append((x, y))

    if return_trajectory:
        return x, y, ATy, traj
    return x, y, ATy


def kkt_residual_loss(ctx: OrtUnrollContext, x: torch.Tensor, y: torch.Tensor,
                      ATy: Optional[torch.Tensor] = None) -> torch.Tensor:
    if ATy is None:
        ATy = _spmv(ctx.AT, y)
    Ax = _spmv(ctx.A, x)
    eq = torch.isfinite(ctx.clb)
    pr = torch.where(eq, Ax - ctx.cub, torch.relu(Ax - ctx.cub))
    du = torch.relu(ATy - ctx.c)
    return pr.dot(pr) + du.dot(du)


def duality_gap(ctx: OrtUnrollContext, x: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
    return ctx.c.dot(x) - ctx.cub.dot(y)


def kkt_gap_loss(ctx: OrtUnrollContext, x: torch.Tensor, y: torch.Tensor,
                 ATy: Optional[torch.Tensor] = None) -> torch.Tensor:
    gap = duality_gap(ctx, x, y)
    return kkt_residual_loss(ctx, x, y, ATy) + gap * gap


def pdlp_metric_loss(ctx: OrtUnrollContext, x: torch.Tensor, y: torch.Tensor,
                     ATy: Optional[torch.Tensor] = None) -> torch.Tensor:
    eq = torch.isfinite(ctx.clb)
    x = torch.clamp(x, ctx.lb, ctx.ub)
    y = torch.where(eq, y, torch.minimum(y, torch.zeros_like(y)))
    ATy = _spmv(ctx.AT, y)
    Ax = _spmv(ctx.A, x)
    pr_s = torch.where(eq, Ax - ctx.cub, torch.relu(Ax - ctx.cub))
    du_s = torch.relu(ATy - ctx.c)
    pr = pr_s / ctx.row_scaling
    du = du_s / ctx.col_scaling
    cx = ctx.c.dot(x)
    by = ctx.cub.dot(y)
    gap = cx - by

    b_orig = torch.where(ctx.cub_finite, ctx.cub_safe,
                         torch.zeros_like(ctx.cub)) / ctx.row_scaling
    c_orig = ctx.c / ctx.col_scaling
    b_norm = torch.linalg.vector_norm(b_orig)
    c_norm = torch.linalg.vector_norm(c_orig)
    gap_den = (1.0 + cx.abs() + by.abs()).detach()

    return (pr.dot(pr) / (1.0 + b_norm) ** 2
            + du.dot(du) / (1.0 + c_norm) ** 2
            + gap * gap / gap_den ** 2)


def rel_kkt_gap_loss(ctx: OrtUnrollContext, x: torch.Tensor, y: torch.Tensor,
                     ATy: Optional[torch.Tensor] = None) -> torch.Tensor:
    if ATy is None:
        ATy = _spmv(ctx.AT, y)
    Ax = _spmv(ctx.A, x)
    eq = torch.isfinite(ctx.clb)
    pr = torch.where(eq, Ax - ctx.cub, torch.relu(Ax - ctx.cub))
    du = torch.relu(ATy - ctx.c)
    gap = ctx.c.dot(x) - ctx.cub.dot(y)

    b_norm = torch.linalg.vector_norm(
        torch.where(ctx.cub_finite, ctx.cub_safe, torch.zeros_like(ctx.cub)))
    c_norm = torch.linalg.vector_norm(ctx.c)
    gap_den = (1.0 + ctx.c.dot(x).abs() + ctx.cub.dot(y).abs()).detach()

    return (pr.dot(pr) / (1.0 + b_norm) ** 2
            + du.dot(du) / (1.0 + c_norm) ** 2
            + gap * gap / gap_den ** 2)
