from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn

from .unroll import OrtUnrollContext, _spmv


class OmsgCorrectionHead(nn.Module):

    def __init__(self, hidden_dim: int = 4, start_k: int = 56) -> None:
        super().__init__()
        h = hidden_dim
        self.h = h
        self.start_k = start_k
        self.phi_y = nn.Linear(4, h)
        self.o_x = nn.Linear(4 + h, 1)
        self.log_rho_x = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.o_x.weight)
        nn.init.zeros_(self.o_x.bias)
        self._deg_y = None

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __call__(self, k: int, x_p: torch.Tensor, x_prev: torch.Tensor,
                 y: torch.Tensor, ATy: torch.Tensor,
                 ctx: OrtUnrollContext) -> torch.Tensor:
        if k < self.start_k:
            return x_p
        if self._deg_y is None or self._deg_y.shape[0] != ctx.m:
            crow = ctx.A.crow_indices()
            self._deg_y = (crow[1:] - crow[:-1]).to(
                dtype=x_p.dtype, device=x_p.device)
        Ax_p = _spmv(ctx.A, x_p)
        Ax_b = Ax_p - ctx.cub
        p_dt = self.phi_y.weight.dtype
        feat_y = torch.stack([y, ctx.cub, Ax_b, torch.relu(Ax_b)], dim=-1)
        H_y = torch.tanh(self.phi_y(feat_y.to(p_dt))).to(x_p.dtype)
        M_x = _spmv_dense(ctx.AT, H_y / (self._deg_y.unsqueeze(-1) + 1))
        feat_x = torch.cat([
            torch.stack([x_p, ctx.c, ATy - ctx.c, x_p - x_prev], dim=-1),
            M_x], dim=-1)
        delta = self.o_x(feat_x.to(p_dt)).squeeze(-1).to(x_p.dtype)
        rho = torch.exp(self.log_rho_x.clamp(-5, 5)).to(x_p.dtype)
        return torch.clamp(x_p + rho * delta, ctx.lb, ctx.ub)


def _spmv_dense(A: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    return A @ M


_OSC_N_FEAT = 10


def _osc_features(x: torch.Tensor, y: torch.Tensor, ATy: torch.Tensor,
                  mem: list, ctx: OrtUnrollContext) -> torch.Tensor:
    eps = 1e-12
    eta, omega = ctx.step_size, ctx.primal_weight
    Ax = _spmv(ctx.A, x)
    pr = torch.linalg.vector_norm(Ax - ctx.cub)
    du = torch.linalg.vector_norm(ctx.c - ATy)
    z = pr.new_zeros(())
    mov = z; nl = z; cosx = z; cosy = z; snl = z; movr = z
    if mem[0] is not None:
        dx1 = x - mem[0]; dy1 = y - mem[1]
        ADx = _spmv(ctx.A, dx1)
        mov = 0.5 * omega * dx1.dot(dx1) + (0.5 / omega) * dy1.dot(dy1)
        nls = -ADx.dot(dy1)
        nl = nls.clamp(min=0.0)
        snl = torch.tanh(nls / (mov + eps))
        if mem[2] is not None:
            dx2 = mem[0] - mem[2]; dy2 = mem[1] - mem[3]
            cosx = dx1.dot(dx2) / (dx1.norm() * dx2.norm() + eps)
            cosy = dy1.dot(dy2) / (dy1.norm() * dy2.norm() + eps)
            mov2 = (0.5 * omega * dx2.dot(dx2)
                    + (0.5 / omega) * dy2.dot(dy2))
            movr = torch.log((mov + eps) / (mov2 + eps))
    util = eta * nl / (mov + eps)
    return torch.stack([
        torch.log1p(pr), torch.log1p(du), torch.log1p(mov),
        torch.log1p(nl), torch.log1p(util),
        torch.log(torch.as_tensor(omega + eps, dtype=x.dtype,
                                  device=x.device)),
        cosx, cosy, snl, movr])


def _osc_features_norm(x: torch.Tensor, y: torch.Tensor, ATy: torch.Tensor,
                       mem: list, ctx: OrtUnrollContext) -> torch.Tensor:
    eps = 1e-12
    eta, omega = ctx.step_size, ctx.primal_weight
    n_f, m_f = float(ctx.n), float(ctx.m)
    sqrt_n, sqrt_m = n_f ** 0.5, m_f ** 0.5
    Ax = _spmv(ctx.A, x)
    pr = torch.linalg.vector_norm(Ax - ctx.cub) / sqrt_m
    du = torch.linalg.vector_norm(ctx.c - ATy) / sqrt_n
    z = pr.new_zeros(())
    mov = z; nl = z; cosx = z; cosy = z; snl = z; movr = z
    if mem[0] is not None:
        dx1 = x - mem[0]; dy1 = y - mem[1]
        ADx = _spmv(ctx.A, dx1)
        mov = (0.5 * omega * dx1.dot(dx1) / n_f
               + (0.5 / omega) * dy1.dot(dy1) / m_f)
        nls = -ADx.dot(dy1) / m_f
        nl = nls.clamp(min=0.0)
        snl = torch.tanh(nls / (mov + eps))
        if mem[2] is not None:
            dx2 = mem[0] - mem[2]; dy2 = mem[1] - mem[3]
            cosx = dx1.dot(dx2) / (dx1.norm() * dx2.norm() + eps)
            cosy = dy1.dot(dy2) / (dy1.norm() * dy2.norm() + eps)
            mov2 = (0.5 * omega * dx2.dot(dx2) / n_f
                    + (0.5 / omega) * dy2.dot(dy2) / m_f)
            movr = torch.log((mov + eps) / (mov2 + eps))
    util = eta * nl / (mov + eps)
    omega_feat = (torch.log(torch.as_tensor(omega + eps, dtype=x.dtype, device=x.device))
                  - 0.8 * math.log(n_f / m_f))
    return torch.stack([
        torch.log1p(pr), torch.log1p(du), torch.log1p(mov),
        torch.log1p(nl), torch.log1p(util),
        omega_feat, cosx, cosy, snl, movr])


_RICH_N_FEAT = _OSC_N_FEAT + 4


def _rich_features(x: torch.Tensor, y: torch.Tensor, ATy: torch.Tensor,
                   mem: list, ctx: OrtUnrollContext) -> torch.Tensor:
    eps = 1e-12
    osc = _osc_features(x, y, ATy, mem, ctx)
    Ax = _spmv(ctx.A, x)
    pr_vec = Ax - ctx.cub
    du_vec = ctx.c - ATy
    pr_l2 = torch.linalg.vector_norm(pr_vec)
    du_l2 = torch.linalg.vector_norm(du_vec)
    linf_l2_pr = pr_vec.abs().max() / (pr_l2 + eps)
    linf_l2_du = du_vec.abs().max() / (du_l2 + eps)
    compl_slack = torch.log1p(torch.linalg.vector_norm(x * du_vec))
    n = torch.as_tensor(float(ctx.n), dtype=x.dtype, device=x.device)
    sparsity = x.abs().sum() / (torch.linalg.vector_norm(x) * n.sqrt() + eps)
    return torch.cat([osc, torch.stack(
        [linf_l2_pr, linf_l2_du, compl_slack, sparsity])])


class OscStepPolicyRich(nn.Module):

    def __init__(self, hidden_sizes: List[int], pos_mode: str = None) -> None:
        super().__init__()
        self.pos_mode = pos_mode
        pos_dim = _POS_DIM[pos_mode] if pos_mode else 0
        in_dim = _RICH_N_FEAT + pos_dim
        dims = [in_dim] + list(hidden_sizes)
        layers: list = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.Tanh()]
        layers.append(nn.Linear(dims[-1], 2))
        self.net = nn.Sequential(*layers).double()
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.m = [None, None, None, None]
        self.K = None

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def reset(self) -> None:
        self.m = [None, None, None, None]

    def __call__(self, k: int, x: torch.Tensor, y: torch.Tensor,
                 ATy: torch.Tensor, ctx: OrtUnrollContext
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = _rich_features(x, y, ATy, self.m, ctx)
        if self.pos_mode:
            pos = _pos_features(k, self.K, self.pos_mode, x.dtype, x.device)
            feat = torch.cat([feat, pos])
        out = self.net(feat.unsqueeze(0)).squeeze(0)
        self.m = [x.detach(), y.detach(), self.m[0], self.m[1]]
        return torch.exp(out[0]), torch.exp(out[1])


class GRUStepPolicy(nn.Module):

    def __init__(self, hidden_dim: int = 64, pos_mode: str = None) -> None:
        super().__init__()
        self.pos_mode = pos_mode
        pos_dim = _POS_DIM[pos_mode] if pos_mode else 0
        in_dim = _RICH_N_FEAT + pos_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRUCell(in_dim, hidden_dim).double()
        self.head = nn.Linear(hidden_dim, 2).double()
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.m = [None, None, None, None]
        self.h = None
        self.K = None

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def reset(self) -> None:
        self.m = [None, None, None, None]
        self.h = None

    def __call__(self, k: int, x: torch.Tensor, y: torch.Tensor,
                 ATy: torch.Tensor, ctx: OrtUnrollContext
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = _rich_features(x, y, ATy, self.m, ctx)
        if self.pos_mode:
            pos = _pos_features(k, self.K, self.pos_mode, x.dtype, x.device)
            feat = torch.cat([feat, pos])
        feat = feat.unsqueeze(0)
        if self.h is None:
            self.h = torch.zeros(1, self.hidden_dim, dtype=feat.dtype,
                                 device=feat.device)
        self.h = self.gru(feat, self.h)
        out = self.head(self.h).squeeze(0)
        self.m = [x.detach(), y.detach(), self.m[0], self.m[1]]
        return torch.exp(out[0]), torch.exp(out[1])


def _sinusoidal_embedding(k: int, dim: int, dtype, device) -> torch.Tensor:
    half = dim // 2
    i = torch.arange(half, dtype=dtype, device=device)
    freqs = torch.exp(-i * (math.log(10000.0) * 2.0 / dim))
    ang = k * freqs
    return torch.cat([torch.sin(ang), torch.cos(ang)])


class DiffusionStylePolicy(nn.Module):

    def __init__(self, hidden: int = 128, n_layers: int = 3,
                 sin_dim: int = 16) -> None:
        super().__init__()
        self.sin_dim = sin_dim
        self.hidden = hidden
        self.state_in = nn.Linear(_OSC_N_FEAT, hidden)
        self.iter_mlp = nn.Sequential(
            nn.Linear(sin_dim + 2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden))
        trunk: list = []
        for _ in range(n_layers - 1):
            trunk += [nn.Linear(hidden, hidden), nn.Tanh()]
        self.trunk = nn.Sequential(*trunk)
        self.head = nn.Linear(hidden, 2)
        self.double()
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.iter_mlp[-1].weight)
        nn.init.zeros_(self.iter_mlp[-1].bias)
        self.m = [None, None, None, None]
        self.K = None

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def reset(self) -> None:
        self.m = [None, None, None, None]

    def __call__(self, k: int, x: torch.Tensor, y: torch.Tensor,
                 ATy: torch.Tensor, ctx: OrtUnrollContext
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = _osc_features(x, y, ATy, self.m, ctx)
        s = torch.tanh(self.state_in(feat))

        sin_emb = _sinusoidal_embedding(k, self.sin_dim, x.dtype, x.device)
        kf = torch.as_tensor(float(k), dtype=x.dtype, device=x.device)
        Kf = torch.as_tensor(float(self.K), dtype=x.dtype, device=x.device)
        iter_raw = torch.cat([sin_emb, torch.log1p(kf).reshape(1),
                              (kf / Kf).reshape(1)])
        e_k = self.iter_mlp(iter_raw)

        h = torch.tanh(s + e_k)
        h = self.trunk(h)
        out = self.head(h)
        self.m = [x.detach(), y.detach(), self.m[0], self.m[1]]
        return torch.exp(out[0]), torch.exp(out[1])


class EmbeddingStylePolicy(nn.Module):

    def __init__(self, K_max: int, emb_dim: int = 32,
                 iter_widths: Tuple[int, ...] = (64, 128, 64),
                 n_layers: int = 3) -> None:
        super().__init__()
        hidden = iter_widths[-1]
        self.hidden = hidden
        self.embed = nn.Embedding(K_max, emb_dim)
        iter_layers: list = []
        dims = [emb_dim] + list(iter_widths)
        for a, b in zip(dims[:-1], dims[1:]):
            iter_layers += [nn.Linear(a, b), nn.Tanh()]
        self.iter_mlp = nn.Sequential(*iter_layers[:-1])
        self.state_in = nn.Linear(_OSC_N_FEAT, hidden)
        trunk: list = []
        for _ in range(n_layers - 1):
            trunk += [nn.Linear(hidden, hidden), nn.Tanh()]
        self.trunk = nn.Sequential(*trunk)
        self.head = nn.Linear(hidden, 2)
        self.double()
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.iter_mlp[-1].weight)
        nn.init.zeros_(self.iter_mlp[-1].bias)
        self.m = [None, None, None, None]
        self.K = None

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def reset(self) -> None:
        self.m = [None, None, None, None]

    def __call__(self, k: int, x: torch.Tensor, y: torch.Tensor,
                 ATy: torch.Tensor, ctx: OrtUnrollContext
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = _osc_features(x, y, ATy, self.m, ctx)
        s = torch.tanh(self.state_in(feat))

        idx = torch.tensor(k, dtype=torch.long, device=x.device)
        emb = self.embed(idx).to(x.dtype)
        e_k = self.iter_mlp(emb)

        h = torch.tanh(s + e_k)
        h = self.trunk(h)
        out = self.head(h)
        self.m = [x.detach(), y.detach(), self.m[0], self.m[1]]
        return torch.exp(out[0]), torch.exp(out[1])


class StepEmbeddingPolicy(nn.Module):

    def __init__(self, K_max: int, emb_dim: int = 32,
                 step_hidden: Tuple[int, ...] = (64, 64),
                 state_hidden: int = 64, n_layers: int = 2,
                 feature_fn=None, activation=None,
                 output_cap: float = None) -> None:
        super().__init__()
        self.feature_fn = feature_fn if feature_fn is not None else _osc_features
        act = activation if activation is not None else nn.Tanh
        self.output_cap = output_cap
        self.embed = nn.Embedding(K_max, emb_dim)
        step_dims = [emb_dim + _OSC_N_FEAT] + list(step_hidden)
        step_layers: list = []
        for a, b in zip(step_dims[:-1], step_dims[1:]):
            step_layers += [nn.Linear(a, b), act()]
        self.step_mlp = nn.Sequential(*step_layers)
        self.state_encoder = nn.Sequential(
            nn.Linear(_OSC_N_FEAT, state_hidden), act())
        combo_dim = state_hidden + step_hidden[-1]
        trunk: list = []
        for _ in range(n_layers):
            trunk += [nn.Linear(combo_dim, combo_dim), act()]
        self.trunk = nn.Sequential(*trunk)
        self.head = nn.Linear(combo_dim, 2)
        self.double()
        nn.init.zeros_(self.head.weight)
        if output_cap is None:
            nn.init.zeros_(self.head.bias)
        else:
            bias0 = math.atanh(min(1.0 / output_cap, 1.0 - 1e-6))
            nn.init.constant_(self.head.bias, bias0)
        self.m = [None, None, None, None]
        self.K = None

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def reset(self) -> None:
        self.m = [None, None, None, None]

    def __call__(self, k: int, x: torch.Tensor, y: torch.Tensor,
                 ATy: torch.Tensor, ctx: OrtUnrollContext
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.feature_fn(x, y, ATy, self.m, ctx)
        idx = torch.tensor(k, dtype=torch.long, device=x.device)
        emb = self.embed(idx).to(x.dtype)
        step_emb = self.step_mlp(torch.cat([emb, feat]))
        state_emb = self.state_encoder(feat)
        h = torch.cat([state_emb, step_emb])
        h = self.trunk(h)
        out = self.head(h)
        self.m = [x.detach(), y.detach(), self.m[0], self.m[1]]
        if self.output_cap is None:
            return torch.exp(out[0]), torch.exp(out[1])
        k = self.output_cap
        return k * torch.tanh(out[0]), k * torch.tanh(out[1])


class OscStepPolicy(nn.Module):

    N_FEAT = _OSC_N_FEAT

    def __init__(self, hidden: int = 32, n_layers: int = 1) -> None:
        super().__init__()
        layers = [nn.Linear(self.N_FEAT, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*layers).double()
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.m = [None, None, None, None]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def reset(self) -> None:
        self.m = [None, None, None, None]

    def __call__(self, k: int, x: torch.Tensor, y: torch.Tensor,
                 ATy: torch.Tensor, ctx: OrtUnrollContext
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = _osc_features(x, y, ATy, self.m, ctx)
        out = self.net(feat.unsqueeze(0)).squeeze(0)
        self.m = [x.detach(), y.detach(), self.m[0], self.m[1]]
        return torch.exp(out[0]), torch.exp(out[1])


class OscStepPolicyDeep(nn.Module):

    N_FEAT = _OSC_N_FEAT

    def __init__(self, hidden_sizes: List[int]) -> None:
        super().__init__()
        dims = [self.N_FEAT] + list(hidden_sizes)
        layers: list = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.Tanh()]
        layers.append(nn.Linear(dims[-1], 2))
        self.net = nn.Sequential(*layers).double()
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.m = [None, None, None, None]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def reset(self) -> None:
        self.m = [None, None, None, None]

    def __call__(self, k: int, x: torch.Tensor, y: torch.Tensor,
                 ATy: torch.Tensor, ctx: OrtUnrollContext
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = _osc_features(x, y, ATy, self.m, ctx)
        out = self.net(feat.unsqueeze(0)).squeeze(0)
        self.m = [x.detach(), y.detach(), self.m[0], self.m[1]]
        return torch.exp(out[0]), torch.exp(out[1])


def _pos_features(k: int, K: int, mode: str, dtype, device) -> torch.Tensor:
    t = torch.tensor(k / max(K - 1, 1), dtype=dtype, device=device)
    if mode == "linear":
        return t.reshape(1)
    if mode == "sincos":
        ang = 2 * torch.pi * t
        return torch.stack([torch.sin(ang), torch.cos(ang)])
    if mode == "multifreq":
        n_freq = 4
        feats = []
        for i in range(n_freq):
            freq = 2.0 ** i
            ang = 2 * torch.pi * freq * t
            feats += [torch.sin(ang), torch.cos(ang)]
        return torch.stack(feats)
    raise ValueError(f"unknown pos mode: {mode}")


_POS_DIM = {"linear": 1, "sincos": 2, "multifreq": 8}


class OscStepPolicyPos(nn.Module):

    def __init__(self, hidden_sizes: List[int], pos_mode: str = "sincos") -> None:
        super().__init__()
        self.pos_mode = pos_mode
        pos_dim = _POS_DIM[pos_mode]
        in_dim = _OSC_N_FEAT + pos_dim
        dims = [in_dim] + list(hidden_sizes)
        layers: list = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.Tanh()]
        layers.append(nn.Linear(dims[-1], 2))
        self.net = nn.Sequential(*layers).double()
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.m = [None, None, None, None]
        self.K = None

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def reset(self) -> None:
        self.m = [None, None, None, None]

    def __call__(self, k: int, x: torch.Tensor, y: torch.Tensor,
                 ATy: torch.Tensor, ctx: OrtUnrollContext
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        osc = _osc_features(x, y, ATy, self.m, ctx)
        pos = _pos_features(k, self.K, self.pos_mode, x.dtype, x.device)
        feat = torch.cat([osc, pos])
        out = self.net(feat.unsqueeze(0)).squeeze(0)
        self.m = [x.detach(), y.detach(), self.m[0], self.m[1]]
        return torch.exp(out[0]), torch.exp(out[1])


class StepPolicyMLP(nn.Module):
    N_FEAT = 6

    def __init__(self, hidden: int = 16, n_layers: int = 1) -> None:
        super().__init__()
        layers = [nn.Linear(self.N_FEAT, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 2))
        self.mlp = nn.Sequential(*layers)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self._prev_x = None
        self._prev_y = None

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def reset(self) -> None:
        self._prev_x = None
        self._prev_y = None

    def __call__(self, k: int, x: torch.Tensor, y: torch.Tensor,
                 ATy: torch.Tensor, ctx: OrtUnrollContext
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        eps = 1e-12
        eta = ctx.step_size
        omega = ctx.primal_weight

        Ax = _spmv(ctx.A, x)
        primal_res = torch.linalg.vector_norm(Ax - ctx.cub)
        dual_res = torch.linalg.vector_norm(ctx.c - ATy)

        if self._prev_x is None:
            movement = primal_res.new_zeros(())
            nl_pos = primal_res.new_zeros(())
        else:
            dx = x - self._prev_x
            dy = y - self._prev_y
            ADx = _spmv(ctx.A, dx)
            movement = (0.5 * omega * dx.dot(dx)
                        + (0.5 / omega) * dy.dot(dy))
            nl_pos = (-ADx.dot(dy)).clamp(min=0.0)
        step_util = eta * nl_pos / (movement + eps)

        feat = torch.stack([
            torch.log1p(primal_res),
            torch.log1p(dual_res),
            torch.log1p(movement),
            torch.log1p(nl_pos),
            torch.log1p(step_util),
            torch.log(torch.as_tensor(omega + eps, dtype=x.dtype,
                                      device=x.device)),
        ])
        p_dt = next(self.mlp.parameters()).dtype
        out = self.mlp(feat.to(p_dt).unsqueeze(0)).squeeze(0).to(x.dtype)
        u2 = torch.exp(out[0])
        u8 = torch.exp(out[1])

        self._prev_x = x.detach()
        self._prev_y = y.detach()
        return u2, u8
