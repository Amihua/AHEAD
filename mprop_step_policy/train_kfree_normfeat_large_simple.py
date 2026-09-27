#!/usr/bin/env python3
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
COMMON = ROOT.parent / "common"
PKG = ROOT
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(COMMON))

from cuPDLP_torch.packing_data import list_packing_files, load_packing_pkl
from cuPDLP_torch.pdhg import AdaptiveStepsizeParams, PdhgParameters, optimize
from cuPDLP_torch.policy import _osc_features_norm, _OSC_N_FEAT
from cuPDLP_torch.saddle_point import RestartParameters, RestartScheme
from cuPDLP_torch.termination import TerminationCriteria
from cuPDLP_torch.unroll import OrtUnrollContext, pdhg_unroll, pdlp_metric_loss

CPU = torch.device("cpu")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-4
K_TRAIN = 32
K_EXTRA_EVAL = 64
STAGE1_EPOCHS, STAGE2_EPOCHS = 150, 300
LR1, LR2 = 5e-3, 3e-3
WS_EVERY = 15
LAMBDA_Y = 1.0

TRAIN_DIR = ROOT / "data" / "packing_large_train"
VALID_DIR = ROOT / "data" / "packing_large_valid"


def solver_params(eps):
    return PdhgParameters(
        verbosity=0, termination_evaluation_frequency=1,
        step_size_policy_params=AdaptiveStepsizeParams(check_frequency=1),
        restart_params=RestartParameters(restart_scheme=RestartScheme.NO_RESTARTS),
        termination_criteria=TerminationCriteria(
            eps_optimal_absolute=eps, eps_optimal_relative=eps,
            iteration_limit=2_000_000))


class KFreeStepPolicyNorm(nn.Module):

    def __init__(self, step_hidden=(64, 64), state_hidden: int = 64, n_layers: int = 2):
        super().__init__()
        step_dims = [_OSC_N_FEAT] + list(step_hidden)
        step_layers = []
        for a, b in zip(step_dims[:-1], step_dims[1:]):
            step_layers += [nn.Linear(a, b), nn.Tanh()]
        self.step_mlp = nn.Sequential(*step_layers)
        self.state_encoder = nn.Sequential(
            nn.Linear(_OSC_N_FEAT, state_hidden), nn.Tanh())
        combo_dim = state_hidden + step_hidden[-1]
        trunk = []
        for _ in range(n_layers):
            trunk += [nn.Linear(combo_dim, combo_dim), nn.Tanh()]
        self.trunk = nn.Sequential(*trunk)
        self.head = nn.Linear(combo_dim, 2)
        self.double()
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.m = [None, None, None, None]

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def reset(self):
        self.m = [None, None, None, None]

    def __call__(self, k, x, y, ATy, ctx):
        feat = _osc_features_norm(x, y, ATy, self.m, ctx)
        step_emb = self.step_mlp(feat)
        state_emb = self.state_encoder(feat)
        h = torch.cat([state_emb, step_emb])
        h = self.trunk(h)
        out = self.head(h)
        self.m = [x.detach(), y.detach(), self.m[0], self.m[1]]
        return torch.exp(out[0]), torch.exp(out[1])


def main():
    torch.manual_seed(0)
    policy = KFreeStepPolicyNorm(step_hidden=(64, 64), state_hidden=64, n_layers=2).to(DEV)
    print(f"KFreeStepPolicyNorm params: {policy.n_params()}", flush=True)

    train_files = list_packing_files(TRAIN_DIR)
    train_qps = [load_packing_pkl(f) for f in train_files]
    print(f"train: {len(train_qps)} problems (packing_large_train)  device={DEV}", flush=True)

    colds_train = [optimize(solver_params(EPS), qp, device=CPU, ort_exact=True).iteration_count
                   for qp in train_qps]
    print(f"cold mean (train) = {np.mean(colds_train):.1f}", flush=True)

    import gzip, pickle
    train_ctxs = []
    for f, qp in zip(train_files, train_qps):
        ctx = OrtUnrollContext(qp, DEV)
        with gzip.open(f, "rb") as fh:
            obj = pickle.load(fh)
        x_star = obj["label"].numpy().astype(np.float64).ravel()
        y_star = obj["dual"].numpy().astype(np.float64).ravel()
        ctx.x_target = torch.as_tensor(x_star, dtype=ctx.dtype, device=DEV) / ctx.col_scaling
        ctx.y_target = torch.as_tensor(y_star, dtype=ctx.dtype, device=DEV) / ctx.row_scaling
        ctx.x_target_norm2 = ctx.x_target.pow(2).sum().item()
        ctx.y_target_norm2 = ctx.y_target.pow(2).sum().item()
        train_ctxs.append(ctx)

    def _soft_cap(term, clip=20.0):
        scale = (clip / (term.detach() + 1e-12)).clamp(max=1.0)
        return term * scale

    def dist_loss_normalized(ctx, x, y, lambda_y=LAMBDA_Y):
        eps_ = 1e-8
        lx = (x - ctx.x_target).pow(2).sum() / (ctx.x_target_norm2 + eps_)
        ly = (y - ctx.y_target).pow(2).sum() / (ctx.y_target_norm2 + eps_)
        return _soft_cap(lx) + lambda_y * _soft_cap(ly)

    def mean_ws(ctxs, qps, colds, K):
        tot = 0
        for qp, ctx in zip(qps, ctxs):
            policy.reset()
            with torch.no_grad():
                x, y, _ = pdhg_unroll(ctx, K, u_fn=policy)
            xo, yo = ctx.unscale(x, y)
            xo = xo.detach().cpu().numpy(); yo = yo.detach().cpu().numpy()
            res = optimize(solver_params(EPS), qp, device=CPU,
                            x_init=np.clip(xo, 0, None), y_init=yo, ort_exact=True)
            tot += res.iteration_count if res.termination_reason.name == "OPTIMAL" else colds[0]
        return tot / len(qps)

    best_ws = float("inf")
    best_state = None
    hist = []
    log_path = Path(__file__).resolve().parent / "train_kfree_normfeat_large_simple_log.json"
    t0 = time.perf_counter()

    def maybe_save(ep, tr):
        nonlocal best_ws, best_state
        w = mean_ws(train_ctxs, train_qps, colds_train, K_TRAIN)
        star = ""
        if w < best_ws:
            best_ws = w
            best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
            star = "  *best"
        hist.append({"epoch": ep, "loss": tr, "mean_ws": w})
        log_path.write_text(json.dumps(
            {"cold_mean": float(np.mean(colds_train)), "best_ws": best_ws, "hist": hist}, indent=1))
        print(f"  [{ep:>10}]  loss={tr:.5g}  mean_ws={w:.2f}  ({time.perf_counter()-t0:.0f}s){star}", flush=True)

    def run_stage(stage, epochs, lr, loss_kind):
        opt = torch.optim.Adam(policy.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        for ep in range(epochs):
            opt.zero_grad()
            tot = 0.0
            for ctx in train_ctxs:
                policy.reset()
                x, y, ATy = pdhg_unroll(ctx, K_TRAIN, u_fn=policy)
                loss = (dist_loss_normalized(ctx, x, y) if loss_kind == "dist"
                        else pdlp_metric_loss(ctx, x, y, ATy)) / len(train_ctxs)
                loss.backward()
                tot += loss.item()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            sched.step()
            if (ep + 1) % WS_EVERY == 0 or ep == epochs - 1:
                maybe_save(f"{stage}-{ep + 1}", tot)

    print("\n[stage 1] dist_loss_normalized", flush=True)
    run_stage("dist", STAGE1_EPOCHS, LR1, "dist")
    if best_state is not None:
        policy.load_state_dict(best_state)
        print(f"  reloaded best-of-stage1 (mean_ws={best_ws:.2f})", flush=True)
    print("\n[stage 2] pdlp_metric_loss polish", flush=True)
    run_stage("pdlp", STAGE2_EPOCHS, LR2, "pdlp")

    policy.load_state_dict(best_state)
    print(f"\ntraining done. cold={np.mean(colds_train):.1f}  best mean_ws(train, K={K_TRAIN})={best_ws:.2f}  "
          f"reduction={100*(1-best_ws/np.mean(colds_train)):.1f}%", flush=True)

    ckpt_path = PKG / "model" / "kfree_normfeat_large_simple.pt"
    torch.save({"policy": best_state, "step_hidden": (64, 64), "state_hidden": 64,
                "n_layers": 2, "K_TRAIN": K_TRAIN}, ckpt_path)
    print(f"saved -> {ckpt_path}", flush=True)

    for K_eval in (K_TRAIN, K_EXTRA_EVAL):
        print(f"\n[eval] packing_large_valid (10, held-out), K={K_eval}", flush=True)
        valid_files = list_packing_files(VALID_DIR)
        valid_qps = [load_packing_pkl(f) for f in valid_files]
        valid_colds = [optimize(solver_params(EPS), qp, device=CPU, ort_exact=True).iteration_count
                       for qp in valid_qps]
        ws_list = []
        for qp, f in zip(valid_qps, valid_files):
            ctx = OrtUnrollContext(qp, DEV)
            policy.reset()
            with torch.no_grad():
                x, y, _ = pdhg_unroll(ctx, K_eval, u_fn=policy)
            xo, yo = ctx.unscale(x, y)
            xo = xo.detach().cpu().numpy(); yo = yo.detach().cpu().numpy()
            res = optimize(solver_params(EPS), qp, device=CPU,
                            x_init=np.clip(xo, 0, None), y_init=yo, ort_exact=True)
            ws_list.append(res.iteration_count if res.termination_reason.name == "OPTIMAL" else None)

        valid_colds_arr = np.array(valid_colds, dtype=float)
        ws_arr = np.array([w if w is not None else c for w, c in zip(ws_list, valid_colds)], dtype=float)
        print(f"mean cold = {valid_colds_arr.mean():.2f}   mean ws(K={K_eval}) = {ws_arr.mean():.2f}   "
              f"reduction = {100*(1-ws_arr.mean()/valid_colds_arr.mean()):+.1f}%   "
              f"win_rate = {100*np.mean(ws_arr < valid_colds_arr):.1f}%", flush=True)

    print(f"\n[reference] StepEmbeddingPolicy fixed K32 = 17.70(+89.2%)   oracle k* = 6.00(+96.4%)")


if __name__ == "__main__":
    main()
