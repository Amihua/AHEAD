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
ICLR_SCRIPTS = ROOT
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(COMMON))

from cuPDLP_torch.unroll import OrtUnrollContext, pdhg_unroll
from cuPDLP_torch.problem import QuadraticProgrammingProblem
from train_kfree_normfeat_large_simple import KFreeStepPolicyNorm
import scaling_n_ip as SIP

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
K_CHOICES = [8, 16, 32, 64, 128]
BATCH_SIZE = 8
LR = 1e-3
LAMBDA_Y = 1.0
CLIP_TERM = 20.0
WS_EVERY = 5

EPOCHS_DEFAULT = {"IP-S": 200, "IP-L": 80}


def xavier_init_policy(policy):
    for module in (policy.step_mlp, policy.state_encoder, policy.trunk):
        for layer in module:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    nn.init.zeros_(policy.head.weight)
    nn.init.zeros_(policy.head.bias)


def _soft_cap(term, clip=CLIP_TERM):
    scale = (clip / (term.detach() + 1e-12)).clamp(max=1.0)
    return term * scale


def dist_loss_1e8(ctx, x, y, lambda_y=LAMBDA_Y):
    eps_ = 1e-8
    lx = (x - ctx.x_target).pow(2).sum() / (ctx.x_target_norm2 + eps_)
    ly = (y - ctx.y_target).pow(2).sum() / (ctx.y_target_norm2 + eps_)
    return _soft_cap(lx) + lambda_y * _soft_cap(ly)


def build_qp(A, b, c, lb, ub, num_equalities):
    return QuadraticProgrammingProblem(
        variable_lower_bound=lb,
        variable_upper_bound=ub,
        objective_vector=c,
        objective_constant=0.0,
        constraint_matrix=A,
        right_hand_side=b,
        num_equalities=num_equalities,
    )


def load_cases(size, highprec_path):
    rows = json.loads(highprec_path.read_text())
    cases = []
    for rec in rows:
        rng = np.random.default_rng(rec["seed"])
        A, b, c, lb, ub, neq = SIP.generate_instance(rng, size)
        qp = build_qp(A, b, c, lb, ub, neq)
        ctx = OrtUnrollContext(qp, DEV)
        x_star = np.array(rec["x_star"], dtype=np.float64)
        y_star = np.array(rec["y_star"], dtype=np.float64)
        ctx.x_target = torch.as_tensor(x_star, dtype=DTYPE, device=DEV) / ctx.col_scaling
        ctx.y_target = torch.as_tensor(y_star, dtype=DTYPE, device=DEV) / ctx.row_scaling
        ctx.x_target_norm2 = ctx.x_target.pow(2).sum().item()
        ctx.y_target_norm2 = ctx.y_target.pow(2).sum().item()
        cases.append({"qp": qp, "ctx": ctx, "tag": f"{size}_seed{rec['seed']}"})
    return cases


def main():
    size = sys.argv[1] if len(sys.argv) > 1 else "IP-S"
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else {"IP-S": 20, "IP-L": 10}[size]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else EPOCHS_DEFAULT[size]
    tag = size.lower().replace("-", "")

    highprec_path = ICLR_SCRIPTS / f"highprec_labels_{tag}_train{n_train}.json"
    ckpt_path = PKG / "model" / f"kfree_ddpm_style_{tag}.pt"
    log_path = ICLR_SCRIPTS / f"train_kfree_ddpm_{tag}_log.json"

    torch.manual_seed(0)
    policy = KFreeStepPolicyNorm(step_hidden=(64, 64), state_hidden=64, n_layers=2).to(DEV)
    xavier_init_policy(policy)
    print(f"size={size}  n_train={n_train}  epochs={epochs}  policy params: {policy.n_params()}  "
          f"FROM SCRATCH Xavier init  K_CHOICES={K_CHOICES}  BATCH_SIZE={BATCH_SIZE}  LR={LR}", flush=True)

    t0_data = time.perf_counter()
    all_cases = load_cases(size, highprec_path)
    n_cases = len(all_cases)
    print(f"loaded {n_cases} {size} cases with 1e-8 targets  ({time.perf_counter()-t0_data:.0f}s)", flush=True)

    best_loss = float("inf")
    best_state = None
    hist = []
    t0 = time.perf_counter()

    def maybe_save(ep, tr):
        nonlocal best_loss, best_state
        star = ""
        if tr < best_loss:
            best_loss = tr
            best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
            star = "  *best"
        hist.append({"epoch": ep, "loss": tr})
        log_path.write_text(json.dumps({"best_loss": best_loss, "hist": hist}, indent=1))
        print(f"  [{ep:>4}/{epochs}]  loss={tr:.5g}  ({time.perf_counter()-t0:.0f}s){star}", flush=True)

    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng_shuffle = np.random.default_rng(0)
    rng_k = np.random.default_rng(1)

    for ep in range(epochs):
        order = rng_shuffle.permutation(n_cases)
        tot = 0.0
        for start in range(0, n_cases, BATCH_SIZE):
            batch_idx = order[start:start + BATCH_SIZE]
            K = int(rng_k.choice(K_CHOICES))
            opt.zero_grad()
            for idx in batch_idx:
                case = all_cases[idx]
                ctx = case["ctx"]
                policy.reset()
                x, y, _ = pdhg_unroll(ctx, K, u_fn=policy)
                loss = dist_loss_1e8(ctx, x, y) / len(batch_idx)
                loss.backward()
                tot += loss.item() * len(batch_idx) / n_cases
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
        sched.step()
        if (ep + 1) % WS_EVERY == 0 or ep == epochs - 1:
            maybe_save(ep + 1, tot)

    policy.load_state_dict(best_state)
    print(f"\ntraining done. best loss = {best_loss:.4e}", flush=True)

    torch.save({"policy": best_state, "step_hidden": (64, 64), "state_hidden": 64, "n_layers": 2,
                "K_CHOICES": K_CHOICES}, ckpt_path)
    print(f"saved -> {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
