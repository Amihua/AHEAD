#!/usr/bin/env python3
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PKG = Path(__file__).resolve().parent
COMMON = PKG.parent.parent / "common"
ICLR = PKG
SCRATCH_DIR = PKG / "data_generation"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(SCRATCH_DIR))
sys.path.insert(0, str(COMMON))

from cuPDLP_torch.unroll import OrtUnrollContext, pdhg_unroll
from train_kfree_normfeat_large_simple import KFreeStepPolicyNorm
import scaling_n_pagerank as SPR

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
_K_CHOICES_OVERRIDE = os.environ.get("PAGERANK_K_CHOICES")
K_CHOICES = [int(k) for k in _K_CHOICES_OVERRIDE.split(",")] if _K_CHOICES_OVERRIDE else [8, 16, 32, 64, 128]
BATCH_SIZE = 8
EPOCHS = 200
LR = 1e-3
LAMBDA_Y = 1.0
CLIP_TERM = 20.0
WS_EVERY = 10

PAGERANK_N_NODES = int(os.environ.get("PAGERANK_N_NODES", "10000"))
PAGERANK_N_TRAIN = int(os.environ.get("PAGERANK_N_TRAIN", "24"))
PAGERANK_DEGREE = 3
PAGERANK_DAMPING = 0.85


def _size_tag(n):
    return {1000: "1k", 10000: "10k", 50000: "50k", 100000: "100k", 1_000_000: "1m"}.get(n, f"{n}")


TAG = _size_tag(PAGERANK_N_NODES)
HIGHPREC_PAGERANK = SCRATCH_DIR / f"highprec_labels_pagerank{TAG}_train{PAGERANK_N_TRAIN}.json"
PAGERANK_TRAIN_SEEDS = list(range(PAGERANK_N_TRAIN))


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


def load_pagerank_cases():
    highprec = json.loads(HIGHPREC_PAGERANK.read_text())
    assert len(PAGERANK_TRAIN_SEEDS) == len(highprec)
    cases = []
    for seed, rec in zip(PAGERANK_TRAIN_SEEDS, highprec):
        rng = np.random.default_rng(seed)
        A_ineq, b_ineq, c, S, m = SPR.generate_instance(rng, PAGERANK_N_NODES, PAGERANK_DEGREE, PAGERANK_DAMPING)
        qp = SPR.build_qp_full(A_ineq, b_ineq, c, S)
        ctx = OrtUnrollContext(qp, DEV)
        x_star = np.array(rec["eps_1e-08"]["x_star"], dtype=np.float64)
        y_star = np.array(rec["eps_1e-08"]["y_star"], dtype=np.float64)
        ctx.x_target = torch.as_tensor(x_star, dtype=DTYPE, device=DEV) / ctx.col_scaling
        ctx.y_target = torch.as_tensor(y_star, dtype=DTYPE, device=DEV) / ctx.row_scaling
        ctx.x_target_norm2 = ctx.x_target.pow(2).sum().item()
        ctx.y_target_norm2 = ctx.y_target.pow(2).sum().item()
        cases.append({"qp": qp, "ctx": ctx, "tag": f"pagerank{TAG}_seed{seed}"})
    return cases


WARM_START_TAG = os.environ.get("PAGERANK_WARM_START_TAG", "")
SEED = int(os.environ.get("PAGERANK_SEED", "0"))
EPOCHS_OVERRIDE = os.environ.get("PAGERANK_EPOCHS")
if EPOCHS_OVERRIDE:
    EPOCHS = int(EPOCHS_OVERRIDE)


def main():
    torch.manual_seed(SEED)
    policy = KFreeStepPolicyNorm(step_hidden=(64, 64), state_hidden=64, n_layers=2).to(DEV)
    if WARM_START_TAG:
        ck = torch.load(PKG / "checkpoints" / f"kfree_ddpm_style_pagerank{WARM_START_TAG}.pt", map_location="cpu", weights_only=False)
        policy.load_state_dict(ck["policy"])
        init_desc = f"WARM-STARTED from kfree_ddpm_style_pagerank{WARM_START_TAG}.pt (same family, cross-scale transfer)"
    else:
        xavier_init_policy(policy)
        init_desc = "FROM SCRATCH Xavier init"
    print(f"size={TAG} (n={PAGERANK_N_NODES})  policy params: {policy.n_params()}  "
          f"{init_desc}  seed={SEED}  K_CHOICES={K_CHOICES}  BATCH_SIZE={BATCH_SIZE}  "
          f"EPOCHS={EPOCHS}  LR={LR}", flush=True)

    t0_data = time.perf_counter()
    all_cases = load_pagerank_cases()
    n_cases = len(all_cases)
    print(f"loaded {n_cases} PageRank-{TAG} cases with 1e-8 targets  ({time.perf_counter()-t0_data:.0f}s)", flush=True)

    best_loss = float("inf")
    best_state = None
    hist = []
    log_path = (PKG / "checkpoints") / f"train_kfree_ddpm_style_pagerank_{TAG}{os.environ.get('PAGERANK_OUT_SUFFIX', '')}_log.json"
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
        print(f"  [{ep:>4}]  loss={tr:.5g}  ({time.perf_counter()-t0:.0f}s){star}", flush=True)

    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    rng_shuffle = np.random.default_rng(0)
    rng_k = np.random.default_rng(1)

    for ep in range(EPOCHS):
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
        if (ep + 1) % WS_EVERY == 0 or ep == EPOCHS - 1:
            maybe_save(ep + 1, tot)

    policy.load_state_dict(best_state)
    print(f"\ntraining done. best loss = {best_loss:.4e}", flush=True)

    out_suffix = os.environ.get("PAGERANK_OUT_SUFFIX", "")
    ckpt_path = PKG / "checkpoints" / f"kfree_ddpm_style_pagerank{TAG}{out_suffix}.pt"
    torch.save({"policy": best_state, "step_hidden": (64, 64), "state_hidden": 64, "n_layers": 2,
                "K_CHOICES": K_CHOICES, "pagerank_n_nodes": PAGERANK_N_NODES}, ckpt_path)
    print(f"saved -> {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
