#!/usr/bin/env python3
import json, os, sys, time
from pathlib import Path
import numpy as np, scipy.sparse as sp, torch, torch.nn as nn

ROOT = Path(__file__).resolve().parent
COMMON = ROOT.parent / "common"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(COMMON))

from cuPDLP_torch.unroll import OrtUnrollContext, pdhg_unroll
from cuPDLP_torch.problem import QuadraticProgrammingProblem
from train_kfree_normfeat_large_simple import KFreeStepPolicyNorm
from train_kfree_ddpm_ip import xavier_init_policy, dist_loss_1e8

DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DT = torch.float64
K_CHOICES = [8, 16, 32, 64, 128]
BATCH = 8
LR = float(os.environ.get("LR", "1e-3"))
EPOCHS = int(os.environ.get("EPOCHS", "60"))
TAG = os.environ.get("TAG", "presolved_ddpm")
MAX_N = int(os.environ.get("MAX_N", "70000"))
VAL_EVERY = 5
NVAL = 20

DATA_DIR = ROOT / "data" / "presolved_train"
CKPT_DIR = ROOT / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)


def load(path):
    z = np.load(path)
    A = sp.csr_matrix((z["A_data"], z["A_indices"], z["A_indptr"]), shape=tuple(z["A_shape"]))
    qp = QuadraticProgrammingProblem(variable_lower_bound=z["lb"], variable_upper_bound=z["ub"],
                                      objective_vector=z["c"], objective_constant=0.0,
                                      constraint_matrix=A, right_hand_side=z["b"],
                                      num_equalities=int(z["num_equalities"]))
    ctx = OrtUnrollContext(qp, DEV)
    T = lambda a: torch.as_tensor(a, dtype=DT, device=DEV)
    ctx.x_target = T(z["x_star"]) / ctx.col_scaling
    ctx.y_target = T(z["y_star"]) / ctx.row_scaling
    ctx.x_target_norm2 = ctx.x_target.pow(2).sum().item()
    ctx.y_target_norm2 = ctx.y_target.pow(2).sum().item()
    ctx.sup = np.where(z["x_star"] > 1e-9)[0]
    ctx.tag = path.stem
    return ctx


def main():
    files = sorted(DATA_DIR.glob("*.npz"))
    files = [f for f in files if int(np.load(f)["A_shape"][1]) <= MAX_N]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(files))
    val_f = [files[i] for i in perm[:NVAL]]
    tr_f = [files[i] for i in perm[NVAL:]]
    print(f"train {len(tr_f)} val {len(val_f)} cases (n_red <= {MAX_N})", flush=True)
    t0 = time.perf_counter()
    tr = [load(f) for f in tr_f]
    va = [load(f) for f in val_f]
    print(f"loaded ({time.perf_counter()-t0:.0f}s)", flush=True)

    torch.manual_seed(0)
    policy = KFreeStepPolicyNorm(step_hidden=(64, 64), state_hidden=64, n_layers=2).to(DEV)
    xavier_init_policy(policy)
    print("policy params", policy.n_params(), flush=True)

    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    rs = np.random.default_rng(0)
    rk = np.random.default_rng(1)
    best = (1e9, None)
    hist = []

    def evaluate():
        out = {}
        for K in (64, 128):
            L, rec = [], []
            with torch.no_grad():
                for ctx in va:
                    policy.reset()
                    x, y, _ = pdhg_unroll(ctx, K, u_fn=policy)
                    L.append(dist_loss_1e8(ctx, x, y).item())
                    xo = (x * ctx.col_scaling).clamp(min=0).cpu().numpy()
                    top = np.argsort(-xo)[:20]
                    rec.append(np.isin(ctx.sup, top).mean())
            out[K] = (float(np.mean(L)), float(np.mean(rec)))
        return out

    for ep in range(EPOCHS):
        order = rs.permutation(len(tr))
        tot = 0.0
        for s in range(0, len(tr), BATCH):
            K = int(rk.choice(K_CHOICES))
            opt.zero_grad()
            for i in order[s:s + BATCH]:
                ctx = tr[i]
                policy.reset()
                x, y, _ = pdhg_unroll(ctx, K, u_fn=policy)
                loss = dist_loss_1e8(ctx, x, y) / BATCH
                loss.backward()
                tot += loss.item()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
        sched.step()
        msg = f"[{ep+1}/{EPOCHS}] train {tot/max(1,len(tr)//BATCH):.4f} ({time.perf_counter()-t0:.0f}s)"
        if (ep + 1) % VAL_EVERY == 0 or ep == EPOCHS - 1:
            ev = evaluate()
            score = ev[128][0] + ev[64][0]
            msg += " | VAL " + " ".join(f"K{K}: loss {v[0]:.3f} recall@20 {v[1]:.2f}" for K, v in ev.items())
            if score < best[0]:
                best = (score, {k: v.detach().clone() for k, v in policy.state_dict().items()})
                msg += " *best"
            hist.append(dict(epoch=ep + 1, val=ev))
        print(msg, flush=True)

    torch.save({"policy": best[1], "step_hidden": (64, 64), "state_hidden": 64, "n_layers": 2,
                "K_CHOICES": K_CHOICES}, CKPT_DIR / f"kfree_ddpm_style_{TAG}.pt")
    (ROOT / f"train_{TAG}_log.json").write_text(json.dumps(hist, indent=1))
    print("saved", TAG)


if __name__ == "__main__":
    main()
