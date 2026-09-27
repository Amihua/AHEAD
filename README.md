# AHEAD: Adaptive Neural Acceleration for Large-Scale Linear Programming


```
AHEAD/
├── common/
│   └── cuPDLP_torch/              torch PDHG unroll package, shared by both
│                                  sections below
├── mprop_step_policy/             AHEAD's step-size policy, mprop family
│                                  (the paper's main Gurobi-backend table)
└── pagerank_100k/                 AHEAD + PDHG-Net baseline, PageRank @ n=100K
                                   (the PageRank scale-comparison figure)
```

Each section has its own `README.md` with exact reproduction steps
(`mprop_step_policy/README.md`, `pagerank_100k/README.md`). This file is
just the map between them.

## `common/cuPDLP_torch/`

Both `mprop_step_policy/` and `pagerank_100k/ahead/` import it via
`sys.path.insert(0, "../common")` (one level up for the former, two levels
for the latter) — each entry-point script does this itself, no manual
`PYTHONPATH` setup needed. Built for **Python 3.12 / x86_64 / Linux**;
rebuild with `common/cuPDLP_torch/build_eigen_spmv.sh` on a different
platform.

## `mprop_step_policy/`

The frozen learned step-size policy (`KFreeStepPolicyNorm`, 38,850 params)
behind the "AHEAD" numbers in the paper's main Gurobi-backend acceleration
table (`tab:accel-summary`), trained on 197 presolved mprop instances
(certified exact `(x*, y*)` labels). Training data is **not bundled**
(~714 MB) — regenerate it from code + fixed seeds instead
(`data_generation/gen_presolved_train_data.py`).

```
mprop_step_policy/
├── README.md
├── train_step_policy.py            Step 2: train the policy
├── train_kfree_normfeat_large_simple.py
├── train_kfree_ddpm_ip.py
├── scaling_n_ip.py
├── data_generation/                Step 1: generate the 197-instance
│   └── ...                         training set (needs gurobipy + ortools)
└── checkpoints/
    ├── kfree_ddpm_style_presolved_ddpm.pt   <- the checkpoint
    └── train_presolved_ddpm_log.json
```

## `pagerank_100k/`

Two independently-trained checkpoints at the PageRank-100K scale
(Barabási–Albert, `n=100000`, attachment parameter `m_BA=3`,
damping `α=0.85`), used together for the AHEAD-vs-PDHG-Net scale-comparison
figure. Only the 100K point is included (not the 1K/10K/50K checkpoints
also shown in that figure).

```
pagerank_100k/
├── README.md
├── ahead/                          AHEAD's step-size policy @ PageRank-100K
│   ├── train_pagerank_policy.py    Step 2: train
│   ├── train_kfree_normfeat_large_simple.py
│   ├── data_generation/            Step 1: generate eps=1e-8 labels (needs
│   │   └── ...                     only ortools.pdlp, no Gurobi)
│   └── checkpoints/
│       └── kfree_ddpm_style_pagerank100k.pt   <- the checkpoint
└── pdhgnet/                        PDHG-Net (ICML'24) baseline @ PageRank-100K
    ├── src/                        genuinely-unmodified upstream repo code
    │   ├── gen_ins.py              + 4 explicitly-documented minimal fixes
    │   ├── model.py                (see pagerank_100k/README.md for exactly
    │   └── train.py                what each one changes and why)
    └── model/
        └── best_model.mdl          <- the checkpoint
```


