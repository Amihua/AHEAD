# PageRank-100K checkpoints — minimal reproducible release



```
ahead/      AHEAD's learned step-size policy, PageRank-100K
pdhgnet/    PDHG-Net (ICML'24) baseline, PageRank-100K, upstream repo code
```

## `ahead/` — AHEAD step-size policy

Same `KFreeStepPolicyNorm` architecture as the mprop release
(`../mprop_step_policy/`), trained from scratch (Xavier init) on
PageRank-100K's own eps=1e-8 labels instead of mprop's.




