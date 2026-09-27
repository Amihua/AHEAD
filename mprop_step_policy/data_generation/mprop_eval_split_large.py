#!/usr/bin/env python3
import numpy as np

N_TRAIN = 200
N_TEST = 50
N_TOTAL = N_TRAIN + N_TEST
N_MIN, N_MAX = 100_000, 1_000_000
SPLIT_SEED = 12345


def _make_pairs():
    rng = np.random.default_rng(SPLIT_SEED)
    log_n = rng.uniform(np.log(N_MIN), np.log(N_MAX), size=N_TOTAL)
    ns = np.round(np.exp(log_n)).astype(int)
    seeds = rng.integers(1, 2**31 - 1, size=N_TOTAL)
    return list(zip(ns.tolist(), seeds.tolist()))


_ALL_PAIRS = _make_pairs()
MPROP_TRAIN_PAIRS = _ALL_PAIRS[:N_TRAIN]
MPROP_TEST_PAIRS = _ALL_PAIRS[N_TRAIN:]

assert len(MPROP_TRAIN_PAIRS) == N_TRAIN
assert len(MPROP_TEST_PAIRS) == N_TEST
