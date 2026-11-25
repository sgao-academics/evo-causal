# -*- coding: utf-8 -*-
"""E30  Multiplicity, and whether the verdict depends on preprocessing.

Two things the real-data section needs and did not have.

1.  Fifteen ratios (five blocks x three leading ratios) were reported without a
    multiplicity correction. Each is a one-sided test against a sparse-graph null, so the
    family is exactly defined and Holm applies. The effect of correcting is reported, not
    its absence.

2.  Each block is assembled by dropping species with an incomplete trait vector and by
    logging any trait spanning more than a factor of twenty. Both are choices. This script
    recomputes the leading ratio with (a) no logging at all, (b) each trait dropped in turn,
    and reports the range, so a reader can see whether a verdict rests on one column.

Run: python exp_holm.py
"""
import os, sys, json
import numpy as np

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(S), 'scripts'))
sys.path.insert(0, S)

import exp_real as E
from run_revision import make_dag
from fetch_real import read_pantheria

DATA = os.path.join(os.path.dirname(S), 'data')
# 5,000 draws rather than 500: with Holm over fifteen tests the smallest attainable p-value with
# 500 draws is 1/501 = 0.002, which sits a hair under 0.05/15. A null that cannot express a
# smaller probability cannot be corrected, so the resolution had to rise before the correction
# meant anything.
NDRAW = 5000


def corr_of(M):
    M = np.asarray(M, float)
    M = M - M.mean(0)
    sd = M.std(0)
    sd[sd < 1e-12] = 1.0
    return np.corrcoef(M / sd, rowvar=False)


def spec3(R):
    w = np.linalg.eigvalsh(R)[::-1]
    out = []
    for k in range(3):
        out.append(float(w[k] / max(w[k + 1:].mean(), 1e-12)))
    return np.array(out)


def graph_null_spec(n, d, k, reps=NDRAW, seed=77):
    """Sparse Gaussian DAG at the same n and d; returns (reps, 3) ratios."""
    rng = np.random.default_rng(seed + k)
    out = np.empty((reps, 3))
    for b in range(reps):
        W, _ = make_dag(d, k, rng)
        A = np.linalg.inv(np.eye(d) - W)
        X = rng.normal(size=(n, d)) @ A
        out[b] = spec3(corr_of(X))
    return out


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        v = (m - rank) * p[i]
        run = max(run, v)
        adj[i] = min(run, 1.0)
    return adj


def main():
    io = __import__('io')
    root = io.open(os.path.join(DATA, 'otol_mammalia.nwk'), encoding='utf-8').read()
    match = json.load(io.open(os.path.join(DATA, 'otol_match.json'), encoding='utf-8'))
    rows = read_pantheria()

    mammal, bird = E.load('e26_real'), E.load('e27_birds')
    out = {'blocks': {}, 'holes': {}}

    # ---------------- 1. the three leading ratios against the sparse-graph null
    test_names, test_p, keys = [], [], []
    for name, o in list(mammal.items()) + list(bird.items()):
        if 'contrast' not in o:
            continue
        n, d = int(o['n']), int(o['d'])
        k = max(2, int(round(d / 3.0)))
        obs = np.array(o['contrast']['r'][:3], float)
        null = graph_null_spec(n, d, k)
        p = [(1.0 + np.sum(null[:, j] >= obs[j])) / (len(null) + 1.0) for j in range(3)]
        out['blocks'][name] = dict(n=n, d=d, k=int(k), obs=[round(float(x), 3) for x in obs],
                                   null_mean=[round(float(x), 3) for x in null.mean(0)],
                                   null_sd=[round(float(x), 3) for x in null.std(0, ddof=1)],
                                   null_q95=[round(float(x), 3) for x in np.quantile(null, 0.95, axis=0)],
                                   p=[round(float(x), 4) for x in p])
        for j in range(3):
            keys.append('%s:r%d' % (name, j + 1))
            test_p.append(p[j])
        print('%-14s k=%2d  obs %s  null %s  p %s'
              % (name, k, np.round(obs, 2), np.round(null.mean(0), 2), np.round(p, 4)),
              flush=True)

    adj = holm(test_p)
    for key, p0, pa in zip(keys, test_p, adj):
        out['holes'][key] = dict(p=round(float(p0), 4), holm=round(float(pa), 4),
                                 survives=bool(pa < 0.05))
    print('Holm over %d tests: %d survive'
          % (len(test_p), sum(1 for a in adj if a < 0.05)), flush=True)

    # ---------------- 2. does a verdict rest on one column, or on the logging rule?
    for name in E.BLOCKS:
        if name not in mammal:
            continue
        blk = E.prep_block(name, rows, match, root, None)
        if blk is None:
            continue
        blk['block'], blk['match'] = name, match
        base = mammal[name]['contrast']['r'][0]
        # trimmed() applies the tree-based row filter, so the sensitivity variants must start
        # from the same rows it keeps rather than from the unpruned block
        X0, labels, C, species = E.trimmed(blk, root)
        cols = list(blk['cols'])

        # (a) no logging: undo log10 on the columns that were logged
        X = X0.copy()
        for j, c in enumerate(cols):
            if c in blk['logged']:
                X[:, j] = np.power(10.0, X[:, j])
        X = (X - X.mean(0)) / (X.std(0) + 1e-12)
        P = E.whiten(X, C)
        raw_r1 = float(E.spectrum(corr_of(P), 1)[0][0])

        # (b) every trait dropped in turn
        loo = {}
        for j, c in enumerate(cols):
            keep = [i for i in range(len(cols)) if i != j]
            if len(keep) < 4:
                continue
            Xj = X0[:, keep]
            Pj = E.whiten(Xj, C)
            loo[c] = round(float(E.spectrum(corr_of(Pj), 1)[0][0]), 3)

        vals = np.array(list(loo.values()))
        out.setdefault('preproc', {})[name] = dict(
            r1_headline=round(float(base), 3), r1_nolog=round(raw_r1, 3),
            loo_min=round(float(vals.min()), 3), loo_max=round(float(vals.max()), 3),
            loo=loo)
        print('%-14s headline %.2f | no log %.2f | leave-one-out %.2f--%.2f'
              % (name, base, raw_r1, vals.min(), vals.max()), flush=True)

    E.save('e30_holm', out)
    print('saved')


if __name__ == '__main__':
    main()
