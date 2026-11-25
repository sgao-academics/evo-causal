# -*- coding: utf-8 -*-
"""E29  The right null for the real matrices: a sparse causal graph, not independence.

E26 compares a real contrast correlation matrix with independent Brownian traits. That
comparison is necessary but weak, because any set of functionally related traits violates
independence -- including the one the analyst hopes to recover. The question the mechanism
actually needs is whether the excess is larger than a *sparse causal graph* would produce
at the same n and d.

So for each block this block simulates random DAGs with d/2, d and 2d edges, draws n
independent-error samples from each, and records the distribution of the leading ratio.
The real value is then placed against that distribution.

This is also the null the protocol tells an analyst to calibrate against, so reporting it
here is the real-data version of the calibration the paper recommends.

Run: python exp_graphnull.py
"""
import os, sys, json
import numpy as np

S = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(os.path.dirname(S), 'scripts')
sys.path.insert(0, S)
sys.path.insert(0, R)

from run_revision import make_dag
from exp_new import load, save, log

R_DAG = 400


def corr_of(M):
    M = np.asarray(M, float)
    M = M - M.mean(0)
    sd = M.std(0)
    sd[sd < 1e-12] = 1.0
    return np.corrcoef(M / sd, rowvar=False)


def lead_ratio(Rm):
    w = np.linalg.eigvalsh(Rm)[::-1]
    return float(w[0] / max(w[1:].mean(), 1e-12))


def graph_null(n, d, k, reps=R_DAG, seed=77):
    rng = np.random.default_rng(seed + k)
    vals = np.empty(reps)
    for b in range(reps):
        W, _ = make_dag(d, k, rng)
        A = np.linalg.inv(np.eye(d) - W)
        X = rng.normal(size=(n, d)) @ A
        vals[b] = lead_ratio(corr_of(X))
    return vals


def main():
    name_in = sys.argv[1] if len(sys.argv) > 1 else 'e26_real'
    res = load(name_in)
    for name, o in res.items():
        if 'graph_null' in o:
            log('skip', name)
            continue
        if 'contrast' not in o:
            continue
        n, d = int(o['n']), int(o['d'])
        r1 = float(o['contrast']['r'][0])
        gn = {}
        # sparse densities only: the paper's own grid is 10 edges at d = 30, and a graph with
        # two edges per variable is 80% complete at d = 6, which no comparative study posits
        for tag, k in (('sparse', max(2, int(round(d / 3.0)))),
                       ('medium', max(3, int(round(2 * d / 3.0)))),
                       ('dense', d)):
            v = graph_null(n, d, k)
            gn[tag] = dict(k=int(k), mean=round(float(v.mean()), 3),
                           sd=round(float(v.std()), 3),
                           q95=round(float(np.quantile(v, 0.95)), 3),
                           max=round(float(v.max()), 3),
                           p=round(float((1 + np.sum(v >= r1)) / (len(v) + 1)), 5))
            log('  %-14s k=%2d  null r1 %.2f+-%.2f  q95 %.2f  max %.2f  | real %.2f  p=%.4f'
                % (name, k, v.mean(), v.std(), np.quantile(v, 0.95), v.max(), r1, gn[tag]['p']))
        o['graph_null'] = gn
        res[name] = o
        save(name_in, res)


if __name__ == '__main__':
    main()
