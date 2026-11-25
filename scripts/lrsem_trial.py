# -*- coding: utf-8 -*-
"""The rank-one-residual search that the manuscript reports as a trial.

The first attempt set the coefficient to a constant, so adding an edge barely changed the
residual and the score never moved. Here each candidate structure is scored after a full
refit, which is what any score-based search does.
"""
import os, sys, time
import numpy as np

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, S)
from run_revision import pic_of, f1_topk, corr_scores
from exp_new import gen


def residuals(P, W):
    m, d = P.shape
    R = np.empty_like(P)
    for j in range(d):
        pa = np.where(W[:, j] != 0)[0]
        if len(pa) == 0:
            R[:, j] = P[:, j] - P[:, j].mean()
        else:
            Z = P[:, pa]
            beta, *_ = np.linalg.lstsq(Z, P[:, j], rcond=None)
            R[:, j] = P[:, j] - Z @ beta - (P[:, j] - Z @ beta).mean()
    return R


def score(P, W, pen):
    m, d = P.shape
    R = residuals(P, W)
    mu = np.linalg.eigvalsh((R.T @ R) / m)
    sig2 = max(mu[:-1].mean(), 1e-12)
    ll = -0.5 * m * ((d - 1) * np.log(sig2) + np.log(max(mu[-1], 1e-12)))
    return -2.0 * ll + pen * (int((W != 0).sum()) + d + 1)


def acyclic(W):
    d = W.shape[0]
    A = (W != 0).astype(float)
    return np.allclose(np.linalg.matrix_power(A, d), 0)


def search(P, max_moves=300):
    m, d = P.shape
    pen = np.log(m)
    W = np.zeros((d, d))
    cur = score(P, W, pen)
    for _ in range(max_moves):
        best = (cur, None)
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                Wn = W.copy()
                Wn[i, j] = 0.0 if W[i, j] != 0 else 1.0
                if not acyclic(Wn):
                    continue
                s = score(P, Wn, pen)
                if s < best[0] - 1e-9:
                    best = (s, Wn)
        if best[1] is None:
            break
        cur, W = best
    return W, cur


if __name__ == '__main__':
    print(__doc__)
    print('The numbers quoted in the manuscript are produced by e21_lrsem.py,')
    print('which runs this search at lambda = 0 and 2 and writes results/e21_lrsem.json.')
