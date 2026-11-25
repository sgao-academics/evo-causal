# -*- coding: utf-8 -*-
"""E28  What the screen costs when the calibration graph is not the truth.

The protocol says: simulate from the graph you are about to propose, take the 95th
percentile of r under that model as the threshold. The Scope section concedes that an
analyst only has a graph they fitted, which is wrong exactly when it matters. This block
measures the concession instead of asserting it, and finds that it is worse than conceded.

Five calibrations are compared on the same datasets, at sizes and dimensions where
comparative matrices actually live:

  iid        the null matrix the paper warns against: independent standard normals
  fitted     the top-k skeleton of the observed matrix, parameters estimated from it,
             at half, one and two times the true edge count
  random     sparse random DAGs at the same n, d and edge count, parameters drawn
             independently of the data -- a reference the analyst can build without
             fitting anything

For each, size (no confounder) and power (confounder present) of the same statistic.
A calibration that is useful has low size and high power; the diagonal on which they are
equal is worthless, and that is where fitted calibration lands.

The last block reports how many leading components have to be removed when the confounder
is more than one direction. Removing one is the whole of the current protocol, and it is
enough only in the one-factor case.

Run: python exp_calib.py
"""
import os, sys, json, time
import numpy as np

S = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(os.path.dirname(S), 'scripts')
sys.path.insert(0, S)
sys.path.insert(0, R)

from run_revision import make_dag
from methods_unified import topk_adj, symmetrise
from exp_new import load, save, log, prf

NSEED = 25
NSIM = 200
NS = (40, 80, 150, 200, 400)
DIMS = (6, 12)
WIDTHS = (('fit_half', 0.5), ('fit_one', 1.0), ('fit_two', 2.0))
NRAND = 400


def corr_of(M):
    M = np.asarray(M, float)
    M = M - M.mean(0)
    sd = M.std(0)
    sd[sd < 1e-12] = 1.0
    return np.corrcoef(M / sd, rowvar=False)


def lead_ratio(R):
    w = np.linalg.eigvalsh(R)[::-1]
    return float(w[0] / max(w[1:].mean(), 1e-12))


def gen(n, d, k, lam, seed, nfac=1):
    rng = np.random.default_rng(seed)
    W, _ = make_dag(d, k, rng)
    A = np.linalg.inv(np.eye(d) - W)
    X = rng.normal(size=(n, d)) @ A
    if lam > 0:
        F = rng.normal(size=(n, nfac))
        Lam = rng.normal(size=(nfac, d))
        X = X + lam * F @ Lam
    return (X - X.mean(0)) / (X.std(0) + 1e-12), W


def topk_mask(A, k):
    d = A.shape[0]
    iu = np.triu_indices(d, 1)
    keep = np.argsort(A[iu])[::-1][:max(k, 0)]
    M = np.zeros((d, d), bool)
    for t in keep:
        M[iu[0][t], iu[1][t]] = True
        M[iu[1][t], iu[0][t]] = True
    return M


def skeleton_to_cov(R, M):
    """Concentration matrix with the skeleton's zeros, parameters taken from R."""
    d = R.shape[0]
    Th = np.linalg.inv(R)
    Th = np.where(M | np.eye(d, dtype=bool), Th, 0.0)
    w = np.linalg.eigvalsh(Th)
    if w.min() <= 1e-6:
        Th = Th + (abs(w.min()) + 0.1) * np.eye(d)
    S = np.linalg.inv(Th)
    dd = np.sqrt(np.diag(S))
    return S / np.outer(dd, dd)


def q95_from_cov(S, n, seed, nsim=NSIM):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(S + 1e-9 * np.eye(S.shape[0]))
    vals = np.empty(nsim)
    for b in range(nsim):
        vals[b] = lead_ratio(corr_of(rng.normal(size=(n, S.shape[0])) @ L.T))
    return float(np.quantile(vals, 0.95))


def q95_random(n, d, k, seed=4242, reps=NRAND):
    rng = np.random.default_rng(seed)
    vals = np.empty(reps)
    for b in range(reps):
        W, _ = make_dag(d, k, rng)
        A = np.linalg.inv(np.eye(d) - W)
        vals[b] = lead_ratio(corr_of(rng.normal(size=(n, d)) @ A))
    return float(np.quantile(vals, 0.95)), float(vals.mean()), float(vals.max())


def calib():
    out = {}
    for d in DIMS:
        # the paper's own density: k = 10 edges at d = 30. Two edges per variable would make
        # the graph 80% complete at d = 6, which is not a sparse model of anything.
        k = max(2, int(round(d / 3.0)))
        for n in NS:
            th_rand, mu_rand, mx_rand = q95_random(n, d, k)
            out['rand_ref_d%d_n%d' % (d, n)] = dict(q95=round(th_rand, 3),
                                                    mean=round(mu_rand, 3),
                                                    max=round(mx_rand, 3), k=k)
            for lam in (0.0, 2.0):
                key = 'd%d_n%d_lam%.1f' % (d, n, lam)
                fire = {t: 0 for t, _ in WIDTHS}
                fire['iid'] = 0
                fire['random'] = 0
                for sd in range(NSEED):
                    X, W = gen(n, d, k, lam, 10000 + sd)
                    R = corr_of(X)
                    r1 = lead_ratio(R)
                    iu = np.triu_indices(d, 1)
                    A = np.abs(R - np.eye(d))
                    if r1 > q95_from_cov(np.eye(d), n, 900 + sd):
                        fire['iid'] += 1
                    if r1 > th_rand:
                        fire['random'] += 1
                    for tag, mult in WIDTHS:
                        M = topk_mask(A, int(round(mult * k)))
                        if r1 > q95_from_cov(skeleton_to_cov(R, M), n, 900 + sd):
                            fire[tag] += 1
                out[key] = {t: round(fire[t] / NSEED, 3) for t in fire}
                log('%s  %s' % (key, '  '.join('%s=%.2f' % kv for kv in out[key].items())))
                save('e28_calib', out)
    return out


def twofactor():
    """How many components must come out when the confounder is more than one direction?"""
    out = {}
    d, k, n = 30, 10, 200
    for nfac in (1, 2):
        for lam in (1.5, 3.0):
            row = {}
            for j in (0, 1, 2):
                fs = []
                for sd in range(5):
                    X, W = gen(n, d, k, lam, 500 + sd, nfac=nfac)
                    R = corr_of(X)
                    for _ in range(j):
                        w, v = np.linalg.eigh(R)
                        i1 = int(np.argmax(w))
                        R = R - w[i1] * np.outer(v[:, i1], v[:, i1])
                    adj = topk_adj(np.abs(R), int((np.abs(W) > 0).sum()))
                    fs.append(prf(symmetrise(adj), W)[2])
                row['remove%d' % j] = round(float(np.mean(fs)), 3)
            out['fac%d_lam%.1f' % (nfac, lam)] = row
            log('  factors=%d lam=%.1f  %s' % (nfac, lam, row))
    save('e28_calib', dict(load('e28_calib'), twofactor=out))
    return out


if __name__ == '__main__':
    t0 = time.time()
    res = load('e28_calib')
    if not any(k.startswith('d') for k in res):
        calib()
    twofactor()
    log('done %.0fs' % (time.time() - t0))
