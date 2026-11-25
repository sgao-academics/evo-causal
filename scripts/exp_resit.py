# -*- coding: utf-8 -*-
"""E29  A nonlinear, additive-noise family on the same grid.

Every procedure in the main comparison assumes a linear-Gaussian mechanism with independent
errors. A reader is entitled to ask whether the wall belongs to the covariance or to that
assumption, so one family that drops the linearity is measured here: the regression-based
additive-noise model (RESIT).

For an ordered pair (i, j) a nonlinear regression of X_j on X_i is fitted and the dependence of
its residual on X_i is measured by the U-centred distance correlation of Szekely and Rizzo, which
has no floor at this sample size. A near-zero value means that direction explains the pair. The
pair is scored by the larger of the two directional values,

    s_ij = max( dcor(res_{j|i}, X_i), dcor(res_{i|j}, X_j) ),

because a pair that one nonlinear function describes shows dependence in the direction the
function does not run, while a pair of unrelated traits shows none either way. Ranking the top-k
entries of that matrix is the selection rule applied to every method in the paper: same k, same
metric, same data.

The learner is a shallow random forest, fixed before the grid was run. An over-flexible
regression makes every residual independent and would manufacture edges out of overfitting; the
depth here is deliberately modest, and it is not tuned on the outcome.
"""
import os, sys, json, time
import numpy as np

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(S), 'scripts'))
sys.path.insert(0, S)
from exp_new import gen, OUT, prf
from run_revision import pic_of, auroc
from methods_unified import topk_adj

from sklearn.ensemble import RandomForestRegressor

LAMS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)


def udcor(x, y):
    """U-centred distance correlation (Szekely and Rizzo 2014)."""
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    n = len(x)
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    A = a - a.sum(1)[:, None] / (n - 2) - a.sum(0)[None, :] / (n - 2) + a.sum() / ((n - 1) * (n - 2))
    B = b - b.sum(1)[:, None] / (n - 2) - b.sum(0)[None, :] / (n - 2) + b.sum() / ((n - 1) * (n - 2))
    np.fill_diagonal(A, 0.0)
    np.fill_diagonal(B, 0.0)
    dcov = (A * B).sum() / (n * (n - 3))
    dvx = (A * A).sum() / (n * (n - 3))
    dvy = (B * B).sum() / (n * (n - 3))
    if dvx <= 0 or dvy <= 0:
        return 0.0
    return float(np.sqrt(max(dcov, 0.0) / np.sqrt(dvx * dvy)))


def learner():
    return RandomForestRegressor(n_estimators=40, max_depth=4, min_samples_leaf=20,
                                 max_features=1.0, random_state=0, n_jobs=1)


def anm_scores(P):
    d = P.shape[1]
    D = np.zeros((d, d))
    for i in range(d):
        xi = P[:, i:i + 1]
        for j in range(i + 1, d):
            xj = P[:, j:j + 1]
            r1 = P[:, j] - learner().fit(xi, P[:, j]).predict(xi)
            r2 = P[:, i] - learner().fit(xj, P[:, i]).predict(xj)
            D[i, j] = udcor(r1, P[:, i])
            D[j, i] = udcor(r2, P[:, j])
    S_ = np.maximum(D, D.T)
    np.fill_diagonal(S_, 0.0)
    return S_


def main():
    name = 'e29_resit'
    p = os.path.join(OUT, name + '.json')
    res = json.load(open(p, encoding='utf-8')) if os.path.exists(p) else {}
    # K is a property of the generated data and does not depend on the method; the values are
    # taken from the same grid in e23 so that the two tables share one axis
    e23 = json.load(open(os.path.join(OUT, 'e23_families10.json'), encoding='utf-8'))
    for lam in LAMS:
        key = 'lam%.1f' % lam
        if key in res:
            print('skip', key, flush=True)
            continue
        t0 = time.time()
        f1, au = [], []
        for sd in range(5):
            X, W, tree, C = gen(200, 30, 10, lam, sd)
            P = pic_of(X, tree)
            Sm = anm_scores(P)
            f1.append(prf(topk_adj(Sm, 10), W)[2])
            au.append(auroc(Sm, W))
        res[key] = dict(d=30, lam=lam, seeds=5,
                        f1=round(float(np.mean(f1)), 3),
                        f1_sd=round(float(np.std(f1, ddof=1)), 3),
                        auroc=round(float(np.mean(au)), 4),
                        K=e23.get(key, {}).get('K'),
                        sec=round(time.time() - t0, 1))
        json.dump(res, open(p, 'w', encoding='utf-8'), indent=1)
        r = res[key]
        print('  [E29] lam=%.1f  F1=%.3f(%.3f)  AUROC=%.3f  K=%s  (%ds)'
              % (lam, r['f1'], r['f1_sd'], r['auroc'], r['K'], r['sec']), flush=True)
    print('saved')


if __name__ == '__main__':
    if '--quick' in sys.argv:
        t0 = time.time()
        X, W, tree, C = gen(200, 30, 10, 1.0, 0)
        P = pic_of(X, tree)
        Sm = anm_scores(P)
        print('one dataset: F1=%.3f AUROC=%.4f (%.0fs)'
              % (prf(topk_adj(Sm, 10), W)[2], auroc(Sm, W), time.time() - t0))
    else:
        main()
