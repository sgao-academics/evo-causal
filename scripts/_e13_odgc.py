# -*- coding: utf-8 -*-
"""E13  A faithful implementation of O(D)GC, the workflow proposed by Chen et al. (2026).

Their rule, as stated in the published abstract of Systematic Biology syag024:
  (i)  detect outliers in the contrasts with a distribution-free boxplot criterion;
  (ii) if influential outliers are present, use Spearman;
  (iii) otherwise use Pearson -- automatically for n >= 30, or guided by a normality
        test in smaller samples.

The preprint version of the same work (bioRxiv 2024.06.16.599156) instead selected the
correlation on a normality test of the contrasts ("Selected-Correlation-PIC"). The
published version replaces that trigger with the outlier criterion. We implement the
published rule, since that is the recommendation a reader of the journal would apply.

All contrasts here have n - 1 >= 30, so branch (iii) is always the automatic one.

We report three routes on the same contrasts: Pearson, unconditional Spearman, and
O(D)GC, plus the fraction of variable pairs on which O(D)GC fires.
"""
import os
import sys
import json
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_revision import simulate, pic_of, auroc, save, OUTDIR
from run_revision2 import f1_undirected


def topr(cc, k):
    d = cc.shape[0]
    ti = np.triu_indices(d, 1)
    v = np.abs(cc[ti])
    th = np.sort(v)[::-1][:k][-1]
    adj = (np.abs(cc) >= th)
    np.fill_diagonal(adj, False)
    return adj | adj.T


def has_outlier(col):
    """Tukey's distribution-free boxplot criterion."""
    q1, q3 = np.percentile(col, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return bool(np.any(col < lo) or np.any(col > hi))


def odgc_corr(P):
    """Chen et al. (2026) O(D)GC: boxplot gate, then Spearman or Pearson pair by pair.

    Vectorised: Pearson on ranks equals Spearman, so both matrices are formed at once and
    the gate selects entry by entry. The result is identical to a pairwise loop.
    """
    d = P.shape[1]
    cp = np.corrcoef(P.T)
    R = np.apply_along_axis(stats.rankdata, 0, P)
    cr = np.corrcoef(R.T)
    out = np.array([has_outlier(P[:, k]) for k in range(d)], dtype=bool)
    gate = out[:, None] | out[None, :]          # a pair switches if either member has one
    C = np.where(gate, cr, cp)
    np.fill_diagonal(C, 1.0)
    iu = np.triu_indices(d, 1)
    return C, float(gate[iu].mean())


def main():
    p = os.path.join(OUTDIR, 'e13_odgc.json')
    res = json.load(open(p, encoding='utf-8')) if os.path.exists(p) else {}
    for lam in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        for noise in ('laplace', 'gauss'):
            key = 'lam%s_%s' % (lam, noise)
            if key in res:
                continue
            t0 = time.time()
            acc = {'pearson': [], 'spearman': [], 'odgc': [], 'fire': [],
                   'auc_pearson': [], 'auc_spearman': [], 'auc_odgc': []}
            for seed in (0, 1, 2):
                X, W, edges, tree, H, Cv = simulate(n=200, d=30, k=10, lam=lam,
                                                    alpha_ou=0.0, noise=noise, seed=seed)
                P = pic_of(X, tree)
                k = len(edges)
                cp = np.corrcoef(P.T)
                cs = stats.spearmanr(P, axis=0).statistic
                co, fire = odgc_corr(P)
                for nm, cc in (('pearson', cp), ('spearman', cs), ('odgc', co)):
                    acc[nm].append(f1_undirected(topr(cc, k), W))
                    acc['auc_' + nm].append(auroc(cc, W))
                acc['fire'].append(fire)
            res[key] = {kk: round(float(np.nanmean(v)), 4) for kk, v in acc.items()}
            res[key].update({'lam': lam, 'noise': noise, 'sec': round(time.time() - t0, 1)})
            r = res[key]
            print('  [E13] %-18s F1  P=%.3f  S=%.3f  O(D)GC=%.3f  fire=%.2f | '
                  'AUC %.3f / %.3f / %.3f'
                  % (key, r['pearson'], r['spearman'], r['odgc'], r['fire'],
                     r['auc_pearson'], r['auc_spearman'], r['auc_odgc']))
            save('e13_odgc', res)
    return res


if __name__ == '__main__':
    main()
