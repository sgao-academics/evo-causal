# -*- coding: utf-8 -*-
"""E12  Does the nonparametric PIC route rescue the confounded case?

Chen et al. (2026, Systematic Biology) report that Spearman correlation on PIC keeps its
accuracy under evolutionary shifts and suppresses false positives, whereas the parametric
route does not. Their regime is outliers and non-normal marginals. Our regime is a shared
latent factor, which is a different failure mode. Testing it costs nothing and settles
whether "use a rank correlation" is a sufficient recommendation.
"""
import os, sys, json, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_revision import simulate, pic_of, f1_topk, auroc, save, OUTDIR
from run_revision2 import f1_undirected


def rank_corr(P):
    return stats.spearmanr(P, axis=0).statistic


def topr(cc, k):
    d = cc.shape[0]
    ti = np.triu_indices(d, 1)
    v = np.abs(cc[ti])
    th = np.sort(v)[::-1][:k][-1]
    adj = (np.abs(cc) >= th); np.fill_diagonal(adj, False)
    return adj | adj.T


def main():
    res = {}
    p = os.path.join(OUTDIR, 'e12_spearman.json')
    if os.path.exists(p):
        res = json.load(open(p, encoding='utf-8'))
    for lam in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        for noise in ('laplace', 'gauss'):
            key = f'lam{lam}_{noise}'
            if key in res:
                continue
            t0 = time.time()
            acc = {'pearson': [], 'spearman': [], 'auc_pearson': [], 'auc_spearman': []}
            for seed in (0, 1, 2):
                X, W, edges, tree, H, C = simulate(n=200, d=30, k=10, lam=lam,
                                                   alpha_ou=0.0, noise=noise, seed=seed)
                P = pic_of(X, tree)
                k = len(edges)
                cp = np.corrcoef(P.T)
                cs = rank_corr(P)
                acc['pearson'].append(f1_undirected(topr(cp, k), W))
                acc['spearman'].append(f1_undirected(topr(cs, k), W))
                acc['auc_pearson'].append(auroc(cp, W))
                acc['auc_spearman'].append(auroc(cs, W))
            res[key] = {kk: round(float(np.nanmean(v)), 4) for kk, v in acc.items()}
            res[key].update({'lam': lam, 'noise': noise, 'sec': round(time.time() - t0, 1)})
            r = res[key]
            print(f"  [E12] {key:20s} F1 pearson={r['pearson']:.3f} spearman={r['spearman']:.3f} | "
                  f"AUC {r['auc_pearson']:.3f} / {r['auc_spearman']:.3f}")
            save('e12_spearman', res)
    return res


if __name__ == '__main__':
    main()
