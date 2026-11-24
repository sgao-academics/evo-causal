# -*- coding: utf-8 -*-
"""E11  Four families on one axis, one sample size, one dimension.

E5 varies the confounding load for the correlation rule; E6 and E8 did the same for PC and
FCI at different dimensions, which makes them impossible to overlay. This experiment puts
PIC+correlation, NOTEARS on PIC, PC on PIC and FCI on PIC on a single lambda grid at
d = 30, n = 200, so the claim "no algorithmic family escapes" rests on one figure rather
than on three incomparable ones.
"""
import os, sys, json, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_revision import (simulate, pic_of, corr_scores, f1_topk, auroc, blomberg_k,
                          cov_brownian, run_notears, save, OUTDIR)
from run_revision2 import f1_undirected, pc_skeleton
from causallearn.search.ConstraintBased.FCI import fci


def main():
    res = {}
    p = os.path.join(OUTDIR, 'e11_families.json')
    if os.path.exists(p):
        res = json.load(open(p, encoding='utf-8'))
    d, n = 30, 200
    for lam in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        key = f'lam{lam}'
        if key in res:
            continue
        t0 = time.time()
        acc = {k: [] for k in ('f1_corr', 'f1_notears', 'f1_pc', 'f1_fci', 'auc', 'K')}
        for seed in (0, 1, 2):
            X, W, edges, tree, H, C = simulate(n=n, d=d, k=10, lam=lam,
                                              alpha_ou=0.0, noise='laplace', seed=seed)
            P = pic_of(X, tree)
            cc = corr_scores(P)
            kk = len(edges)
            ti = np.triu_indices(d, 1)
            th = np.sort(np.abs(cc[ti]))[::-1][:kk][-1]
            a2 = (np.abs(cc) >= th); np.fill_diagonal(a2, False)
            acc['f1_corr'].append(f1_undirected(a2 | a2.T, W))
            acc['auc'].append(auroc(cc, W))

            Am = run_notears(P, lam1=0.1, w_th=0.3)
            if Am is not None:
                acc['f1_notears'].append(f1_undirected((np.abs(Am) > 0) | (np.abs(Am.T) > 0), W))

            acc['f1_pc'].append(f1_undirected(pc_skeleton(P, alpha=0.01, max_depth=3), W))

            G, el = fci(P, verbose=False, show_progress=False)
            adj = np.zeros((d, d), bool)
            for e in el:
                i = int(str(e.get_node1().get_name()).lstrip('X'))
                j = int(str(e.get_node2().get_name()).lstrip('X'))
                if i < d and j < d:
                    adj[i, j] = adj[j, i] = True
            acc['f1_fci'].append(f1_undirected(adj, W))

            V = cov_brownian(H)
            acc['K'].append(float(np.nanmean([blomberg_k(X[:, j], V, nsim=100, seed=seed * 7 + j)
                                              for j in range(5)])))

        res[key] = {'lam': lam, 'd': d, 'n': n,
                    'K_measured': round(float(np.mean(acc['K'])), 3),
                    'f1_corr': round(float(np.mean(acc['f1_corr'])), 3),
                    'f1_notears': round(float(np.mean(acc['f1_notears'])), 3) if acc['f1_notears'] else None,
                    'f1_pc': round(float(np.mean(acc['f1_pc'])), 3),
                    'f1_fci': round(float(np.mean(acc['f1_fci'])), 3),
                    'auroc': round(float(np.nanmean(acc['auc'])), 3),
                    'sec': round(time.time() - t0, 1)}
        r = res[key]
        print(f"  [E11] lam={lam:4.1f} K={r['K_measured']:.3f} corr={r['f1_corr']:.3f} "
              f"NOTEARS={r['f1_notears']} PC={r['f1_pc']:.3f} FCI={r['f1_fci']:.3f} "
              f"AUC={r['auroc']:.3f} ({r['sec']}s)")
        save('e11_families', res)
    return res


if __name__ == '__main__':
    main()
