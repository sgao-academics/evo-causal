# -*- coding: utf-8 -*-
"""E10  The evaluation-rule artefact, isolated.

Nothing in this experiment is phylogenetic: lam=0 means there is no confounding at all, so
every method is being handed the easiest possible problem. Any drop in F1 as the number of
traits grows is therefore caused by the decision rule, not by biology. The 99th-percentile
rule keeps 1% of the d(d-1)/2 candidate pairs, i.e. it keeps MORE edges as d grows while the
number of true edges stays fixed at 10.
"""
import os, sys, json, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_revision import simulate, pic_of, corr_scores, f1_topk, auroc, save, OUTDIR
from run_revision2 import f1_undirected


def main():
    res = {}
    p = os.path.join(OUTDIR, 'e10_threshold.json')
    if os.path.exists(p):
        res = json.load(open(p, encoding='utf-8'))
    for d in (30, 50, 100, 150, 200):
        key = f'd{d}'
        if key in res:
            continue
        t0 = time.time()
        rows = []
        for seed in (0, 1, 2):
            # lam = 0: no phylogenetic confounding whatsoever
            X, W, edges, tree, H, C = simulate(n=200, d=d, k=10, lam=0.0,
                                               alpha_ou=0.0, noise='laplace', seed=seed)
            cc = corr_scores(X)                 # raw data, nothing to correct
            k = len(edges)
            ti = np.triu_indices(d, 1)
            th = np.percentile(np.abs(cc[ti]), 99)
            adj = (np.abs(cc) >= th); np.fill_diagonal(adj, False)
            n_sel = int(adj.sum() // 2)
            rows.append({'d': d, 'k_true': k, 'k_p99': n_sel,
                         'f1_p99': f1_undirected(adj | adj.T, W),
                         'f1_topk': f1_topk(cc, W, k),
                         'auroc': auroc(cc, W)})
        res[key] = {kk: round(float(np.mean([r[kk] for r in rows])), 4) for kk in
                    ('k_true', 'k_p99', 'f1_p99', 'f1_topk', 'auroc')}
        res[key]['d'] = d
        res[key]['sec'] = round(time.time() - t0, 1)
        r = res[key]
        print(f"  [E10] d={d:3d} true_k={r['k_true']:.0f} p99_keeps={r['k_p99']:5.0f} "
              f"F1@p99={r['f1_p99']:.3f} F1@topk={r['f1_topk']:.3f} AUC={r['auroc']:.3f}")
        save('e10_threshold', res)
    return res


if __name__ == '__main__':
    main()
