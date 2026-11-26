# -*- coding: utf-8 -*-
"""Seed-level spread for E10, the experiment behind Fig. S1.

_e10_threshold.py stores means only, so Fig. S1 has never been able to show an error bar. This
writes the per-seed mean and standard deviation beside it, under its own name, and leaves
e10_threshold.json untouched.

Two facts come out of it, and both are used by the figure.

  * The 99th-percentile rule's admitted count is 0.01*d(d-1)/2 by construction, and at lambda = 0
    the correlation ranking is essentially perfect, so the kept set and the true set are nested at
    every d. Its F1 is therefore 2*min(k,k99)/(k+k99) -- arithmetic, with a seed standard deviation
    of exactly zero. A zero-length error bar on that curve would imply a measurement that does not
    exist, so the figure draws none.
  * top-k does vary, by 0.058 at the two smallest d, and only there.

The recomputation also serves as a check: the means printed here must equal the ones already in
e10_threshold.json, otherwise the script and the file disagree and the figure is drawing something
else.
"""
import os, sys, json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_revision import simulate, corr_scores, f1_topk, auroc, save, OUTDIR
from run_revision2 import f1_undirected

FIELDS = ('k_true', 'k_p99', 'f1_p99', 'f1_topk', 'auroc')


def main():
    stored = {}
    p0 = os.path.join(OUTDIR, 'e10_threshold.json')
    if os.path.exists(p0):
        stored = json.load(open(p0, encoding='utf-8'))

    out = {}
    for d in (30, 50, 100, 150, 200):
        rows = []
        for seed in (0, 1, 2):
            X, W, edges, tree, H, C = simulate(n=200, d=d, k=10, lam=0.0,
                                               alpha_ou=0.0, noise='laplace', seed=seed)
            cc = corr_scores(X)
            k = len(edges)
            ti = np.triu_indices(d, 1)
            th = np.percentile(np.abs(cc[ti]), 99)
            adj = (np.abs(cc) >= th)
            np.fill_diagonal(adj, False)
            rows.append({'k_true': float(k), 'k_p99': float(adj.sum() // 2),
                         'f1_p99': float(f1_undirected(adj | adj.T, W)),
                         'f1_topk': float(f1_topk(cc, W, k)),
                         'auroc': float(auroc(cc, W))})
        key = 'd%d' % d
        rec = {'d': d, 'seeds': len(rows)}
        for f in FIELDS:
            v = np.array([r[f] for r in rows])
            rec[f] = round(float(v.mean()), 4)
            rec[f + '_sd'] = round(float(v.std(ddof=1)), 4)
            if key in stored and abs(rec[f] - stored[key][f]) > 5e-4:
                print('  MISMATCH %s %s: %.4f vs stored %.4f'
                      % (key, f, rec[f], stored[key][f]))
        out[key] = rec
        print('  d=%3d  k99=%.0f  F1@p99=%.4f+-%.4f  F1@topk=%.4f+-%.4f  AUROC=%.4f'
              % (d, rec['k_p99'], rec['f1_p99'], rec['f1_p99_sd'],
                 rec['f1_topk'], rec['f1_topk_sd'], rec['auroc']))

    save('e10_threshold_seeds', out)
    return out


if __name__ == '__main__':
    main()
