# -*- coding: utf-8 -*-
"""NOTEARS AUROC on the E11 grid, so the claim "its ranking is worse to begin with" has a
number behind it that lives in a result file rather than in a console log.
"""
import os, sys, json, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))



from run_revision import simulate, pic_of, corr_scores, auroc, OUTDIR
from methods_unified import notears_score

out = {}
p = os.path.join(OUTDIR, 'e11_auroc.json')
if os.path.exists(p):
    out = json.load(open(p, encoding='utf-8'))

for lam in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
    key = f'lam{lam}'
    if key in out:
        continue
    t0 = time.time()
    nt, cc = [], []
    for seed in (0, 1, 2):
        X, W, edges, tree, H, C = simulate(n=200, d=30, k=10, lam=lam,
                                          alpha_ou=0.0, noise='laplace', seed=seed)
        P = pic_of(X, tree)
        nt.append(auroc(notears_score(P), W))
        cc.append(auroc(corr_scores(P), W))
    out[key] = {'lam': lam,
                'auroc_notears': round(float(np.mean(nt)), 3),
                'auroc_corr': round(float(np.mean(cc)), 3),
                'sec': round(time.time() - t0, 1)}
    r = out[key]
    print(f"  [E11-AUROC] lam={lam:4.1f} NOTEARS={r['auroc_notears']:.3f} "
          f"corr={r['auroc_corr']:.3f} ({r['sec']}s)")
    json.dump(out, open(p, 'w', encoding='utf-8'), indent=1)

print('saved', p)
