# -*- coding: utf-8 -*-
"""The four procedures on the E11 grid, at ten seeds instead of three.

E22 showed the seed-to-seed standard deviation of NOTEARS at d=30 is about 0.15, which is
larger than the total movement the three-seed grid attributed to confounding. This re-runs the
grid so the ranking inversion is stated with an uncertainty attached.
"""
import os, sys, json, time
import numpy as np

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(S), 'scripts'))
sys.path.insert(0, S)
from exp_new import gen, OUT, prf, kstat
from run_revision import pic_of, corr_scores, run_notears, auroc
from methods_unified import pc_skeleton, fci_pattern, topk_adj

res = {}
name = 'e23_families10'
if os.path.exists(os.path.join(OUT, name + '.json')):
    res = json.load(open(os.path.join(OUT, name + '.json'), encoding='utf-8'))

for lam in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
    key = 'lam%.1f' % lam
    if key in res:
        continue
    t0 = time.time()
    rows = {m: [] for m in ('corr', 'notears', 'pc', 'fci')}
    au, K, mk = [], [], []
    for sd in range(10):
        X, W, tree, C = gen(200, 30, 10, lam, sd)
        P = pic_of(X, tree)
        R = corr_scores(P)
        rows['corr'].append(prf(topk_adj(R, 10), W)[2])
        au.append(auroc(R, W))
        E = run_notears(P)
        rows['notears'].append(prf(topk_adj(np.abs(E), 10), W)[2] if E is not None else np.nan)
        rows['pc'].append(prf(pc_skeleton(P), W)[2])
        adj, marks = fci_pattern(P)
        rows['fci'].append(prf(adj, W)[2])
        mk.append(marks)
        K.append(kstat(X, C, ntr=3, seed=sd))
    res[key] = dict(lam=lam, seeds=10, K=round(float(np.mean(K)), 3),
                    auroc=round(float(np.mean(au)), 4),
                    fci_marks=round(float(np.mean(mk)), 4),
                    sec=round(time.time() - t0, 1))
    for m, v in rows.items():
        v = [x for x in v if x == x]
        res[key][m] = round(float(np.mean(v)), 3)
        res[key][m + '_sd'] = round(float(np.std(v, ddof=1)), 3)
    json.dump(res, open(os.path.join(OUT, name + '.json'), 'w', encoding='utf-8'), indent=1)
    r = res[key]
    print('  [E23] lam=%.1f  corr=%.3f notears=%.3f(%.3f) pc=%.3f fci=%.3f  AUROC=%.3f  (%ds)'
          % (lam, r['corr'], r['notears'], r['notears_sd'], r['pc'], r['fci'], r['auroc'], r['sec']),
          flush=True)
print('saved')
