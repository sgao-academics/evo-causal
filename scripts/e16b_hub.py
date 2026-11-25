# -*- coding: utf-8 -*-
"""Concentration of the admitted set, for Corollary (ii) of Proposition 1.

hub = (degree of the most connected admitted node) / (mean degree of a k-edge graph),
so 0.5 is what an evenly spread k-edge graph gives and 1.0 is a perfect star.
"""
import os, sys, json
import numpy as np

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(S), 'scripts'))
sys.path.insert(0, S)
from exp_new import gen, OUT
from run_revision import pic_of, corr_scores
from methods_unified import topk_adj

res = {}
for lam in (0.0, 1.0, 2.0, 4.0, 6.0):
    hs = []
    for sd in range(10):
        X, W, tree, C = gen(200, 30, 10, lam, sd)
        adj = topk_adj(corr_scores(pic_of(X, tree)), 10)
        deg = adj.sum(1)
        hs.append(float(deg.max() / (2 * 10 / 30.0)))
    res['d30_lam%.1f' % lam] = dict(lam=lam, seeds=10,
                                    hub=round(float(np.mean(hs)), 3),
                                    sd=round(float(np.std(hs)), 3))
    print('  lam=%.1f hub=%.2f' % (lam, res['d30_lam%.1f' % lam]['hub']), flush=True)

p = os.path.join(OUT, 'e16b_hub.json')
json.dump(res, open(p, 'w', encoding='utf-8'), indent=1)
print('saved', p)
