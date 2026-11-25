# -*- coding: utf-8 -*-
"""E32  How much of the verdict rests on admitting exactly k edges, and how wide the
seed-to-seed interval is.

Two questions a reader is entitled to ask about the evaluation rule.

1.  Every headline number admits exactly k edges when k edges are true.  That is a choice,
    and a reader who does not know k in their own data cannot reproduce it.  This stage
    re-scores the same datasets admitting half as many and half again as many, so the size
    of the dependence on k is measured rather than argued.

2.  Accuracy is reported as a mean over ten seeds, with a standard deviation.  A percentile
    interval over thirty seeds is reported here for the same cells, so the reader can see the
    spread of the estimate itself and not only its centre.  The interval is over seeds
    because the tree and the graph are fixed within a cell: resampling species would break
    the covariance the whitening depends on, so a row bootstrap is not available here.

Run: python exp_kband.py
"""
import os
import sys

import numpy as np

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(S), 'scripts'))
sys.path.insert(0, S)

from exp_new import gen, prf, save, load, log, NSPEC, KED, D30, D50
from run_revision import pic_of
from methods_unified import topk_adj

NAME = 'e32_kband'
LAMS = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0)
SEEDS = 30


def corr(P):
    M = np.asarray(P, float)
    M = M - M.mean(0)
    sd = M.std(0)
    sd[sd < 1e-12] = 1.0
    return np.corrcoef(M / sd, rowvar=False)


def main():
    res = load(NAME)
    for d in (D30, D50):
        for lam in LAMS:
            key = 'd%d_lam%.1f' % (d, lam)
            if key in res:
                log('skip', key)
                continue
            ks = sorted({max(1, int(round(KED * f))) for f in (0.5, 1.0, 1.5)})
            got = {k: [] for k in ks}
            for sd in range(SEEDS):
                X, W, tree, C = gen(NSPEC, d, KED, lam, sd, latent='tree')
                R = corr(pic_of(X, tree))
                for k in ks:
                    got[k].append(float(prf(topk_adj(R, k), W)[2]))
            res[key] = {str(k): dict(k=int(k), mean=round(float(np.mean(v)), 4),
                                     sd=round(float(np.std(v)), 4),
                                     lo=round(float(np.percentile(v, 2.5)), 4),
                                     hi=round(float(np.percentile(v, 97.5)), 4),
                                     seeds=SEEDS)
                        for k, v in got.items()}
            save(NAME, res)
            log('%-12s ' % key + '  '.join('k=%d %.2f [%.2f,%.2f]'
                                           % (k, res[key][str(k)]['mean'],
                                              res[key][str(k)]['lo'],
                                              res[key][str(k)]['hi']) for k in ks))
    log('written')


if __name__ == '__main__':
    main()
