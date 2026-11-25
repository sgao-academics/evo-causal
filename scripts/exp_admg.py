# -*- coding: utf-8 -*-
"""E25  ADMG family: does a search that is allowed to posit a latent common cause keep
its accuracy where the DAG-only methods lose theirs?

The competing methods in the paper all assume independent noise, which is the assumption
the confounder violates.  The ADMG family drops it: a bidirected edge i <-> j is a latent
common cause that has been integrated out, and the likelihood is maximised over both the
directed and the bidirected part (Richardson 2003; the ML problem is the one RICF solves,
Drton, Fox & Wang 2020).  If the paper's mechanism account is right, this family should be
the one that survives, at the price of a larger search.

Grid: d = 10 and 15, k set to keep the density of the main grid at 1%, six levels of
lambda, three seeds, every method scored by F1 against the true skeleton under top-k with
k equal to the true edge count.  Also records how many bidirected edges the search returns,
because a search that "survives" by declaring everything confounded has not survived.

Run: python exp_admg.py [--only d10]
"""
import os, sys, json, time, argparse
import numpy as np

S = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(os.path.dirname(S), 'scripts')
sys.path.insert(0, R)
sys.path.insert(0, S)

from run_revision import make_dag, blomberg_k, pic_of, auroc, corr_scores, run_notears
from methods_unified import topk_adj, symmetrise
from exp_new import gen, prf, kstat, load, save, path, log
import admg

OUT = os.path.join(os.path.dirname(S), 'results')

DIMS = {'d10': (10, 4), 'd15': (15, 6)}
LAMS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
SEEDS = 3


def e25(which=('d10',)):
    name = 'e25_admg'
    res = load(name)
    for tag in which:
        d, k = DIMS[tag]
        for lam in LAMS:
            key = '%s_lam%.1f' % (tag, lam)
            if key in res:
                log('skip', key)
                continue
            t0 = time.time()
            f1 = dict(corr=[], notears=[], admg=[], pc=[])
            nb = []; kvals = []
            for sd in range(SEEDS):
                X, W, tree, C = gen(200, d, k, lam, sd)
                T = np.abs(W) > 0
                T = T | T.T
                Rt = corr_scores(X)
                f1['corr'].append(prf(topk_adj(Rt, k), W)[2])
                Wn = run_notears(X)
                f1['notears'].append(prf(topk_adj(Wn, k), W)[2] if Wn is not None else float('nan'))
                r = admg.search_admg(X, max_edges=3 * k)
                bm = r['bmask']
                est = symmetrise(bm | bm.T)
                f1['admg'].append(prf(est, W)[2])
                nb.append(int(np.triu(r['dmask'], 1).sum()))
                kvals.append(kstat(X, C, ntr=2, seed=sd))
            res[key] = dict(lam=lam, d=d, k=k, seeds=SEEDS,
                            K=round(float(np.mean(kvals)), 3),
                            f1_corr=round(float(np.mean(f1['corr'])), 3),
                            f1_notears=round(float(np.mean(f1['notears'])), 3),
                            f1_admg=round(float(np.mean(f1['admg'])), 3),
                            admg_bidir=float(np.mean(nb)),
                            admg_bidir_all=nb,
                            f1_corr_seeds=[round(x, 3) for x in f1['corr']],
                            f1_notears_seeds=[round(x, 3) for x in f1['notears']],
                            f1_admg_seeds=[round(x, 3) for x in f1['admg']])
            save(name, res)
            log('%-14s K=%.3f corr=%.3f notears=%.3f admg=%.3f bidir=%.1f  %.0fs' %
                (key, res[key]['K'], res[key]['f1_corr'], res[key]['f1_notears'],
                 res[key]['f1_admg'], res[key]['admg_bidir'], time.time() - t0))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='d10')
    a = ap.parse_args()
    e25(tuple(a.only.split(',')))
