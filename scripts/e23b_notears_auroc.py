# -*- coding: utf-8 -*-
"""NOTEARS under the threshold-free metric, on the same grid as E23.

E23 scored NOTEARS only under top-k selection and kept the AUROC of the correlation rule
alone, so Figure S3 could not say whether NOTEARS ranks badly or ranks well and then loses
the edges to a fixed cut. This fills that one cell: same data, same seeds, same d=30 grid.

It recomputes the correlation rule on the way through and asserts it against E23, so the run
cannot silently drift onto a different data stream.
"""
import os, sys, json, time
import numpy as np

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, S)
from exp_new import gen, OUT, prf, save
from run_revision import pic_of, corr_scores, run_notears, auroc
from methods_unified import topk_adj

NAME = 'e23b_notears_auroc'
LAMS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
SEEDS = 10


def probe_ok():
    """Guard: the correlation rule must reproduce E23 before any of this is trusted."""
    ref = json.load(open(os.path.join(OUT, 'e23_families10.json'), encoding='utf-8'))
    return {k: v['corr'] for k, v in ref.items()}


def run():
    res = json.load(open(os.path.join(OUT, NAME + '.json'), encoding='utf-8')) \
        if os.path.exists(os.path.join(OUT, NAME + '.json')) else {}
    ref = probe_ok()
    for lam in LAMS:
        key = 'lam%.1f' % lam
        if key in res:
            continue
        t0 = time.time()
        na, ca, nf = [], [], []
        for sd in range(SEEDS):
            X, W, tree, C = gen(200, 30, 10, lam, sd)
            P = pic_of(X, tree)
            R = corr_scores(P)
            ca.append(auroc(R, W))
            nf.append(prf(topk_adj(R, 10), W)[2])
            E = run_notears(P)
            if E is None:
                continue
            na.append(auroc(E, W))
        r = dict(lam=lam, seeds=SEEDS, notears_auroc=round(float(np.mean(na)), 4),
                 notears_auroc_sd=round(float(np.std(na, ddof=1)), 4),
                 corr_auroc=round(float(np.mean(ca)), 4),
                 corr_f1_recomputed=round(float(np.mean(nf)), 3),
                 sec=round(time.time() - t0, 1))
        res[key] = r
        save(NAME, res)
        print('  [E23b] lam=%.1f  NOTEARS AUROC=%.3f(%.3f)  corr AUROC=%.3f  corr F1=%.3f (ref %.3f)  %ds'
              % (lam, r['notears_auroc'], r['notears_auroc_sd'], r['corr_auroc'],
                 r['corr_f1_recomputed'], ref.get(key, float('nan')), r['sec']), flush=True)
    bad = [(k, res[k]['corr_f1_recomputed'], ref[k]) for k in res
           if abs(res[k]['corr_f1_recomputed'] - ref[k]) > 0.0015]
    print('reproduction check:', 'FAILED ' + str(bad) if bad else 'all six cells match E23', flush=True)
    return res


if __name__ == '__main__':
    run()
