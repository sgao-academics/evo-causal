# -*- coding: utf-8 -*-
"""Archive two numbers that the text quotes.

e21_lrsem   the trial in which the residual covariance is modelled as rank-one plus isotropic,
            which is the obvious way to "just model the latent" and does not work.
e15c_r_vs_n the screen statistic at fixed confounding as the sample size changes, which is why
            the null has to be recalibrated for each dataset.
"""
import os, sys, json
import numpy as np

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(S), 'scripts'))
sys.path.insert(0, S)
from exp_new import gen, OUT, prf, load, save
from run_revision import pic_of, corr_scores
from methods_unified import topk_adj, f1_undirected
from lrsem_trial import search


def lrsem():
    res = {}
    for lam in (0.0, 2.0):
        f, p = [], []
        for sd in (0, 1, 2):
            X, W, tree, C = gen(200, 30, 10, lam, sd)
            P = pic_of(X, tree)
            p.append(f1_undirected(topk_adj(corr_scores(P), 10), W))
            Wh, _ = search(P)
            f.append(f1_undirected((Wh != 0) | (Wh.T != 0), W))
        res['lam%.1f' % lam] = dict(lam=lam, seeds=3,
                                    lrsem=round(float(np.mean(f)), 3),
                                    plain=round(float(np.mean(p)), 3))
        print('  LR-SEM lam=%.1f  search=%.3f  plain=%.3f' % (lam, np.mean(f), np.mean(p)),
              flush=True)
    save('e21_lrsem', res)


def r_vs_n():
    res = {}
    for lam in (0.0, 2.0):
        for n in (100, 200, 500, 1000, 2000):
            rs = []
            for sd in range(5):
                X, W, tree, C = gen(n, 30, 10, lam, sd)
                R = corr_scores(pic_of(X, tree))
                w = np.linalg.eigvalsh(R)
                rs.append(float(w[-1] / w[:-1].mean()))
            res['lam%.1f_n%d' % (lam, n)] = dict(lam=lam, n=n, seeds=5,
                                                 r=round(float(np.mean(rs)), 3))
            print('  r lam=%.1f n=%-5d r=%.2f' % (lam, n, np.mean(rs)), flush=True)
    save('e15c_r_vs_n', res)


if __name__ == '__main__':
    lrsem()
    r_vs_n()
