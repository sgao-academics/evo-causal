# -*- coding: utf-8 -*-
"""E31b  Null thresholds and stability for the named case.

The r1 values in e31 mean nothing without the null they are compared to: for n iid d-vectors the
contrast correlation matrix is distributed as a sample correlation, so the threshold is simulated
exactly and no tree enters it.

Also re-runs the four variants under three Grafen rho values and equal lengths, and over 80%
subsamples without replacement, so the reported ordering is not an artefact of one convention or
of which species the tree happened to keep.
"""
import os, io, json, sys, time
import numpy as np

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, S)
sys.path.insert(0, os.path.join(os.path.dirname(S), 'tool'))
import fetch_real as FR
from pic_screen import tree as PT
import _e31_named_case as E

COLS = E.COLS
VARIANTS = E.VARIANTS


def corr_of(M):
    M = np.asarray(M, float)
    M = M - M.mean(0)
    sd = M.std(0)
    sd[sd < 1e-12] = 1.0
    return np.corrcoef(M / sd, rowvar=False)


def r1_of(R):
    w = np.linalg.eigvalsh(R)[::-1]
    return float(w[0] / w[1:].mean())


def null_q(n, d, reps=4000, seed=11):
    rng = np.random.default_rng(seed)
    out = np.empty(reps)
    for b in range(reps):
        out[b] = r1_of(corr_of(rng.normal(size=(n, d))))
    return dict(n=int(n), d=int(d), reps=int(reps),
                mean=round(float(out.mean()), 3), sd=round(float(out.std()), 3),
                q95=round(float(np.quantile(out, 0.95)), 3),
                q99=round(float(np.quantile(out, 0.99)), 3),
                max=round(float(out.max()), 3),
                sec=round(float(out.std() / np.sqrt(reps)), 4))


def main():
    t0 = time.time()
    species, X, match, logged = E.load()
    idx, C = E.principle_tree(species, match)     # collapsed, correct Grafen, checked
    X = X[idx]
    frames = [species[i].replace(' ', '_') for i in idx]
    n = len(frames)
    print('n =', n)

    nulls = {}
    for d in (6, 5):
        nulls['d%d' % d] = null_q(n, d, reps=4000)
        q = nulls['d%d' % d]
        print('null d=%d  mean=%.2f  q95=%.2f  q99=%.2f  max=%.2f'
              % (d, q['mean'], q['q95'], q['q99'], q['max']))

    with io.open(FR.NEWICK_CLADE, encoding='utf-8') as f:
        tmpl = f.read()

    sens = {}
    for conv, rho in (('grafen', 0.5), ('grafen', 1.0), ('grafen', 2.0), ('equal', 1.0)):
        r = PT.prune(PT.parse_newick(tmpl), set(frames))
        if conv == 'equal':
            PT.equal_lengths(r)
        else:
            PT.grafen_lengths(r, rho=rho)
        lo, hi = PT.check_ultrametric(r)
        Cv = PT.brownian_cov(r, frames)
        P = E.whiten(X, Cv)
        row = {}
        for name, drop in VARIANTS:
            keep = [j for j, (c, _) in enumerate(COLS) if c not in drop]
            row[name] = round(r1_of(corr_of(P[:, keep])), 2)
        key = '%s_%s' % (conv, rho)
        sens[key] = row
        print('%-14s depth %.3f/%.3f  %s' % (key, lo, hi, row))

    # stability: 80% subsamples without replacement, so no species is counted twice.  Resampling
    # with replacement would repeat rows and inflate r1 by construction, which is not a
    # confidence interval for anything.
    rng = np.random.default_rng(7)
    sub = {name: [] for name, _ in VARIANTS}
    B = 300
    m = int(round(0.8 * n))
    for b in range(B):
        ii = np.sort(rng.choice(n, m, replace=False))
        try:
            L = np.linalg.cholesky(C[np.ix_(ii, ii)] + 1e-10 * np.eye(m))
        except np.linalg.LinAlgError:
            continue
        Pb = np.linalg.solve(L, X[ii])
        for name, drop in VARIANTS:
            keep = [j for j, (c, _) in enumerate(COLS) if c not in drop]
            sub[name].append(r1_of(corr_of(Pb[:, keep])))
    subsum = {k: dict(median=round(float(np.median(v)), 2),
                      lo=round(float(np.quantile(v, 0.05)), 2),
                      hi=round(float(np.quantile(v, 0.95)), 2), reps=len(v))
              for k, v in sub.items()}
    print('\n80%% subsamples, no replacement (n=%d of %d, %d reps)' % (m, n, len(sub['full'])))
    for k, v in subsum.items():
        print('  %-14s %s' % (k, v))

    out = dict(n=n, nulls=nulls, sensitivity=sens, subsample=subsum)
    dst = os.path.join(os.path.dirname(S), 'results', 'e31b_null.json')
    with io.open(dst, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('\nsaved', dst, ' %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
