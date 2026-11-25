# -*- coding: utf-8 -*-
"""E26  Real-data anchor: where do real comparative matrices sit?

The paper's rule is stated on a simulated axis, and the objection that matters is whether
real data ever land in the region the rule calls dangerous.  This block answers it with
measured data instead of with a reported number from the literature:

  traits   PanTHERIA (Jones et al. 2009), 4,629 mammal species, species-level measurements
  tree     Open Tree of Life synthesis (Hinchliff et al. 2015), pruned to the species that
           have a complete trait vector, branch lengths by Grafen (1989)

Three blocks of increasing width, because the answer may depend on how many traits the
matrix has.  For each block it reports

  K          Blomberg's K per trait  -> which band of the paper's rule the matrix sits in
  r          leading eigenvalue over the mean of the rest, of the trait correlation matrix
  r_null     the same statistic under independent Brownian motion on the same tree, which
             is what phylogenetic signal alone produces; the z score is the excess
  tetrad     share of vanishing-tetrad tests that reject, against the same null

The null is the point.  If a real matrix has r above what independent Brownian motion
gives on its own tree, then part of its covariation is not shared history but a common
factor, which is the object the paper is about.

Run: python exp_real.py
"""
import os, sys, json, time
import numpy as np

S = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(os.path.dirname(S), 'scripts')
sys.path.insert(0, R)
sys.path.insert(0, S)

import phylo
import admg
from run_revision import blomberg_k
from fetch_real import BLOCKS, PANTH, NEWICK_CLADE, read_pantheria
from exp_new import load, save, log

DATA = os.path.join(os.path.dirname(S), 'data')
NNULL = 300
NSIM_K = 200


def lead_ratio(Rm):
    Rm = np.nan_to_num(np.asarray(Rm, float), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(Rm, 1.0)
    Rm = 0.5 * (Rm + Rm.T)
    w = np.linalg.eigvalsh(Rm)
    return float(w[-1] / max(w[:-1].mean(), 1e-12)), float(w[-1]), float(w[-2])


def prep_block(name, rows, match, tree_root, names_avail):
    """Complete-case species for this block, transformed and standardised."""
    cols = [(c, lab) for c, lab in BLOCKS[name]]
    # drop duplicated column names that only exist as guards
    seen = set(); keep = []
    for c, lab in cols:
        if c in seen:
            continue
        seen.add(c); keep.append((c, lab))
    cols = keep
    ok = []
    for sp, v in rows:
        if not match.get(sp):
            continue
        if all(c in v for c, _ in cols):
            ok.append((sp, v))
    if not ok:
        return None
    species = [sp for sp, _ in ok]
    X = np.array([[v[c] for c, _ in cols] for _, v in ok], float)
    # log-transform a column when it is strictly positive and spans more than 20-fold;
    # counts, proportions, codes and signed coordinates are left alone
    logs = []
    for j in range(X.shape[1]):
        col = X[:, j]
        if col.min() > 0 and col.max() / max(col.min(), 1e-12) > 20:
            X[:, j] = np.log10(col)
            logs.append(cols[j][0])
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    # a column that is constant in the complete-case subset carries no information and
    # makes the correlation matrix singular, so it is dropped rather than imputed
    alive = X.std(0) > 1e-8
    dropped = [c for c, a in zip([c for c, _ in cols], alive) if not a]
    X = X[:, alive]
    cols = [c for c, a in zip(cols, alive) if a]
    return dict(species=species, X=X, cols=[c for c, _ in cols],
                labels=[l for _, l in cols], logged=logs, dropped=dropped)


def otol_label(sp, ott):
    """The synthesis subtree labels a tip with the bare binomial, underscores for spaces.

    The `_ott<id>` suffix appears only when `label_format='id'` is requested, so matching
    on the suffixed form silently prunes every tip and leaves an empty table -- which is
    what the first run of this block did.
    """
    return sp.replace(' ', '_')


def block_stats(blk, root_template, use_grafen=True):
    species = blk['species']
    X = blk['X']
    label_of = {otol_label(s, blk['match'][s]): s for s in species}
    root = phylo.parse_newick(root_template)
    root, kept = phylo.prune(root, set(label_of))
    kept_set = set(kept)
    keep_idx = [i for i, s in enumerate(species) if otol_label(s, blk['match'][s]) in kept_set]
    species = [species[i] for i in keep_idx]
    labels = [otol_label(s, blk['match'][s]) for s in species]
    X = X[keep_idx]
    if use_grafen:
        phylo.grafen_lengths(root)
    else:
        phylo.equal_lengths(root)
    C = phylo.brownian_cov(root, labels)
    n, d = X.shape
    if n < 5 or d < 2:
        print('  block %s has n=%d d=%d, too small to score' % (blk.get('block'), n, d))
        return None
    if not np.isfinite(X).all():
        print('  block %s has non-finite entries, dropped %d rows' %
              (blk.get('block'), int((~np.isfinite(X).all(axis=1)).sum())))
        keep = np.isfinite(X).all(axis=1)
        X = X[keep]
        species = [s for s, k_ in zip(species, keep) if k_]
        labels = [l for l, k_ in zip(labels, keep) if k_]
        root2 = phylo.prune(phylo.parse_newick(root_template), set(labels))[0]
        phylo.grafen_lengths(root2) if use_grafen else phylo.equal_lengths(root2)
        C = phylo.brownian_cov(root2, labels)
        n, d = X.shape

    K = [float(blomberg_k(X[:, j], C, nsim=NSIM_K, seed=1000 + j)) for j in range(d)]
    Rm = np.corrcoef(X, rowvar=False)
    r_obs, l1, l2 = lead_ratio(Rm)

    # independent Brownian motion on the same tree: shared history and nothing else
    Lc = np.linalg.cholesky(C + 1e-10 * np.eye(n))
    rng = np.random.default_rng(7)
    r_null = np.empty(NNULL)
    for b in range(NNULL):
        Xn = Lc @ rng.normal(size=(n, d))
        Xn = (Xn - Xn.mean(0)) / (Xn.std(0) + 1e-12)
        r_null[b] = lead_ratio(np.corrcoef(Xn, rowvar=False))[0]
    z = (r_obs - r_null.mean()) / max(r_null.std(), 1e-12)

    # tetrads on the real matrix and on one null replicate
    tt = admg.tetrad_test(X, B=300, seed=3)
    Xn = Lc @ rng.normal(size=(n, d))
    Xn = (Xn - Xn.mean(0)) / (Xn.std(0) + 1e-12)
    tt_null = admg.tetrad_test(Xn, B=300, seed=3)

    out = dict(block=blk.get('block'), n=int(n), d=int(d),
               K=[round(k, 3) for k in K], K_mean=round(float(np.mean(K)), 3),
               K_median=round(float(np.median(K)), 3),
               K_min=round(float(np.min(K)), 3), K_max=round(float(np.max(K)), 3),
               K_over_0p30=int(sum(1 for k in K if k > 0.30)),
               r=round(r_obs, 3), lambda1=round(l1, 2),
               r_null_mean=round(float(r_null.mean()), 3),
               r_null_sd=round(float(r_null.std()), 3),
               r_null_q95=round(float(np.quantile(r_null, 0.95)), 3),
               r_z=round(float(z), 2),
               tetrad_reject=round(tt['reject_share'], 3),
               tetrad_reject_null=round(tt_null['reject_share'], 3),
               tetrad_n=tt['n_tetrads'],
               grafen=use_grafen,
               cols=blk['cols'], labels=blk['labels'], logged=blk['logged'])
    return out


def main():
    res = load('e26_real')
    root_template = __import__('io').open(NEWICK_CLADE, encoding='utf-8').read()
    rows = read_pantheria()
    log('species with any trait', len(rows))
    match = json.load(__import__('io').open(
        os.path.join(DATA, 'otol_match.json'), encoding='utf-8'))
    log('matched', sum(1 for v in match.values() if v))

    for name in BLOCKS:
        key = name
        if key in res:
            log('skip', key)
            continue
        t0 = time.time()
        blk = prep_block(name, rows, match, root_template, None)
        if blk is None:
            log('no complete-case species for', name)
            continue
        blk['block'] = name
        blk['match'] = match
        o = block_stats(blk, root_template, use_grafen=True)
        o_el = block_stats(blk, root_template, use_grafen=False)
        if o is None:
            log('block', name, 'too small after pruning')
            continue
        if o_el is None:
            o_el = dict(r=float('nan'), r_z=float('nan'), K_mean=float('nan'),
                        tetrad_reject=float('nan'))
        o['sensitivity_equal_lengths'] = dict(
            r=o_el['r'], r_z=o_el['r_z'], K_mean=o_el['K_mean'],
            tetrad_reject=o_el['tetrad_reject'])
        res[key] = o
        save('e26_real', res)
        log('%-18s n=%4d d=%2d  K=%.3f [%.2f,%.2f]  r=%.2f (null %.2f±%.2f, z=%.1f)'
            '  tetrad %.2f vs null %.2f  %.0fs' %
            (name, o['n'], o['d'], o['K_mean'], o['K_min'], o['K_max'], o['r'],
             o['r_null_mean'], o['r_null_sd'], o['r_z'],
             o['tetrad_reject'], o['tetrad_reject_null'], time.time() - t0))


if __name__ == '__main__':
    main()
