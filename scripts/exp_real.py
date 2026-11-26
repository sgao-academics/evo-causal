# -*- coding: utf-8 -*-
"""E26  Real-data anchor: where do real comparative matrices sit?   (protocol v2)

The paper's rule is stated on simulated data. The objection that matters is whether real
comparative matrices ever land in the region the rule calls dangerous.

  traits   PanTHERIA (Jones et al. 2009), 4,629 mammal species
  tree     Open Tree of Life synthesis (Hinchliff et al. 2015), pruned to the species that
           have a complete trait vector, branch lengths by Grafen (1989)

Three changes from v1, each of which removes a way the reported numbers could be wrong.

1.  The statistic is computed on the CONTRAST correlation matrix, which is the matrix the
    paper's screen is defined on. v1 computed it on the raw trait correlation, so Table 6
    was reporting a different quantity from the equations it illustrates.

2.  The null is exact rather than simulated from a tree. If each trait is independent
    Brownian motion, the whitened data are independent standard normals whatever the tree,
    so the contrast correlation matrix is distributed exactly as the sample correlation of
    n independent d-vectors. No tree enters the null, and no branch-length convention can
    bias it.

3.  Significance is an empirical p-value, not a normal approximation to a ratio of
    eigenvalues, and the tetrad test takes its standard error from the same exact null
    instead of from an iid row bootstrap (which understates the error on data whose rows
    are phylogenetically dependent and therefore over-rejects).

The report also names the variable the ranking collapses onto, because that is the
prediction a comparative biologist can check against their own data.

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
NNULL = 4000          # null draws for the eigenvalue ratios
NSIM_K = 200          # Brownian sims inside Blomberg's K
NSIM_TET = 600        # null draws for the tetrad standard errors
KMAX = 3


# ---------------------------------------------------------------- basic statistics
def corr_of(M):
    M = np.asarray(M, float)
    M = M - M.mean(0)
    sd = M.std(0)
    sd[sd < 1e-12] = 1.0
    return np.corrcoef(M / sd, rowvar=False)


def spectrum(R, kmax=KMAX):
    """lambda_k over the mean of the eigenvalues below it, for k = 1..kmax."""
    w = np.linalg.eigvalsh(R)[::-1]
    out = []
    for k in range(kmax):
        rest = w[k + 1:]
        out.append(float(w[k] / max(rest.mean(), 1e-12)))
    return out, w


def null_spectrum(n, d, reps=NNULL, seed=11, kmax=KMAX):
    rng = np.random.default_rng(seed)
    out = np.empty((reps, kmax))
    for b in range(reps):
        out[b] = spectrum(corr_of(rng.normal(size=(n, d))), kmax)[0]
    return out


def emp_p(null, obs):
    null = np.asarray(null, float)
    return float((1.0 + np.sum(null >= obs)) / (len(null) + 1.0))


def tetrad_vals(R):
    p = R.shape[0]
    out = []
    for i in range(p):
        for j in range(i + 1, p):
            for k in range(j + 1, p):
                for l in range(k + 1, p):
                    out.append(R[i, j] * R[k, l] - R[i, k] * R[j, l])
    return np.asarray(out, float)


def tetrad_par(R, n, reps=NSIM_TET, seed=5):
    """Vanishing-tetrad test whose standard errors come from the exact null.

    The null is the sample correlation of n independent d-vectors, which is what independent
    Brownian traits become after whitening. Half the replicates give the standard errors and
    the other half are scored as if they were data, which measures the size of the test on
    the very hypothesis it is testing.
    """
    if R.shape[0] < 4:
        return dict(n_tetrads=0, reject=float('nan'), size=float('nan'),
                    median_abs_z=float('nan'))
    obs = tetrad_vals(R)
    p = R.shape[0]
    rng = np.random.default_rng(seed)
    T = np.empty((reps, len(obs)))
    for b in range(reps):
        T[b] = tetrad_vals(corr_of(rng.normal(size=(n, p))))
    h = reps // 2
    sd = T[:h].std(0)
    z = np.abs(obs) / np.maximum(sd, 1e-15)
    z0 = np.abs(T[h:]) / np.maximum(sd, 1e-15)
    return dict(n_tetrads=int(len(obs)),
                reject=float(np.mean(z > 1.96)),
                size=float(np.mean(z0 > 1.96)),
                median_abs_z=float(np.median(z)))


def hub(R, k):
    """Which variable does a top-k ranking collapse onto?"""
    d = R.shape[0]
    A = np.abs(R - np.diag(np.diag(R)))
    iu = np.triu_indices(d, 1)
    v = A[iu]
    keep = np.argsort(v)[::-1][:k]
    deg = np.zeros(d)
    for t in keep:
        deg[iu[0][t]] += 1
        deg[iu[1][t]] += 1
    order = np.argsort(deg)[::-1]
    return dict(k=int(k), deg=[int(x) for x in deg],
                top=int(order[0]), top_deg=int(deg[order[0]]),
                share=float(deg[order[0]] / max(2.0 * k, 1.0)),
                second=int(order[1]), second_deg=int(deg[order[1]]))


def whiten(X, C):
    n = C.shape[0]
    Lc = np.linalg.cholesky(C + 1e-10 * np.eye(n))
    return np.linalg.solve(Lc, X)


# ---------------------------------------------------------------- tree conventions
def power_lengths(root, rho):
    """Grafen's power convention.  One implementation, in phylo.py.

    The local version that used to sit here differenced the depths the wrong way round
    (parent minus child, which is negative on every edge) and then clamped to 1e-6, so every
    branch length collapsed to a constant and the rho variants were not measuring rho at all.
    """
    return phylo.power_lengths(root, rho)


def cov_lambda(C, lam):
    """Pagel's lambda: shrink the off-diagonal towards zero, keep the diagonal."""
    D = np.diag(np.diag(C))
    return lam * C + (1.0 - lam) * D


# ---------------------------------------------------------------- block preparation
def prep_block(name, rows, match, tree_root, names_avail):
    cols = [(c, lab) for c, lab in BLOCKS[name]]
    seen = set()
    keep = []
    for c, lab in cols:
        if c in seen:
            continue
        seen.add(c)
        keep.append((c, lab))
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
    logs = []
    for j in range(X.shape[1]):
        col = X[:, j]
        if col.min() > 0 and col.max() / max(col.min(), 1e-12) > 20:
            X[:, j] = np.log10(col)
            logs.append(cols[j][0])
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    alive = X.std(0) > 1e-8
    dropped = [c for c, a in zip([c for c, _ in cols], alive) if not a]
    X = X[:, alive]
    cols = [c for c, a in zip(cols, alive) if a]
    return dict(species=species, X=X, cols=[c for c, _ in cols],
                labels=[l for _, l in cols], logged=logs, dropped=dropped)


def otol_label(sp, ott):
    """The synthesis subtree labels a tip with the bare binomial, underscores for spaces."""
    return sp.replace(' ', '_')


def trimmed(blk, root_template, convention='grafen', rho=1.0):
    """Prune the tree to this block's species and return (X, labels, C)."""
    species = blk['species']
    X = blk['X']
    label_of = {otol_label(s, blk['match'][s]): s for s in species}
    root = phylo.parse_newick(root_template)
    root, kept = phylo.prune(root, set(label_of))
    kept_set = set(kept)
    idx = [i for i, s in enumerate(species) if otol_label(s, blk['match'][s]) in kept_set]
    species = [species[i] for i in idx]
    labels = [otol_label(s, blk['match'][s]) for s in species]
    X = X[idx]
    good = np.isfinite(X).all(axis=1)
    if not good.all():
        X = X[good]
        labels = [l for l, g in zip(labels, good) if g]
        species = [s for s, g in zip(species, good) if g]
        root = phylo.parse_newick(root_template)
        root, _ = phylo.prune(root, set(labels))
    # pruning a synthesis tree to a subset leaves degree-one nodes behind; they carry no
    # branching information and the node-count convention would charge for them
    root = phylo.collapse_unary(root)
    if convention == 'grafen':
        if rho == 1.0:
            phylo.grafen_lengths(root)
        else:
            power_lengths(root, rho)
    elif convention == 'equal':
        phylo.equal_lengths(root)
    C = phylo.brownian_cov(root, labels)
    return X, labels, C, species


def block_stats(blk, root_template, verbose=True):
    X, labels, C, species = trimmed(blk, root_template)
    n, d = X.shape
    if n < 6 or d < 4:
        log('  block %s too small: n=%d d=%d' % (blk.get('block'), n, d))
        return None

    K = [float(blomberg_k(X[:, j], C, nsim=NSIM_K, seed=1000 + j)) for j in range(d)]

    P = whiten(X, C)
    Rc = corr_of(P)
    Rr = corr_of(X)

    spec_c, w_c = spectrum(Rc)
    spec_r, _ = spectrum(Rr)
    null = null_spectrum(n, d)
    pvals = [emp_p(null[:, j], spec_c[j]) for j in range(KMAX)]

    tt = tetrad_par(Rc, n)

    # the ranking, and what it collapses onto, before and after deflation
    hb = hub(Rc, k=d)
    w, v = np.linalg.eigh(Rc)
    i1 = int(np.argmax(w))
    Rd = Rc - w[i1] * np.outer(v[:, i1], v[:, i1])
    hd = hub(Rd, k=d)
    spec_d, _ = spectrum(Rd, 1)

    out = dict(block=blk.get('block'), n=int(n), d=int(d), species=len(species),
               K=[round(k, 3) for k in K], K_mean=round(float(np.mean(K)), 3),
               K_median=round(float(np.median(K)), 3),
               K_min=round(float(np.min(K)), 3), K_max=round(float(np.max(K)), 3),
               K_over_0p30=int(sum(1 for k in K if k > 0.30)),
               cols=blk['cols'], labels=blk['labels'], logged=blk['logged'],
               dropped=blk['dropped'],
               contrast=dict(r=[round(x, 3) for x in spec_c],
                             p=[round(x, 5) for x in pvals],
                             lam=[round(float(x), 3) for x in w_c[:KMAX]],
                             null_mean=[round(float(x), 3) for x in null.mean(0)],
                             null_q95=[round(float(x), 3) for x in np.quantile(null, 0.95, axis=0)],
                             null_max=[round(float(x), 3) for x in null.max(0)]),
               raw=dict(r=round(spec_r[0], 3)),
               tetrad=tt,
               hub=dict(top=hb['top'], top_label=blk['labels'][hb['top']],
                        top_deg=hb['top_deg'], share=round(hb['share'], 3),
                        second_label=blk['labels'][hb['second']],
                        second_deg=hb['second_deg'], deg=hb['deg']),
               hub_deflated=dict(top=hd['top'], top_label=blk['labels'][hd['top']],
                                 top_deg=hd['top_deg'], share=round(hd['share'], 3),
                                 deg=hd['deg']),
               r_after_deflation=round(spec_d[0], 3))
    return out


def variants(blk, root_template):
    """Is the answer an artefact of the branch-length convention or the assumed lambda?"""
    out = {}
    for tag, conv, rho in (('grafen_1.0', 'grafen', 1.0),
                           ('grafen_0.5', 'grafen', 0.5),
                           ('grafen_2.0', 'grafen', 2.0),
                           ('equal', 'equal', 1.0)):
        X, labels, C, _ = trimmed(blk, root_template, convention=conv, rho=rho)
        n, d = X.shape
        P = whiten(X, C)
        Rc = corr_of(P)
        s, _ = spectrum(Rc, 1)
        null = null_spectrum(n, d, reps=1500, seed=21, kmax=1)
        # K is a per-tree statistic and is reported once, under the headline convention;
        # what the variants test is whether r moves with the convention, not K
        out[tag] = dict(n=int(n), d=int(d), r=round(s[0], 3),
                        p=round(emp_p(null[:, 0], s[0]), 5),
                        K_mean=float('nan'))
    for lam in (0.8, 0.5):
        X, labels, C, _ = trimmed(blk, root_template)
        Cl = cov_lambda(C, lam)
        n, d = X.shape
        P = whiten(X, Cl)
        Rc = corr_of(P)
        s, _ = spectrum(Rc, 1)
        null = null_spectrum(n, d, reps=1500, seed=31, kmax=1)
        out['lambda_%.1f' % lam] = dict(n=int(n), d=int(d), r=round(s[0], 3),
                                        p=round(emp_p(null[:, 0], s[0]), 5),
                                        K_mean=float('nan'))
    return out


def main():
    res = load('e26_real')
    io = __import__('io')
    root_template = io.open(NEWICK_CLADE, encoding='utf-8').read()
    rows = read_pantheria()
    log('species with any trait', len(rows))
    match = json.load(io.open(os.path.join(DATA, 'otol_match.json'), encoding='utf-8'))
    log('matched', sum(1 for v in match.values() if v))

    for name in BLOCKS:
        if name in res and 'contrast' in res[name]:
            log('skip', name)
            continue
        t0 = time.time()
        blk = prep_block(name, rows, match, root_template, None)
        if blk is None:
            log('no complete-case species for', name)
            continue
        blk['block'] = name
        blk['match'] = match
        o = block_stats(blk, root_template)
        if o is None:
            continue
        o['variants'] = variants(blk, root_template)
        res[name] = o
        save('e26_real', res)
        log('%-14s n=%4d d=%2d K=%.2f [%.2f,%.2f]  contrast r1=%.2f p=%.1e '
            'r2=%.2f p=%.2f r3=%.2f p=%.2f  tetrad %.2f (size %.2f)  hub=%s  %.0fs'
            % (name, o['n'], o['d'], o['K_mean'], o['K_min'], o['K_max'],
               o['contrast']['r'][0], o['contrast']['p'][0],
               o['contrast']['r'][1], o['contrast']['p'][1],
               o['contrast']['r'][2], o['contrast']['p'][2],
               o['tetrad']['reject'], o['tetrad']['size'],
               o['hub']['top_label'], time.time() - t0))


if __name__ == '__main__':
    main()
