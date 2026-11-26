# -*- coding: utf-8 -*-
"""The screen itself.

    r1        lambda_1 / mean(remaining eigenvalues) of the contrast correlation matrix.
              Under independent Brownian traits the whitened data are independent standard
              normals, so the null distribution of r1 depends on (n, d) only and can be
              simulated exactly: no tree enters it, and no branch-length convention biases it.

    shares    squared loadings of the variables on the leading eigenvector.  This is the
              question the ratio cannot answer: an r1 of 3.5 that sits evenly across six
              traits means something different from an r1 of 3.5 that sits on one.

    hub       which variable a top-d ranking of |R| collapses onto.

    deflated  the same ratio after subtracting the leading component, with its own p-value.
              A diagnostic that fires and a repair that works are different claims; this
              separates them.
"""
import numpy as np

from . import tree as T


class ScreenResult(object):
    def __init__(self, d):
        self.n = 0
        self.d = d
        self.r1 = float('nan')
        self.p = float('nan')
        self.threshold = float('nan')
        self.eigenvalues = None
        self.share = {}
        self.top = None
        self.top_share = float('nan')
        self.entropy = float('nan')
        self.deflated_r1 = float('nan')
        self.deflated_p = float('nan')
        self.hub = None
        self.hub_share = float('nan')
        self.verdict = ''

    def __repr__(self):
        return ('<ScreenResult n=%d d=%d  r1=%.2f (threshold %.2f, p=%.3g)  '
                'top=%s %.2f  deflated r1=%.2f (p=%.2f)>\n  %s'
                % (self.n, self.d, self.r1, self.threshold, self.p, self.top,
                   self.top_share, self.deflated_r1, self.deflated_p, self.verdict))


def _standardise(X):
    X = np.asarray(X, float)
    X = X - X.mean(0)
    sd = X.std(0)
    sd[sd < 1e-12] = 1.0
    return X / sd


def _whiten(X, C):
    L = np.linalg.cholesky(C + 1e-10 * np.eye(C.shape[0]))
    return np.linalg.solve(L, X)


def _corr(M):
    M = M - M.mean(0)
    return np.corrcoef(M, rowvar=False)


def _r1(R):
    w = np.linalg.eigvalsh(R)[::-1]
    return float(w[0] / max(w[1:].mean(), 1e-12)), w


def _null_r1(n, d, nsim, seed):
    rng = np.random.default_rng(seed)
    out = np.empty(nsim)
    for b in range(nsim):
        out[b] = _r1(np.corrcoef(rng.normal(size=(n, d)), rowvar=False))[0]
    return out


def _hub(R, k):
    d = R.shape[0]
    A = np.abs(R - np.diag(np.diag(R)))
    iu = np.triu_indices(d, 1)
    v = A[iu]
    keep = np.argsort(v)[::-1][:k]
    deg = np.zeros(d)
    for t in keep:
        deg[iu[0][t]] += 1
        deg[iu[1][t]] += 1
    o = np.argsort(deg)[::-1]
    return int(o[0]), float(deg[o[0]] / max(2.0 * k, 1.0))


def screen(X, newick=None, species=None, labels=None, tree='grafen', rho=1.0, nsim=4000,
           seed=11, alpha=0.05):
    """Screen a comparative matrix.

    Parameters
    ----------
    X : (n, d) array
        Trait values, one row per species.  Any column whose ratio of max to min exceeds
        twenty is log10-transformed first, the usual convention for allometric variables.
    newick : str, optional
        Tree as a Newick string.  Required unless a covariance matrix is passed as `tree`.
    species : list of str, optional
        Tip labels, in the row order of X.  This is the species axis, not the trait axis.
    labels : list of str, optional
        Trait names, one per column of X.
    tree : str or (n, n) array
        'grafen' (default), 'equal', or a precomputed covariance matrix on those species.
    rho : float
        Exponent of Grafen's power convention; 1.0 is the node-count convention.
    nsim : int
        Null draws for the threshold.  4000 gives a stable 95th percentile to about 0.01.

    Returns
    -------
    ScreenResult
    """
    X = np.asarray(X, float)
    if X.ndim != 2:
        raise ValueError('X must be 2-D (species x traits)')
    n, d = X.shape
    if d < 4:
        raise ValueError('the screen needs at least four traits; with three there is nothing to '
                         'distinguish from noise')
    if species is None:
        species = ['s%d' % (i + 1) for i in range(n)]
    species = list(species)
    if len(species) != n:
        raise ValueError('species has %d entries, X has %d rows' % (len(species), n))
    if labels is None:
        labels = ['v%d' % (j + 1) for j in range(d)]
    labels = list(labels)
    if len(labels) != d:
        raise ValueError('labels has %d entries, X has %d columns' % (len(labels), d))

    Xs = _standardise(X)
    # log-transform columns spanning more than a factor of twenty, then standardise again
    need = []
    for j in range(d):
        col = np.asarray(X[:, j], float)
        if col.min() > 0 and col.max() / max(col.min(), 1e-12) > 20:
            need.append(j)
    if need:
        Xl = np.array(X, float, copy=True)
        for j in need:
            Xl[:, j] = np.log10(Xl[:, j])
        Xs = _standardise(Xl)

    if isinstance(tree, np.ndarray):
        C = np.asarray(tree, float)
        if C.shape != (n, n):
            raise ValueError('covariance is %s, expected (%d, %d)' % (C.shape, n, n))
    else:
        if newick is None:
            raise ValueError('either newick or a covariance matrix is required')
        if tree not in ('grafen', 'equal'):
            raise ValueError("tree must be 'grafen', 'equal' or a covariance matrix, got %r"
                             % (tree,))
        root = T.parse_newick(newick)
        root = T.prune(root, species)
        if tree == 'equal':
            T.equal_lengths(root)
        else:
            T.grafen_lengths(root, rho=rho)
        lo, hi = T.check_ultrametric(root)
        if hi - lo > 1e-6:
            raise ValueError('tips are not contemporaneous (%.4f to %.4f); contrasts require it'
                             % (lo, hi))
        C = T.brownian_cov(root, species)

    P = _whiten(Xs, C)
    R = _corr(P)
    r1, w = _r1(R)
    null = _null_r1(n, d, nsim, seed)
    thr = float(np.quantile(null, 0.95))
    p = float((1.0 + np.sum(null >= r1)) / (nsim + 1.0))

    evals, evecs = np.linalg.eigh(R)
    u = evecs[:, -1]
    u = u * np.sign(u[np.argmax(np.abs(u))])
    sh = u ** 2 / (u ** 2).sum()
    order = np.argsort(sh)[::-1]

    Rd = R - evals[-1] * np.outer(u, u)
    rd, _ = _r1(Rd)
    pd = float((1.0 + np.sum(null >= rd)) / (nsim + 1.0))

    hi_, hs = _hub(R, d)

    res = ScreenResult(d)
    res.n = n
    res.r1 = round(r1, 3)
    res.p = p
    res.threshold = round(thr, 3)
    res.eigenvalues = [round(float(x), 3) for x in w]
    res.share = dict((labels[j], round(float(sh[j]), 3)) for j in order)
    res.top = labels[order[0]]
    res.top_share = round(float(sh[order[0]]), 3)
    res.entropy = round(float(-(sh * np.log(sh + 1e-12)).sum() / np.log(d)), 3)
    res.deflated_r1 = round(rd, 3)
    res.deflated_p = pd
    res.hub = labels[hi_]
    res.hub_share = round(hs, 3)
    if p > alpha:
        res.verdict = ('no single direction dominates; the matrix can be analysed at the level '
                       'of edges')
    elif res.deflated_p <= alpha:
        res.verdict = ('one direction dominates and subtracting it does not clear the excess: '
                       'the direction is a property of the trait set, and edge-level claims '
                       'are not identified')
    else:
        res.verdict = ('one direction dominates but subtracting it returns the matrix to the '
                       'null; the leading variable (%s) can be conditioned on' % res.top)
    return res
