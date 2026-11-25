# -*- coding: utf-8 -*-
"""Linear-Gaussian ADMG: maximum likelihood, BIC, a greedy structure search, and the
vanishing-tetrad test.

Model.  With observed X (n x p), a directed edge i -> j collected in B, and a bidirected
edge i <-> j collected in Omega off the diagonal,

    X_j = sum_{i in pa(j)} B[j, i] X_i + e_j,     e ~ N(0, Omega),

which is the acyclic directed mixed graph (ADMG) of Richardson (2003).  A bidirected edge
stands for a latent common cause that has been integrated out; the marginal covariance is

    Sigma = (I - B)^{-T} Omega (I - B)^{-1}.

Estimator.  The Gaussian log-likelihood depends on (B, Omega) only through
f(B, Omega) = log det Omega + tr(S Omega^{-1}) with S the sample covariance of the residuals
R = X (I - B).  Maximum likelihood for this family is solved in the literature by RICF, an
alternating conditional-fitting scheme (Drton, Fox & Wang 2020, Ann. Statist.), because the
problem is not convex.  Here the same objective is minimised directly by L-BFGS with the
analytic gradient derived in `_grad`, which reaches the same stationary points and lets the
bidirected pattern be held fixed exactly; a multi-start guards against local optima.  The
two routes target one likelihood, and `selftest` checks the fit against the closed-form
answer where a closed form exists.
"""
import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------- parameterisation
def _unpack(theta, dmask, bmask, p):
    """theta -> (B, Omega, logdet-free pieces).  Diagonal of Omega is exp(tau)."""
    nb = int(bmask.sum())
    nd = int(np.triu(dmask, 1).sum())
    B = np.zeros((p, p))
    B[bmask] = theta[:nb]
    Omega = np.zeros((p, p))
    Omega[np.diag_indices(p)] = np.exp(theta[nb:nb + p])
    iu = np.triu_indices(p, 1)
    sel = dmask[iu]
    Omega[iu[0][sel], iu[1][sel]] = theta[nb + p:nb + p + nd]
    Omega[iu[1][sel], iu[0][sel]] = theta[nb + p:nb + p + nd]
    return B, Omega


def _f_and_parts(theta, XtX, n, dmask, bmask, p):
    """Objective log det Omega + tr(Sigma^{-1} S) and the pieces the gradient needs.

    Sigma = A^{-1} Omega A^{-T} with A = I - B, hence Sigma^{-1} = A^T Omega^{-1} A and
    det Sigma = det Omega, so the objective is log det Omega + tr(A S A^T Omega^{-1})
    with S = X'X / n.  (Writing the trace with S conjugated the other way, as
    tr(A^T S A Omega^{-1}), is a different function and does not reduce to OLS when B has
    full support; that mistake was caught by `selftest` before any experiment ran.)
    """
    B, Om = _unpack(theta, dmask, bmask, p)
    try:
        np.linalg.cholesky(Om)
    except np.linalg.LinAlgError:
        return None, None
    logdet = float(np.linalg.slogdet(Om)[1])
    if not np.isfinite(logdet):
        return None, None
    OmInv = np.linalg.inv(Om)
    A = np.eye(p) - B
    S = XtX / n
    AS = A @ S
    tr = float(np.sum((AS @ A.T) * OmInv.T))
    return logdet + tr, (A, S, OmInv)


def _objective(theta, XtX, n, dmask, bmask, p):
    f, _ = _f_and_parts(theta, XtX, n, dmask, bmask, p)
    return 1e10 if f is None else f


def _grad(theta, XtX, n, dmask, bmask, p):
    f, parts = _f_and_parts(theta, XtX, n, dmask, bmask, p)
    if parts is None:
        return np.zeros_like(theta)
    A, S, OmInv = parts
    nb = int(bmask.sum())
    nd = int(np.triu(dmask, 1).sum())
    g = np.zeros_like(theta)
    # d/dA tr(A S A^T Om^-1) = Om^-1 A S + Om^-T A S^T = 2 Om^-1 A S, and dB = -dA.
    # (Conjugating the trace the other way gives 2 S A^T Om^-1, which is a different
    # matrix unless A is symmetric; `check_grad` would flag it immediately.)
    g[:nb] = (-2.0 * (OmInv @ A @ S))[bmask]
    # d/dOmega of log det Omega + tr(R Omega^-1) is Omega^-1 - Omega^-1 R Omega^-1,
    # with R = A S A^T the residual covariance
    Rres = A @ S @ A.T
    M = OmInv - OmInv @ Rres @ OmInv
    # diagonal enters as tau = log(omega_jj)
    g[nb:nb + p] = M[np.diag_indices(p)] * np.exp(theta[nb:nb + p])
    iu = np.triu_indices(p, 1)
    sel = dmask[iu]
    g[nb + p:nb + p + nd] = 2.0 * M[iu[0][sel], iu[1][sel]]
    return g


def ml_fit(X, bmask, dmask, starts=2, seed=0):
    """ML fit of the ADMG whose directed edges are the True entries of bmask and whose
    bidirected edges are the True entries of dmask (dmask diagonal is ignored)."""
    X = np.asarray(X, float)
    n, p = X.shape
    Xc = X - X.mean(0)
    XtX = Xc.T @ Xc
    S = XtX / n
    bmask = np.asarray(bmask, bool).copy()
    np.fill_diagonal(bmask, False)
    dmask = np.asarray(dmask, bool).copy()
    np.fill_diagonal(dmask, False)
    dmask = dmask | dmask.T
    nb = int(bmask.sum())
    nd = int(np.triu(dmask, 1).sum())
    rng = np.random.default_rng(seed)
    best = None
    for s in range(starts):
        th0 = np.zeros(nb + p + nd)
        th0[nb:nb + p] = np.log(np.maximum(np.diag(S), 1e-8) * (1.0 if s == 0 else 0.5))
        if s:
            th0[:nb] = rng.normal(0, 0.1, nb)
        res = minimize(_objective, th0, args=(XtX, n, dmask, bmask, p), jac=_grad,
                       method='L-BFGS-B', options=dict(maxiter=800, ftol=1e-12, gtol=1e-8))
        if best is None or res.fun < best.fun:
            best = res
    B, Om = _unpack(best.x, dmask, bmask, p)
    loglik = -0.5 * n * best.fun
    # number of free parameters
    k = nb + p + nd
    bic = -2 * loglik + k * np.log(n)
    return dict(B=B, Omega=Om, loglik=loglik, bic=bic, k=k, converged=best.success)


# ---------------------------------------------------------------- bic without a fit
def bic_dag(X, bmask):
    """BIC of a Gaussian DAG with independent noise, in closed form.

    With Omega diagonal the ML fit separates by equation and equals ordinary least
    squares, so the deviance is n * sum_j log(RSS_j / n) and the parameter count is p
    means plus p variances plus one slope per edge.  This is the score of Chickering
    (2002) and is what makes a full forward/backward search affordable at d = 15; the
    numerical ML above is only needed once bidirected edges enter.
    """
    X = np.asarray(X, float)
    n, p = X.shape
    Xc = X - X.mean(0)
    bmask = np.asarray(bmask, bool).copy()
    np.fill_diagonal(bmask, False)
    dev = 0.0
    k = 2 * p
    for j in range(p):
        pa = bmask[j]
        if pa.any():
            Xp = Xc[:, pa]
            beta = np.linalg.lstsq(Xp, Xc[:, j], rcond=None)[0]
            r = Xc[:, j] - Xp @ beta
            k += int(pa.sum())
        else:
            r = Xc[:, j]
        rss = max(float(r @ r), 1e-12)
        # the p equations contribute p separate terms: -2 loglik = n * sum_j log(rss_j/n).
        # Summing the rss first and taking one logarithm instead lets the well-explained
        # equations be diluted by the others, which makes every edge look like a bad buy.
        dev += n * np.log(rss / n)
    return dev + k * np.log(n)


def bic_admg(X, bmask, dmask):
    """BIC of an ADMG with bidirected edges, on the same scale as `bic_dag`.

    bic_dag counts 2p for the p means and p variances; ml_fit counts only the p variances,
    because the means are removed by centring.  The missing p * log n is added here, and
    leaving it out is invisible until one notices that the search keeps buying bidirected
    edges with parameters it was not being charged for.
    """
    n, p = np.shape(X)[0], np.shape(X)[1]
    r = ml_fit(X, bmask, dmask, starts=2)
    return r['bic'] + p * np.log(n), r


def _residual_corr(X, bmask):
    X = np.asarray(X, float)
    Xc = X - X.mean(0)
    p = X.shape[1]
    E = np.zeros_like(Xc)
    for j in range(p):
        pa = bmask[j]
        if pa.any():
            beta = np.linalg.lstsq(Xc[:, pa], Xc[:, j], rcond=None)[0]
            E[:, j] = Xc[:, j] - Xc[:, pa] @ beta
        else:
            E[:, j] = Xc[:, j]
    R = np.corrcoef(E, rowvar=False)
    R = np.nan_to_num(R)
    np.fill_diagonal(R, 0.0)
    return R


# ---------------------------------------------------------------- greedy search
def search_admg(X, max_bidir=4, max_edges=None, verbose=False, top_resid=6):
    """Structure search over ADMGs in three stages.

    Stage 1, forward: add the directed edge that lowers the closed-form BIC most, until no
    admissible addition helps.  Every candidate is an OLS fit, so this is cheap and exact.
    Stage 2, backward: delete edges and reverse edges while the BIC falls.
    Stage 3, bidirected: look at the correlation of the stage-2 residual matrix; the
    confounding the paper studies is rank one, so it is visible as one pair of residuals
    that stay correlated after the directed part is in.  Only the `top_resid` largest
    |residual correlation| pairs are tried as bidirected edges, scored by the full ADMG
    likelihood, and kept when the BIC falls.  Restricting the candidates this way is what
    keeps a 15-variable search affordable, and it is stated in the paper rather than
    hidden: a search that tried every pair would differ only in running time.

    Cycles in the directed part are excluded at every step.
    """
    X = np.asarray(X, float)
    n, p = X.shape
    if max_edges is None:
        max_edges = 2 * p
    bmask = np.zeros((p, p), bool)
    dmask = np.zeros((p, p), bool)
    cur = bic_dag(X, bmask)

    # ---- stage 1: forward
    while int(bmask.sum()) < max_edges:
        best = None
        for j in range(p):
            for i in range(p):
                if i == j or bmask[j, i]:
                    continue
                cand = bmask.copy(); cand[j, i] = True
                if _has_cycle(cand):
                    continue
                b = bic_dag(X, cand)
                if best is None or b < best[0]:
                    best = (b, cand)
        if best is None or best[0] > cur - 1e-8:
            break
        cur, bmask = best
        if verbose:
            print('   fwd edges %d bic %.2f' % (int(bmask.sum()), cur))

    # ---- stage 2: backward, plus reversals
    for _ in range(4 * p):
        moved = False
        E = [tuple(x) for x in np.argwhere(bmask)]
        best = None
        for (j, i) in E:                                  # delete
            cand = bmask.copy(); cand[j, i] = False
            b = bic_dag(X, cand)
            if best is None or b < best[0]:
                best = (b, cand)
        for (j, i) in E:                                  # reverse
            cand = bmask.copy(); cand[j, i] = False; cand[i, j] = True
            if _has_cycle(cand):
                continue
            b = bic_dag(X, cand)
            if best is None or b < best[0]:
                best = (b, cand)
        if best is not None and best[0] < cur - 1e-8:
            cur, bmask = best
            moved = True
        if not moved:
            break

    # ---- stage 3: bidirected edges where the residuals still correlate
    best_bic = bic_admg(X, bmask, dmask)[0]
    for _ in range(max_bidir):
        Rc = _residual_corr(X, bmask)
        iu = np.triu_indices(p, 1)
        order = np.argsort(-np.abs(Rc[iu]))
        cands = [(iu[0][t], iu[1][t]) for t in order[:top_resid]
                 if not dmask[iu[0][t], iu[1][t]]]
        best = None
        for (a, b_) in cands:
            cand = dmask.copy(); cand[a, b_] = cand[b_, a] = True
            bic, _r = bic_admg(X, bmask, cand)
            if best is None or bic < best[0]:
                best = (bic, cand)
        if best is None or best[0] > best_bic - 1e-8:
            break
        best_bic, dmask = best
        if verbose:
            print('   bidir %d bic %.2f' % (int(np.triu(dmask, 1).sum()), best_bic))

    fit = ml_fit(X, bmask, dmask, starts=2) if dmask.any() else None
    return dict(B=fit['B'] if fit is not None else None,
                bmask=bmask, dmask=dmask, bic=best_bic)


def _has_cycle(bmask):
    """True if the directed graph with adjacency bmask (bmask[j,i] = i->j) has a cycle."""
    p = bmask.shape[0]
    indeg = bmask.sum(0).astype(int)
    # bmask[j,i]: edge i->j, so in-degree of j is sum over i of bmask[j,i]
    indeg = bmask.sum(1).astype(int)
    stack = [j for j in range(p) if indeg[j] == 0]
    seen = 0
    children = {}
    for j in range(p):
        for i in range(p):
            if bmask[j, i]:
                children.setdefault(i, []).append(j)
    while stack:
        v = stack.pop(); seen += 1
        for u in children.get(v, []):
            indeg[u] -= 1
            if indeg[u] == 0:
                stack.append(u)
    return seen != p


# ---------------------------------------------------------------- vanishing tetrads
def tetrad_test(X, pairs=None, B=400, seed=0, resample=None):
    """Vanishing-tetrad test for a Gaussian DAG model.

    For any four variables i,j,k,l, a linear-Gaussian DAG with uncorrelated errors implies
    that all three tetrads vanish; a latent common cause makes them non-zero.  The statistic
    is the large-sample z of each tetrad with a bootstrap standard error, in the spirit of
    the constraint-based tests of Silva (2006) and Drton, Foygel & Sullivant (2009).
    Returns the share of tested tetrads that reject at 5%.

    `resample` is the matrix whose rows are bootstrapped.  It defaults to X, which is correct
    only when rows of X are exchangeable.  Comparative data are emphatically not: related
    species share covariance, so an iid row bootstrap understates the standard error and the
    test over-rejects on data that satisfies the null.  Passing the whitened contrasts -- whose
    rows *are* exchangeable under the Brownian null -- restores the size of the test.  The
    statistic itself is always evaluated on X, so the two choices differ only in their null.
    """
    X = np.asarray(X, float)
    n, p = X.shape
    Bm = X if resample is None else np.asarray(resample, float)
    rng = np.random.default_rng(seed)

    def tetrads(S):
        out = []
        for i in range(p):
            for j in range(i + 1, p):
                for k in range(j + 1, p):
                    for l in range(k + 1, p):
                        out.append((S[i, j] * S[k, l] - S[i, k] * S[j, l],
                                    (i, j, k, l)))
        return out

    if p < 4:
        return dict(n_tetrads=0, reject_share=float('nan'), median_abs_z=float('nan'),
                    max_abs_z=float('nan'))
    S = np.corrcoef(X, rowvar=False)
    obs = tetrads(S)
    if not obs:
        return dict(n_tetrads=0, reject_share=float('nan'), median_abs_z=float('nan'),
                    max_abs_z=float('nan'))
    vals = np.array([t for t, _ in obs])
    nb = Bm.shape[0]
    boots = np.empty((B, len(obs)))
    for b in range(B):
        idx = rng.integers(0, nb, nb)
        Sb = np.corrcoef(Bm[idx], rowvar=False)
        boots[b] = [t for t, _ in tetrads(Sb)]
    se = boots.std(0)
    z = np.abs(vals) / np.maximum(se, 1e-12)
    return dict(n_tetrads=len(obs),
                reject_share=float(np.mean(z > 1.96)),
                median_abs_z=float(np.median(z)),
                max_abs_z=float(z.max()))


# ---------------------------------------------------------------- self-test
def check_grad(seed=0):
    """Analytic vs central-difference gradient of the objective. Run before any fit."""
    rng = np.random.default_rng(seed)
    p, n = 5, 300
    X = rng.normal(size=(n, p)) @ np.linalg.cholesky(0.5 * np.eye(p) + 0.5 * np.ones((p, p)))
    Xc = X - X.mean(0)
    XtX = Xc.T @ Xc
    bmask = np.zeros((p, p), bool)
    bmask[1, 0] = bmask[2, 0] = bmask[2, 1] = bmask[3, 2] = True
    dmask = np.zeros((p, p), bool); dmask[0, 4] = dmask[4, 0] = True
    nb = int(bmask.sum()); nd = 1
    th = rng.normal(0, 0.4, nb + p + nd)
    th[nb:nb + p] = np.log(np.ones(p))
    g = _grad(th, XtX, n, dmask, bmask, p)
    gn = np.zeros_like(th)
    h = 1e-6
    for k in range(len(th)):
        e = np.zeros_like(th); e[k] = h
        gn[k] = (_objective(th + e, XtX, n, dmask, bmask, p) -
                 _objective(th - e, XtX, n, dmask, bmask, p)) / (2 * h)
    err = np.abs(g - gn).max()
    print('gradient check: max |analytic - numeric| = %.2e  (should be < 1e-4)' % err)
    return err


def selftest():
    """Fit an ADMG whose answer is known in closed form and check the recovery."""
    assert check_grad() < 1e-3, 'gradient is wrong, stop'
    rng = np.random.default_rng(0)
    n, p = 4000, 4
    B = np.zeros((p, p))
    B[1, 0] = 0.8; B[2, 0] = 0.5; B[2, 1] = -0.4
    om = np.array([1.0, 2.0, 0.5, 1.5])
    Om = np.diag(om)
    Om[0, 3] = Om[3, 0] = 0.6 * np.sqrt(om[0] * om[3])   # latent common cause 0 <-> 3
    A = np.linalg.inv(np.eye(p) - B)
    Sig = A @ Om @ A.T
    X = rng.multivariate_normal(np.zeros(p), Sig, n)

    bmask = B != 0
    dmask = np.zeros((p, p), bool); dmask[0, 3] = dmask[3, 0] = True
    r = ml_fit(X, bmask, dmask, starts=3)
    print('recovered B\n', np.round(r['B'], 3))
    print('true      B\n', np.round(B, 3))
    print('max |B err| %.4f' % np.abs(r['B'] - B).max())
    print('Omega diag', np.round(np.diag(r['Omega']), 3), 'true', om)
    print('Omega[0,3] %.3f  true %.3f' % (r['Omega'][0, 3], Om[0, 3]))

    # no-latent case: ML must equal OLS on the true DAG
    r2 = ml_fit(X, bmask, np.zeros((p, p), bool), starts=1)
    print('no-bidirected max |B err| %.4f' % np.abs(r2['B'] - B).max())

    # bidirected detection: does BIC prefer the true pattern?
    s_bd = ml_fit(X, bmask, dmask, starts=3)['bic']
    s_no = ml_fit(X, bmask, np.zeros((p, p), bool), starts=3)['bic']
    print('BIC with bidirected %.2f  without %.2f  (lower is the true model)' % (s_bd, s_no))
    print('tetrad test on this confounded data:', tetrad_test(X, B=80))

    # and on data with no latent at all: X = E A^T with E of diagonal covariance
    X0 = (rng.normal(size=(n, p)) * np.sqrt(om)) @ A.T
    print('tetrad test on unconfounded data:', tetrad_test(X0, B=80))


if __name__ == '__main__':
    selftest()
