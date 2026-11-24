# -*- coding: utf-8 -*-
"""Second rebuilt pipeline for the evo_causal revision.

E5  Dose-response re-parameterised on a MEASURABLE axis.
    Reviewer 3 (#5) is right: the OU rate alpha is a property of the generator, not of a
    dataset. No comparative biologist can report "my data have alpha = 0.1". Every simulated
    dataset is therefore scored with Blomberg's K (GLS, the quantity people actually report
    for empirical data), and performance is plotted against that measured K. This turns the
    "wall" from a statement about a hidden generator parameter into a statement about an
    observable of the data.

E6  A third methodological family: constraint-based discovery (PC with Fisher-z on PIC).
    Reviewers noted that NOTEARS / DAGMA / GOLEM / transformer are all score-or-gradient
    based. If the wall also holds for a constraint-based method it is not an artefact of one
    family.

E7  Phylogenetic path analysis (PPA) as a competing paradigm (Reviewer 3 #2).
    PPA does not discover; it compares pre-specified candidate DAGs under a phylogenetic
    GLS likelihood. We implement exactly that (full enumeration over one topological order)
    and report two regimes: candidate set containing the truth, and candidate set not
    containing it. This quantifies the real trade-off instead of dismissing PPA.
"""
import os, sys, json, time, itertools, argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from run_revision import (build_ultrametric_tree, mrca_heights, cov_brownian, cov_ou,
                          chol_psd, make_dag, f1_topk, auroc, blomberg_k, pic_of,
                          corr_scores, simulate, OUTDIR, load, save)

rng_global = np.random.default_rng(7)


# ------------------------------------------------------------------ E5
def exp_k_axis():
    """F1 as a function of the MEASURED phylogenetic signal, not the generator parameter."""
    res = load('e5_k_axis')
    grid = [(d, noise, lam) for d in (30, 50)
            for noise in ('laplace',)
            for lam in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)]
    for d, noise, lam in grid:
        key = f'd{d}_{noise}_lam{lam}'
        if key in res:
            continue
        t0 = time.time()
        Ks, fp, fr, aucs = [], [], [], []
        for seed in (0, 1, 2):
            X, W, edges, tree, H, C = simulate(n=200, d=d, k=10, lam=lam,
                                               alpha_ou=0.0, noise=noise, seed=seed)
            k = len(edges)
            P = pic_of(X, tree)
            cc, cr = corr_scores(P), corr_scores(X)

            # Blomberg's K on the first trait is unstable; average over traits for a
            # single dataset-level summary, which is what an empirical study would report.
            V = cov_brownian(H)
            ks = [blomberg_k(X[:, j], V, nsim=120, seed=seed * 17 + j) for j in range(min(d, 6))]
            Ks.append(float(np.nanmean(ks)))

            fp.append(f1_topk(cc, W, k))
            fr.append(f1_topk(cr, W, k))
            aucs.append(auroc(cc, W))
        res[key] = {'d': d, 'noise': noise, 'lam': lam,
                    'K_measured': round(float(np.mean(Ks)), 4),
                    'K_sd': round(float(np.std(Ks)), 4),
                    'f1_pic': round(float(np.mean(fp)), 4),
                    'f1_raw': round(float(np.mean(fr)), 4),
                    'auroc_pic': round(float(np.nanmean(aucs)), 4),
                    'sec': round(time.time() - t0, 1)}
        print(f"  [E5] {key:22s} K={res[key]['K_measured']:.3f} "
              f"PIC={res[key]['f1_pic']:.3f} raw={res[key]['f1_raw']:.3f} "
              f"auc={res[key]['auroc_pic']:.3f} ({res[key]['sec']}s)")
        save('e5_k_axis', res)
    return res


# ------------------------------------------------------------------ E6  PC algorithm
def _fisher_z_p(r, n, k):
    """Two-sided p-value for a partial correlation r estimated from n samples with k
    conditioning variables."""
    if n - k - 3 <= 0 or abs(r) >= 1.0:
        return 1.0
    z = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(n - k - 3)
    return float(2 * (1 - stats.norm.cdf(abs(z))))


def _partial_corr(C, i, j, S):
    """Partial correlation of i and j given S, from a correlation matrix."""
    if not S:
        return C[i, j]
    idx = list(S) + [i, j]
    sub = C[np.ix_(idx, idx)]
    try:
        P = np.linalg.pinv(sub)
    except Exception:
        return 0.0
    denom = np.sqrt(abs(P[-2, -2] * P[-1, -1]))
    return float(-P[-2, -1] / denom) if denom > 1e-12 else 0.0


def pc_skeleton(P, alpha=0.01, max_depth=3):
    """PC algorithm, skeleton phase only. Returns an undirected adjacency matrix.
    Skeleton F1 is the fair comparison: the gradient methods are also scored on edge sets,
    and orienting v-structures is a separate problem that would confound the comparison."""
    d = P.shape[1]
    n = P.shape[0]
    C = np.corrcoef(P.T)
    adj = np.ones((d, d), bool)
    np.fill_diagonal(adj, False)
    depth = 0
    while depth <= max_depth:
        removed = []
        for i in range(d):
            for j in range(i + 1, d):
                if not adj[i, j]:
                    continue
                nbrs = [x for x in range(d) if adj[i, x] and x != j]
                if len(nbrs) < depth:
                    continue
                found = False
                for S in itertools.combinations(nbrs, depth):
                    r = _partial_corr(C, i, j, S)
                    if _fisher_z_p(r, n, depth) > alpha:
                        found = True
                        break
                if found:
                    removed.append((i, j))
        if not removed:
            break
        for i, j in removed:
            adj[i, j] = adj[j, i] = False
        depth += 1
    return adj


def f1_undirected(est, W):
    T = (np.abs(W) > 0); T = T | T.T
    tp = float((est & T).sum()); fp = float((est & ~T).sum()); fn = float((~est & T).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def exp_pc():
    """Constraint-based family on PIC contrasts, scored on the skeleton."""
    res = load('e6_pc')
    for d in (15, 30):
        for lam in (0.0, 1.0, 2.0):
            key = f'd{d}_lam{lam}'
            if key in res:
                continue
            t0 = time.time()
            f_pc, f_corr = [], []
            for seed in (0, 1, 2):
                X, W, edges, tree, H, C = simulate(n=200, d=d, k=10, lam=lam,
                                                   alpha_ou=0.0, noise='laplace', seed=seed)
                P = pic_of(X, tree)
                t1 = time.time()
                sk = pc_skeleton(P, alpha=0.01, max_depth=3)
                f_pc.append(f1_undirected(sk, W))
                cc = corr_scores(P)
                k = len(edges)
                ti = np.triu_indices(d, 1)
                th = np.sort(np.abs(cc[ti]))[::-1][:k][-1]
                adj = (np.abs(cc) >= th); np.fill_diagonal(adj, False)
                f_corr.append(f1_undirected(adj | adj.T, W))
            res[key] = {'d': d, 'lam': lam,
                        'f1_pc': round(float(np.mean(f_pc)), 4),
                        'f1_pic_corr': round(float(np.mean(f_corr)), 4),
                        'sec': round(time.time() - t0, 1)}
            print(f"  [E6] {key:12s} PC={res[key]['f1_pc']:.3f} "
                  f"PIC+corr={res[key]['f1_pic_corr']:.3f} ({res[key]['sec']}s)")
            save('e6_pc', res)
    return res


# ------------------------------------------------------------------ E7  PPA
def _gls_bic(X, A, Vinv, logdetV, n):
    """BIC of a linear SEM whose structure is A (parents of column j = nonzero rows of A[:,j]),
    estimated under the known phylogenetic covariance V."""
    d = X.shape[1]
    tot = 0.0
    k = 0
    for j in range(d):
        pa = np.where(A[:, j] != 0)[0]
        y = X[:, j]
        if len(pa) == 0:
            beta = np.zeros((0,))
            r = y - y.mean()
        else:
            Z = X[:, pa]
            # GLS: beta = (Z' V^-1 Z)^-1 Z' V^-1 y
            Zi = Vinv @ Z
            yi = Vinv @ y
            try:
                beta = np.linalg.solve(Z.T @ Zi, Z.T @ yi)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(Z, y, rcond=None)[0]
            r = y - Z @ beta
        s2 = float(r @ (Vinv @ r)) / n
        s2 = max(s2, 1e-12)
        tot += n * np.log(s2)
        k += len(pa) + 1
    return tot + logdetV + k * np.log(n)


def exp_ppa():
    """Phylogenetic path analysis: enumerate DAGs over a fixed order, score by GLS-BIC,
    report accuracy and the failure mode when the truth is outside the candidate set."""
    res = load('e7_ppa')
    d = 5
    order_pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    n_cand = 2 ** len(order_pairs)
    for lam in (0.0, 1.0, 2.0):
        for regime in ('full', 'restricted'):
            key = f'd{d}_lam{lam}_{regime}'
            if key in res:
                continue
            t0 = time.time()
            exact, best_f1, pic_f1, n_models = [], [], [], []
            for seed in (0, 1, 2):
                X, W, edges, tree, H, C = simulate(n=120, d=d, k=6, lam=lam,
                                                   alpha_ou=0.0, noise='laplace', seed=seed)
                V = cov_brownian(H) * 1.0
                Vinv = np.linalg.inv(V)
                logdetV = float(np.linalg.slogdet(V)[1])

                cands = []
                for mask in range(n_cand):
                    A = np.zeros((d, d))
                    for b, (i, j) in enumerate(order_pairs):
                        if mask >> b & 1:
                            A[i, j] = 1.0
                    if regime == 'restricted':
                        # A biologist would propose a handful of competing hypotheses.
                        # Keep only models with <= 3 edges: a realistic candidate set.
                        if A.sum() > 3:
                            continue
                    cands.append(A)
                scores = [_gls_bic(X, A, Vinv, logdetV, X.shape[0]) for A in cands]
                best = cands[int(np.argmin(scores))]
                exact.append(float(np.allclose(best, (np.abs(W) > 0).astype(float))))
                best_f1.append(f1_undirected((best != 0) | (best.T != 0), W))
                P = pic_of(X, tree)
                cc = corr_scores(P)
                kk = len(edges)
                ti = np.triu_indices(d, 1)
                th = np.sort(np.abs(cc[ti]))[::-1][:kk][-1]
                adj = (np.abs(cc) >= th); np.fill_diagonal(adj, False)
                pic_f1.append(f1_undirected(adj | adj.T, W))
                n_models.append(len(cands))
            res[key] = {'d': d, 'lam': lam, 'regime': regime,
                        'n_candidate_models': int(np.mean(n_models)),
                        'ppa_exact_recovery': round(float(np.mean(exact)), 3),
                        'ppa_f1': round(float(np.mean(best_f1)), 4),
                        'pic_corr_f1': round(float(np.mean(pic_f1)), 4),
                        'sec': round(time.time() - t0, 1)}
            print(f"  [E7] {key:22s} models={res[key]['n_candidate_models']:4d} "
                  f"PPA_exact={res[key]['ppa_exact_recovery']:.2f} "
                  f"PPA_F1={res[key]['ppa_f1']:.3f} PIC+corr_F1={res[key]['pic_corr_f1']:.3f} "
                  f"({res[key]['sec']}s)")
            save('e7_ppa', res)
    return res


# ------------------------------------------------------------------ E8  FCI
def exp_fci():
    """FCI on PIC contrasts. FCI is the only family here that is explicitly built for latent
    confounding, which is exactly what a phylogenetically evolving unmeasured trait is. We
    report skeleton F1 AND the fraction of adjacencies that carry a latent-discordant
    endpoint mark, because that mark is the method telling us a confounder is present."""
    from causallearn.search.ConstraintBased.FCI import fci
    res = load('e8_fci')
    for d in (10, 15):
        for lam in (0.0, 1.0, 2.0):
            key = f'd{d}_lam{lam}'
            if key in res:
                continue
            t0 = time.time()
            f_fci, f_corr, frac_marks = [], [], []
            for seed in (0, 1, 2):
                X, W, edges, tree, H, C = simulate(n=200, d=d, k=8, lam=lam,
                                                   alpha_ou=0.0, noise='laplace', seed=seed)
                P = pic_of(X, tree)
                G, edge_list = fci(P, verbose=False, show_progress=False)
                adj = np.zeros((d, d), bool)
                marks = 0
                tot = 0
                for e in edge_list:
                    i = int(str(e.get_node1().get_name()).lstrip('X'))
                    j = int(str(e.get_node2().get_name()).lstrip('X'))
                    if i >= d or j >= d:
                        continue
                    adj[i, j] = adj[j, i] = True
                    tot += 1
                    s1, s2 = str(e.get_endpoint1()), str(e.get_endpoint2())
                    if ('CIRCLE' in s1 or 'CIRCLE' in s2
                            or ('ARROW' in s1 and 'ARROW' in s2)):
                        marks += 1
                f_fci.append(f1_undirected(adj, W))
                frac_marks.append(marks / tot if tot else 0.0)
                cc = corr_scores(P)
                kk = len(edges)
                ti = np.triu_indices(d, 1)
                th = np.sort(np.abs(cc[ti]))[::-1][:kk][-1]
                a2 = (np.abs(cc) >= th); np.fill_diagonal(a2, False)
                f_corr.append(f1_undirected(a2 | a2.T, W))
            res[key] = {'d': d, 'lam': lam,
                        'f1_fci': round(float(np.mean(f_fci)), 4),
                        'f1_pic_corr': round(float(np.mean(f_corr)), 4),
                        'frac_latent_marks': round(float(np.mean(frac_marks)), 3),
                        'sec': round(time.time() - t0, 1)}
            print(f"  [E8] {key:12s} FCI={res[key]['f1_fci']:.3f} "
                  f"PIC+corr={res[key]['f1_pic_corr']:.3f} "
                  f"latent_marks={res[key]['frac_latent_marks']:.2f} ({res[key]['sec']}s)")
            save('e8_fci', res)
    return res


# ------------------------------------------------------------------ E9  sample-size scaling
def exp_nsense():
    """How many species does it take? F1 vs n at a fixed, empirically plausible K.
    This is the question an empirical reader actually asks, and neither the rejected
    manuscript nor its reviews answered it."""
    res = load('e9_nsense')
    from run_revision import run_notears
    for lam in (1.0, 2.0):
        for n in (100, 200, 500, 1000, 2000):
            key = f'lam{lam}_n{n}'
            if key in res:
                continue
            t0 = time.time()
            f_pic, f_pc, f_nt, Ks = [], [], [], []
            for seed in (0, 1, 2):
                X, W, edges, tree, H, C = simulate(n=n, d=30, k=10, lam=lam,
                                                   alpha_ou=0.0, noise='laplace', seed=seed)
                P = pic_of(X, tree)
                cc = corr_scores(P)
                kk = len(edges)
                ti = np.triu_indices(30, 1)
                th = np.sort(np.abs(cc[ti]))[::-1][:kk][-1]
                a2 = (np.abs(cc) >= th); np.fill_diagonal(a2, False)
                f_pic.append(f1_undirected(a2 | a2.T, W))
                f_pc.append(f1_undirected(pc_skeleton(P, alpha=0.01, max_depth=3), W))
                Am = run_notears(P)
                if Am is not None:
                    f_nt.append(f1_undirected((np.abs(Am) > 0) | (np.abs(Am.T) > 0), W))
                V = cov_brownian(H)
                Ks.append(float(np.nanmean([blomberg_k(X[:, j], V, nsim=100, seed=seed * 13 + j)
                                            for j in range(4)])))
            res[key] = {'lam': lam, 'n': n,
                        'K_measured': round(float(np.mean(Ks)), 3),
                        'f1_pic_corr': round(float(np.mean(f_pic)), 3),
                        'f1_pc': round(float(np.mean(f_pc)), 3),
                        'f1_notears': round(float(np.mean(f_nt)), 3) if f_nt else None,
                        'sec': round(time.time() - t0, 1)}
            r = res[key]
            print(f"  [E9] {key:14s} K={r['K_measured']:.2f} PIC+corr={r['f1_pic_corr']:.3f} "
                  f"PC={r['f1_pc']:.3f} NOTEARS={r['f1_notears']} ({r['sec']}s)")
            save('e9_nsense', res)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='all')
    a = ap.parse_args()
    which = a.only.split(',') if a.only != 'all' else ['e5', 'e6', 'e7', 'e8']
    print('=' * 78)
    if 'e5' in which:
        print('\nE5  dose-response on the measured Blomberg K axis\n'); exp_k_axis()
    if 'e6' in which:
        print('\nE6  constraint-based family (PC skeleton on PIC)\n'); exp_pc()
    if 'e7' in which:
        print('\nE7  phylogenetic path analysis baseline\n'); exp_ppa()
    if 'e8' in which:
        print('\nE8  FCI (latent-confounding-aware family)\n'); exp_fci()
    if 'e9' in which:
        print('\nE9  sample-size scaling\n'); exp_nsense()
    print('\nresults ->', OUTDIR)
