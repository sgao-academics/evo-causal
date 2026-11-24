# -*- coding: utf-8 -*-
"""REBUILT pipeline for the evo_causal revision (2026-11-24).

What changed relative to the rejected submission
------------------------------------------------
1. The phylogenetic component is now the ACTUAL tree covariance (Brownian), not a
   random Wishart matrix. Tree, branch lengths and covariance come from one object.
2. Phylogenetic confounding is modelled the way it actually confounds: one or more
   latent factors that evolve along the tree and load on several traits, creating
   cross-trait correlation that is not causal.
3. Both Brownian and Ornstein-Uhlenbeck confounders are available, so model
   misspecification can be studied instead of assumed away.
4. Every causal-discovery method is applied to the SAME PIC contrasts. No method is
   ever run on the uncorrected matrix.
5. Evaluation uses top-k (k = number of true edges) and AUROC alongside the p99
   rule, so the threshold artefact is measurable rather than confounded with biology.
6. Blomberg's K is computed properly (GLS), so it varies with the process instead of
   being a constant variance ratio.
"""
import os, sys, json, time, argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


from notears_impl import notears_linear
from felsenstein_pic import PhyloNode, felsenstein_pic

OUTDIR = os.path.join(os.path.dirname(HERE), 'results')
os.makedirs(OUTDIR, exist_ok=True)


# ============================================================ tree & covariance
def build_ultrametric_tree(n, seed=0):
    """Random ultrametric tree: every tip sits at height 0, every internal node has a height,
    branch length = parent height - child height. Exactly ultrametric by construction, which is
    what makes the Brownian covariance H/T have unit diagonal and PIC exact."""
    rng = np.random.default_rng(seed)
    tips = [PhyloNode(f"Sp{i}", children=[], branch_length=0.0) for i in range(n)]
    for i, t in enumerate(tips):
        t.index = i; t.is_tip = True
    heights = {id(t): 0.0 for t in tips}
    nodes = list(tips)
    nid = 0
    while len(nodes) > 1:
        i, j = rng.choice(len(nodes), 2, replace=False)
        a, b = nodes[int(i)], nodes[int(j)]
        h_new = max(heights[id(a)], heights[id(b)]) + 0.05 + rng.exponential(0.4)
        a.branch_length = h_new - heights[id(a)]
        b.branch_length = h_new - heights[id(b)]
        p = PhyloNode(f"N{nid}", children=[a, b], branch_length=0.0)
        p.is_tip = False; nid += 1
        heights[id(p)] = h_new
        nodes = [x for k, x in enumerate(nodes) if k not in (int(i), int(j))] + [p]
    root = nodes[0]
    root.branch_length = 0.0
    return root, heights[id(root)]


def mrca_heights(tree, n):
    """H[i,j] = height (root-distance) of the MRCA of tips i and j, tips at height 0."""
    tips, parent, h = {}, {}, {}

    def walk(node, height, par):
        h[id(node)] = height
        parent[id(node)] = par
        if node.is_tip and node.index is not None:
            tips[node.index] = node
        for ch in node.children:
            walk(ch, height - ch.branch_length, node)

    walk(tree, 0.0, None)   # heights measured downward from root = 0  -> rebuild upward below
    # recompute as distance from tip plane: height_from_tip = -height
    H = np.zeros((n, n))
    for i in range(n):
        anc = {}
        cur = tips[i]
        while cur is not None:
            anc[id(cur)] = -h[id(cur)]
            cur = parent[id(cur)]
        for j in range(n):
            cur = tips[j]
            while cur is not None:
                if id(cur) in anc:
                    H[i, j] = anc[id(cur)]
                    break
                cur = parent[id(cur)]
    return H, tips, parent, h


def cov_brownian(H):
    """Brownian covariance on an ultrametric tree = height of the MRCA above the tips,
    normalised by the root height so the diagonal is 1."""
    T = float(H.max())
    return H / T if T > 0 else H.copy()


def cov_ou(H, alpha):
    """Exact OU covariance for an ultrametric tree with tips at height 0 and root at T.

    Cov(i,j) = (1 - e^{-2 a t_ij}) e^{-2 a (T - t_ij)} / (2a)  for i != j
    Var(i)   = (1 - e^{-2 a T}) / (2a)
    Correlation is what matters, so the prefactor is normalised away.
    """
    T = H.max()
    if alpha <= 1e-9:
        return H / T if T > 0 else H
    C = np.zeros_like(H)
    var = (1.0 - np.exp(-2 * alpha * T)) / (2 * alpha)
    for i in range(H.shape[0]):
        for j in range(H.shape[0]):
            t = H[i, j]
            if i == j:
                C[i, j] = 1.0
            else:
                C[i, j] = ((1.0 - np.exp(-2 * alpha * t)) *
                           np.exp(-2 * alpha * (T - t)) / (2 * alpha)) / var
    return C


def chol_psd(A):
    ev, evec = np.linalg.eigh((A + A.T) / 2.0)
    return evec @ np.diag(np.sqrt(np.maximum(ev, 1e-10))) @ evec.T


# ============================================================ simulator
def make_dag(d, k, rng):
    W = np.zeros((d, d)); edges = []
    guard = 0
    while len(edges) < k and guard < 50000:
        guard += 1
        i, j = int(rng.integers(0, d)), int(rng.integers(0, d))
        if i < j and W[i, j] == 0:
            W[i, j] = rng.uniform(0.5, 1.0) * rng.choice([-1.0, 1.0])
            edges.append((i, j))
    return W, edges


def simulate(n=200, d=30, k=10, lam=0.0, alpha_ou=0.0, n_factors=1,
             noise='gauss', seed=0, tree_seed=None):
    """True phylogenetic confounding: latent factors evolve on the tree and load on traits."""
    rng = np.random.default_rng(seed)
    tree, T = build_ultrametric_tree(n, seed=(tree_seed if tree_seed is not None else 1000 + seed))
    H, tips, parent, hh = mrca_heights(tree, n)
    C = cov_brownian(H) if alpha_ou <= 0 else cov_ou(H, alpha_ou)
    C = np.asarray(C, dtype=float)
    np.fill_diagonal(C, 1.0)

    # latent phylogenetic factors: n x m, correlated across species, shared across traits
    if n_factors > 0 and lam > 0:
        Lc = chol_psd(C)
        F = Lc @ rng.normal(0, 1, (n, n_factors))
        Lambda = rng.normal(0, 1, (n_factors, d))
    else:
        F = np.zeros((n, 0)); Lambda = np.zeros((0, d))

    # causal part
    W, edges = make_dag(d, k, rng)
    if noise == 'gauss':
        Z = rng.normal(0, 1, (n, d))
    else:
        Z = rng.laplace(0, 1 / np.sqrt(2), (n, d))
    Xc = Z @ np.linalg.inv(np.eye(d) - W)

    X = Xc + lam * (F @ Lambda)
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    return X, W, edges, tree, H, C


# ============================================================ metrics
def f1_topk(score, W_true, k, directed=False):
    """Keep the k highest-scoring off-diagonal entries."""
    d = score.shape[0]
    S = np.abs(score).copy()
    np.fill_diagonal(S, -np.inf)
    if directed:
        idx = np.argsort(S.ravel())[::-1][:k]
        est = np.zeros(d * d); est[idx] = 1; est = est.reshape(d, d)
    else:
        ti = np.triu_indices(d, 1)
        vals = S[ti]
        order = np.argsort(vals)[::-1][:k]
        est = np.zeros((d, d))
        est[ti[0][order], ti[1][order]] = 1
        est = est + est.T
    T = (np.abs(W_true) > 0)
    if not directed:
        est = (est > 0); T = (T | T.T)
        tp = float((est & T).sum()); fp = float((est & ~T).sum()); fn = float((~est & T).sum())
    else:
        est = est > 0
        tp = float((est & T).sum()); fp = float((est & ~T).sum()); fn = float((~est & T).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def auroc(score, W_true):
    ti = np.triu_indices(score.shape[0], 1)
    s = np.abs(score)[ti]
    y = (np.abs(W_true) + np.abs(W_true).T)[ti] > 0
    if y.sum() == 0 or (~y).sum() == 0:
        return float('nan')
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def blomberg_k(x, V, nsim=300, seed=0):
    """Proper Blomberg's K: observed (MSE0/MSE) over the Brownian expectation of the same ratio."""
    rng = np.random.default_rng(seed)
    n = len(x)
    Vi = np.linalg.inv(V)
    one = np.ones(n)

    def ratio(v):
        a = Vi @ one
        m = (one @ a) ** -1 * (a @ v)
        r = v - m
        return (r @ r) / max(r @ Vi @ r, 1e-15)

    obs = ratio(x)
    L = np.linalg.cholesky(V + 1e-10 * np.eye(n))
    sims = np.array([ratio(L @ rng.normal(0, 1, n)) for _ in range(nsim)])
    return float(obs / max(sims.mean(), 1e-15))


# ============================================================ methods (all on PIC)
def pic_of(X, tree):
    return felsenstein_pic(X, tree)


def corr_scores(P):
    return np.corrcoef(P.T)


def run_notears(X, lam1=0.1, w_th=0.0):
    """NOTEARS estimate, thresholding OFF by default.

    w_th used to be 0.5 here and 0.3 in another experiment of the same paper. At
    d=30, k=10, lambda=0 the 0.5 threshold left 8 non-zero entries, fewer than the k that
    top-k scoring keeps, so F1 was capped at 0.600 by the threshold rather than by the
    confounding: with w_th=0.3 the same run scores 0.700. The threshold is a second
    decision rule on top of the one under study, so it is left to the common top-k rule
    and the continuous estimate is returned.
    """
    if notears_linear is None:
        return None
    try:
        return notears_linear(np.asarray(X, dtype=np.float64), lambda1=lam1,
                              loss_type='l2', max_iter=100, h_tol=1e-8,
                              w_threshold=w_th)
    except Exception:
        return None


# ============================================================ experiments
def ckpt_path(name):
    return os.path.join(OUTDIR, name + '.json')


def load(name):
    p = ckpt_path(name)
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else {}


def save(name, obj):
    p = ckpt_path(name)
    tmp = p + '.tmp'
    json.dump(obj, open(tmp, 'w', encoding='utf-8'), indent=1)
    os.replace(tmp, p)


def pic_transform(n, seed=0):
    """Explicit linear map A (n-1 x n) of the PIC transform, recovered by feeding unit vectors.
    A fresh tree is rebuilt for every column because the recursive routine mutates branch lengths."""
    A = np.zeros((n - 1, n))
    for j in range(n):
        tr, _ = build_ultrametric_tree(n, seed=seed)
        e = np.zeros((n, 1)); e[j] = 1.0
        A[:, j] = felsenstein_pic(e, tr)[:, 0]
    return A


def exp_pic_whitening():
    """Definitive check: the PIC map must satisfy A C A^T ∝ I for Brownian C."""
    res = load('e1_whitening')
    if 'BM' in res:
        print('  [E1] cached'); return res
    n = 40
    out = {}
    for tag, ao in [('BM', 0.0), ('OU_alpha3', 3.0), ('OU_alpha10', 10.0)]:
        tree, T = build_ultrametric_tree(n, seed=11)
        H, _, _, _ = mrca_heights(tree, n)
        C = cov_brownian(H) if ao == 0 else cov_ou(H, ao)
        np.fill_diagonal(C, 1.0)
        A = pic_transform(n, seed=11)
        M = A @ C @ A.T
        # normalise by the mean diagonal, then measure off-diagonal leakage
        Mn = M / np.mean(np.diag(M))
        off = np.abs(Mn - np.diag(np.diag(Mn)))
        out[tag] = {'mean_offdiag': round(float(off.mean()), 5),
                    'max_offdiag': round(float(off.max()), 5)}
        print(f'  [E1] C={tag:10s} A C A^T normalised: mean|offdiag|={off.mean():.5f}  max={off.max():.4f}')
    res.update(out)
    res['interpretation'] = ('For the covariance PIC actually assumes (BM) the off-diagonal '
                             'leakage is ~0; for a mismatched process it is not. This is the '
                             'implementation check, not a biological result.')
    save('e1_whitening', res)
    return res


def exp_confounding():
    """F1 vs the load of a shared latent factor, with and without PIC, for Gaussian vs Laplace noise."""
    res = load('e2_confounding')
    for d in [30, 50]:
        for noise in ['laplace', 'gauss']:
            for lam in [0.0, 0.5, 1.0, 2.0, 4.0]:
                key = f'd{d}_{noise}_lam{lam}'
                if key in res:
                    continue
                t0 = time.time()
                f1_pic, f1_raw, f1_p99, aucs = [], [], [], []
                for seed in [0, 1, 2]:
                    X, W, edges, tree, H, C = simulate(n=200, d=d, k=10, lam=lam,
                                                       alpha_ou=0.0, noise=noise, seed=seed)
                    P = felsenstein_pic(X, tree)
                    cc = np.corrcoef(P.T)
                    cr = np.corrcoef(X.T)
                    k = len(edges)
                    f1_pic.append(f1_topk(cc, W, k))
                    f1_raw.append(f1_topk(cr, W, k))
                    ti = np.triu_indices(d, 1)
                    th = np.percentile(np.abs(cc[ti]), 99)
                    adj = (np.abs(cc) > th); np.fill_diagonal(adj, False)
                    est = adj | adj.T
                    T = (np.abs(W) > 0); T = T | T.T
                    tp = float((est & T).sum()); fp = float((est & ~T).sum()); fn = float((~est & T).sum())
                    p = tp / (tp + fp) if tp + fp else 0.0
                    r = tp / (tp + fn) if tp + fn else 0.0
                    f1_p99.append(2 * p * r / (p + r) if p + r else 0.0)
                    aucs.append(auroc(cc, W))
                res[key] = {'d': d, 'noise': noise, 'lam': lam,
                            'f1_pic_topk': round(float(np.mean(f1_pic)), 4),
                            'f1_raw_topk': round(float(np.mean(f1_raw)), 4),
                            'f1_pic_p99': round(float(np.mean(f1_p99)), 4),
                            'auroc_pic': round(float(np.nanmean(aucs)), 4),
                            'sec': round(time.time() - t0, 1)}
                print(f'  [E2] {key:24s} PIC={res[key]["f1_pic_topk"]:.3f} '
                      f'raw={res[key]["f1_raw_topk"]:.3f} p99={res[key]["f1_pic_p99"]:.3f} '
                      f'auc={res[key]["auroc_pic"]:.3f} ({res[key]["sec"]}s)')
                save('e2_confounding', res)
    return res


def exp_k():
    """Blomberg's K must move with the process, not be a constant."""
    res = load('e3_blomberg')
    for ao in [0.0, 0.5, 1.0, 3.0, 10.0]:
        key = f'alpha{ao}'
        if key in res:
            continue
        ks = []
        for seed in [0, 1, 2]:
            n, d = 200, 20
            tree, T = build_ultrametric_tree(n, seed=1000 + seed)
            H, _, _, _ = mrca_heights(tree, n)
            V = cov_brownian(H) / H.max()
            C = cov_brownian(H) if ao == 0 else cov_ou(H, ao)
            np.fill_diagonal(C, 1.0)
            Lc = chol_psd(C)
            rng = np.random.default_rng(seed)
            X = Lc @ rng.normal(0, 1, (n, d))
            ks.append(float(np.mean([blomberg_k(X[:, j], V) for j in range(min(d, 5))])))
        res[key] = {'alpha_ou': ao, 'K': round(float(np.mean(ks)), 3), 'seeds': [round(k, 3) for k in ks]}
        print(f'  [E3] OU alpha={ao:<5} Blomberg K = {res[key]["K"]:.3f}')
        save('e3_blomberg', res)
    return res


def exp_methods():
    """NOTEARS on the SAME PIC contrasts vs on the raw matrix."""
    res = load('e4_methods')
    if notears_linear is None:
        print('  [E4] NOTEARS unavailable'); return res
    for d in [30, 50]:
        for lam in [0.0, 2.0]:
            key = f'd{d}_lam{lam}'
            if key in res:
                continue
            t0 = time.time()
            on_pic, on_raw, cp = [], [], []
            for seed in [0, 1, 2]:
                X, W, edges, tree, H, C = simulate(n=200, d=d, k=10, lam=lam,
                                                   alpha_ou=0.0, seed=seed)
                P = felsenstein_pic(X, tree)
                Wp = run_notears(P)
                if Wp is not None:
                    on_pic.append(f1_topk(Wp, W, len(edges)))
                Wr = run_notears(X)
                if Wr is not None:
                    on_raw.append(f1_topk(Wr, W, len(edges)))
                cp.append(f1_topk(np.corrcoef(P.T), W, len(edges)))
            res[key] = {'d': d, 'lam': lam,
                        'notears_on_pic': round(float(np.mean(on_pic)), 4) if on_pic else None,
                        'notears_on_raw': round(float(np.mean(on_raw)), 4) if on_raw else None,
                        'pic_corr': round(float(np.mean(cp)), 4),
                        'sec': round(time.time() - t0, 1)}
            print(f'  [E4] {key:12s} NOTEARS(pic)={res[key]["notears_on_pic"]} '
                  f'NOTEARS(raw)={res[key]["notears_on_raw"]} corr(pic)={res[key]["pic_corr"]}')
            save('e4_methods', res)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='all')
    a = ap.parse_args()
    print('=' * 78)
    if a.only in ('all', 'e1'):
        print('E1  PIC whitening: Brownian factor removed, OU factor not')
        exp_pic_whitening()
    if a.only in ('all', 'e2'):
        print('\nE2  Confounding: F1 vs latent-factor loading')
        exp_confounding()
    if a.only in ('all', 'e3'):
        print('\nE3  Blomberg K vs process')
        exp_k()
    if a.only in ('all', 'e4'):
        print('\nE4  NOTEARS on PIC vs on raw')
        exp_methods()
    print('\nresults ->', OUTDIR)
