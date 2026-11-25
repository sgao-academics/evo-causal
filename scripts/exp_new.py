# -*- coding: utf-8 -*-
"""Second-round experiments for the revision. Every block checkpoints into results/.

E14  dissociation      phylogenetic signal raised without a confounder, and a confounder
                       raised without phylogenetic signal, so that K and the mechanism are
                       separated instead of being the same variable.
E15  screen            leading/rest eigenvalue ratio with a null calibrated under the true
                       DAG, reported as size and power.
E16  deflation         score-level removal of the leading eigencomponent, plain vs gated.
E17  ppa order         path analysis with the true topological order removed.
E18  k vs n            K at fixed lambda as the number of species changes.
E19  precision/recall  the same grids as E5/E11, rerun to record the confusion matrix.
E20  ges               hill-climbing BIC, a fourth search paradigm.
E22  seed spread       ten seeds on four cells, to state the uncertainty of the thresholds.

Run with --only E14,E15,... ; every block is skipped if its file is complete.
"""
import os, sys, json, time, argparse
import numpy as np

S = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(os.path.dirname(S), 'scripts')
sys.path.insert(0, R)
sys.path.insert(0, S)

from run_revision import (build_ultrametric_tree, mrca_heights, cov_brownian, chol_psd,
                          make_dag, simulate, blomberg_k, pic_of, f1_topk, auroc,
                          corr_scores, run_notears)
from run_revision2 import _gls_bic
from methods_unified import topk_adj, symmetrise

OUT = os.path.join(os.path.dirname(S), 'results')
os.makedirs(OUT, exist_ok=True)

D30, D50, NSPEC, KED = 30, 50, 200, 10


# ------------------------------------------------------------------ helpers
def path(name):
    return os.path.join(OUT, name + '.json')


def load(name):
    return json.load(open(path(name), encoding='utf-8')) if os.path.exists(path(name)) else {}


def save(name, obj):
    tmp = path(name) + '.tmp'
    json.dump(obj, open(tmp, 'w', encoding='utf-8'), indent=1)
    os.replace(tmp, path(name))


def log(*a):
    print(*a, flush=True)


def prf(est, W):
    """precision, recall, F1 of an undirected estimate against the skeleton of W."""
    est = symmetrise(np.asarray(est, bool))
    T = (np.abs(W) > 0); T = T | T.T
    tp = float((est & T).sum()); fp = float((est & ~T).sum()); fn = float((~est & T).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def gen(n, d, k, lam, seed, latent='tree', trait_signal=0.0, noise='laplace'):
    """run_revision.simulate plus two knobs.

    latent        'tree' (factors evolve on the tree, the shipped behaviour) or 'iid'
                  (a confounder with no phylogenetic structure at all)
    trait_signal  weight on an INDEPENDENT Brownian draw per trait: raises the marginal
                  phylogenetic signal without creating cross-trait covariance.
    """
    rng = np.random.default_rng(seed)
    tree, _ = build_ultrametric_tree(n, seed=1000 + seed)
    H, _, _, _ = mrca_heights(tree, n)
    C = cov_brownian(H).astype(float)
    np.fill_diagonal(C, 1.0)
    Lc = chol_psd(C)
    if lam > 0:
        F = (Lc @ rng.normal(0, 1, (n, 1))) if latent == 'tree' else rng.normal(0, 1, (n, 1))
        Lam = rng.normal(0, 1, (1, d))
    else:
        F = np.zeros((n, 1)); Lam = np.zeros((1, d))
    W, edges = make_dag(d, k, rng)
    Z = rng.normal(0, 1, (n, d)) if noise == 'gauss' else rng.laplace(0, 1 / np.sqrt(2), (n, d))
    X = Z @ np.linalg.inv(np.eye(d) - W) + lam * (F @ Lam)
    if trait_signal > 0:
        X = X + trait_signal * (Lc @ rng.normal(0, 1, (n, d)))
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    return X, W, tree, C


def kstat(X, C, ntr=5, seed=0):
    return float(np.mean([blomberg_k(X[:, j], C, nsim=200, seed=seed + j) for j in range(ntr)]))


def lead_ratio(P):
    """Leading eigenvalue of the contrast correlation matrix over the mean of the rest."""
    R = corr_scores(P)
    w = np.linalg.eigvalsh(R)
    return float(w[-1] / max(w[:-1].mean(), 1e-12)), float(w[-1])


def deflate_score(R):
    w, v = np.linalg.eigh(R)
    i = int(np.argmax(w))
    return R - w[i] * np.outer(v[:, i], v[:, i])


# ------------------------------------------------------------------ E14
def e14():
    name = 'e14_dissociation'
    res = load(name)
    arms = [('signal', [0.0, 1.0, 2.0, 3.0, 4.0, 6.0]),
            ('iid', [1.0, 2.0, 3.0, 4.0, 6.0]),
            ('tree', [1.0, 2.0, 3.0, 4.0, 6.0])]
    for d in (D30, D50):
        for arm, levels in arms:
            for lev in levels:
                key = '%s_d%d_lv%.1f' % (arm, d, lev)
                if key in res:
                    continue
                t0 = time.time()
                K, f1p, f1r, au, P_, Rc, rr = [], [], [], [], [], [], []
                for sd in range(10):
                    if arm == 'signal':
                        X, W, tree, C = gen(NSPEC, d, KED, 0.0, sd, trait_signal=lev)
                    elif arm == 'iid':
                        X, W, tree, C = gen(NSPEC, d, KED, lev, sd, latent='iid')
                    else:
                        X, W, tree, C = gen(NSPEC, d, KED, lev, sd, latent='tree')
                    K.append(kstat(X, C, ntr=3, seed=sd))
                    Pt = pic_of(X, tree)
                    Rc.append(corr_scores(Pt))
                    Rt = corr_scores(X)
                    p, r_, f = prf(topk_adj(Rc[-1], KED), W)
                    p2, r2, f2 = prf(topk_adj(Rt, KED), W)
                    f1p.append(f); f1r.append(f2); P_.append(p); Rc_ = r_
                    au.append(auroc(Rc[-1], W)); rr.append(r_)
                res[key] = dict(arm=arm, d=d, level=lev, seeds=10,
                                K=round(float(np.mean(K)), 4), K_sd=round(float(np.std(K)), 4),
                                f1_pic=round(float(np.mean(f1p)), 4),
                                f1_raw=round(float(np.mean(f1r)), 4),
                                precision=round(float(np.mean(P_)), 4),
                                recall=round(float(np.mean(rr)), 4),
                                auroc=round(float(np.mean(au)), 4),
                                sec=round(time.time() - t0, 1))
                save(name, res)
                r = res[key]
                log('  [E14] %-18s K=%.3f  F1_pic=%.3f  F1_raw=%.3f  AUROC=%.3f  (%.0fs)'
                    % (key, r['K'], r['f1_pic'], r['f1_raw'], r['auroc'], r['sec']))
    return res


# ------------------------------------------------------------------ E15
def e15():
    """Screen: null calibrated by simulating from the TRUE DAG with iid errors."""
    name = 'e15_screen'
    res = load(name)
    for d in (D30, D50):
        # null: 200 fresh datasets with no confounder at all
        if 'null_d%d' % d not in res:
            t0 = time.time()
            rs = []
            for sd in range(200):
                X, W, tree, C = gen(NSPEC, d, KED, 0.0, 5000 + sd)
                P = pic_of(X, tree)
                rs.append(lead_ratio(P)[0])
            thr = float(np.quantile(rs, 0.95))
            res['null_d%d' % d] = dict(n=200, mean=round(float(np.mean(rs)), 3),
                                       sd=round(float(np.std(rs)), 3),
                                       q05=round(float(np.quantile(rs, 0.05)), 3),
                                       q50=round(float(np.median(rs)), 3),
                                       q95=round(thr, 3), thr=round(thr, 3),
                                       sec=round(time.time() - t0, 1))
            save(name, res)
            log('  [E15] null d=%d  mean=%.3f sd=%.3f  thr95=%.3f  (%.0fs)'
                % (d, res['null_d%d' % d]['mean'], res['null_d%d' % d]['sd'], thr,
                   res['null_d%d' % d]['sec']))
        thr = res['null_d%d' % d]['thr']
        cells = [('none', 0.0, 0.0), ('iid', 1.0, None), ('iid', 2.0, None), ('iid', 4.0, None),
                 ('tree', 0.5, None), ('tree', 1.0, None), ('tree', 2.0, None),
                 ('tree', 3.0, None), ('tree', 4.0, None), ('tree', 6.0, None)]
        for lat, lev, s in cells:
            key = '%s_d%d_lv%.1f' % (lat, d, lev)
            if key in res:
                continue
            rs = []
            for sd in range(10):
                if lat == 'none':
                    X, W, tree, C = gen(NSPEC, d, KED, 0.0, sd)
                else:
                    X, W, tree, C = gen(NSPEC, d, KED, lev, sd, latent=lat)
                P = pic_of(X, tree)
                rs.append(lead_ratio(P)[0])
            res[key] = dict(latent=lat, d=d, level=lev, seeds=10,
                            r=round(float(np.mean(rs)), 3),
                            fired=int(sum(1 for x in rs if x > thr)),
                            thr=thr)
            save(name, res)
            log('  [E15] %-14s r=%.3f  fired %d/10' % (key, res[key]['r'], res[key]['fired']))
    return res


# ------------------------------------------------------------------ E16
def e16():
    name = 'e16_deflate'
    res = load(name)
    for d in (D30, D50):
        for lam in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
            key = 'd%d_lam%.1f' % (d, lam)
            if key in res:
                continue
            t0 = time.time()
            plain, defl, gate, au_p, au_d = [], [], [], [], []
            for sd in range(10):
                X, W, tree, C = gen(NSPEC, d, KED, lam, sd)
                P = pic_of(X, tree)
                R = corr_scores(P)
                plain.append(f1_topk(R, W, KED))
                Rd = deflate_score(R)
                defl.append(f1_topk(Rd, W, KED))
                gate.append(f1_topk(Rd, W, KED) if lam > 0 else f1_topk(R, W, KED))
                au_p.append(auroc(R, W)); au_d.append(auroc(Rd, W))
            res[key] = dict(d=d, lam=lam, seeds=10,
                            plain=round(float(np.mean(plain)), 3),
                            plain_sd=round(float(np.std(plain)), 3),
                            deflate=round(float(np.mean(defl)), 3),
                            deflate_sd=round(float(np.std(defl)), 3),
                            auroc_plain=round(float(np.mean(au_p)), 4),
                            auroc_deflate=round(float(np.mean(au_d)), 4),
                            sec=round(time.time() - t0, 1))
            save(name, res)
            r = res[key]
            log('  [E16] %-11s plain=%.3f(%.3f)  deflate=%.3f(%.3f)  AUROC %.3f->%.3f'
                % (key, r['plain'], r['plain_sd'], r['deflate'], r['deflate_sd'],
                   r['auroc_plain'], r['auroc_deflate']))
    # NOTEARS on the deflated matrix, two levels only
    for lam in (2.0, 3.0):
        key = 'notears_d30_lam%.1f' % lam
        if key in res:
            continue
        a, b = [], []
        for sd in range(3):
            X, W, tree, C = gen(NSPEC, D30, KED, lam, sd)
            P = pic_of(X, tree)
            E1 = run_notears(P)
            Z = P - P.mean(0)
            U, s, Vt = np.linalg.svd(Z, full_matrices=False)
            P2 = Z - np.outer(U[:, 0] * s[0], Vt[0])
            E2 = run_notears(P2)
            if E1 is None or E2 is None:
                continue
            a.append(f1_topk(np.abs(E1), W, KED)); b.append(f1_topk(np.abs(E2), W, KED))
        res[key] = dict(lam=lam, seeds=len(a), plain=round(float(np.mean(a)), 3),
                        deflated=round(float(np.mean(b)), 3))
        save(name, res)
        log('  [E16] NOTEARS lam=%.1f plain=%.3f deflated=%.3f'
            % (lam, res[key]['plain'], res[key]['deflated']))
    return res


# ------------------------------------------------------------------ E17
def all_dags(d):
    import itertools
    seen = {}
    for perm in itertools.permutations(range(d)):
        pairs = [(perm[i], perm[j]) for i in range(d) for j in range(i + 1, d)]
        for mask in range(1 << len(pairs)):
            A = np.zeros((d, d))
            for bidx, (i, j) in enumerate(pairs):
                if mask >> bidx & 1:
                    A[i, j] = 1.0
            seen.setdefault(A.tobytes(), A)
    return list(seen.values())


def e17():
    name = 'e17_ppa_order'
    res = load(name)
    d, nn, k = 5, 120, 6
    if 'dags' not in res:
        dag = all_dags(d)
        small = [A for A in dag if A.sum() <= 3]
        res['dags'] = dict(full=len(dag), restricted=len(small))
        res['_full'] = [A.tolist() for A in dag]
        res['_small'] = [A.tolist() for A in small]
        save(name, res)
        log('  [E17] candidate sets: full=%d restricted=%d' % (len(dag), len(small)))
    full = [np.array(A) for A in res['_full']]
    small = [np.array(A) for A in res['_small']]
    for lam in (0.0, 1.0, 2.0):
        for regime, cand in (('full', full), ('restricted', small)):
            key = 'lam%.1f_%s' % (lam, regime)
            if key in res:
                continue
            t0 = time.time()
            f1s, ex, pic = [], [], []
            for sd in range(5):
                X, W, edges, tree, H, C = simulate(n=nn, d=d, k=k, lam=lam,
                                                   alpha_ou=0.0, noise='laplace', seed=sd)
                V = cov_brownian(H); Vinv = np.linalg.inv(V)
                ld = float(np.linalg.slogdet(V)[1])
                sc = np.array([_gls_bic(X, A, Vinv, ld, nn) for A in cand])
                best = cand[int(np.argmin(sc))]
                f1s.append(prf((best != 0) | (best.T != 0), W)[2])
                ex.append(float(np.allclose(best, (np.abs(W) > 0).astype(float))))
                P = pic_of(X, tree); cc = corr_scores(P)
                pic.append(prf(topk_adj(cc, len(edges)), W)[2])
            res[key] = dict(lam=lam, regime=regime, seeds=5, n_models=len(cand),
                            ppa_f1=round(float(np.mean(f1s)), 4),
                            ppa_exact=round(float(np.mean(ex)), 3),
                            pic_f1=round(float(np.mean(pic)), 4),
                            sec=round(time.time() - t0, 1))
            save(name, res)
            r = res[key]
            log('  [E17] %-18s PPA=%.3f exact=%.2f  PIC=%.3f  (%ds)'
                % (key, r['ppa_f1'], r['ppa_exact'], r['pic_f1'], r['sec']))
    for kk in ('_full', '_small'):
        res.pop(kk, None)
    save(name, res)
    return res


# ------------------------------------------------------------------ E18
def e18():
    name = 'e18_k_vs_n'
    res = load(name)
    for lam in (1.0, 2.0):
        for n in (100, 200, 500, 1000, 2000):
            key = 'lam%.1f_n%d' % (lam, n)
            if key in res:
                continue
            t0 = time.time()
            Ks, f1s = [], []
            for sd in range(10):
                X, W, tree, C = gen(n, D30, KED, lam, sd)
                Ks.append(kstat(X, C, ntr=3, seed=sd))
                f1s.append(f1_topk(corr_scores(pic_of(X, tree)), W, KED))
            res[key] = dict(lam=lam, n=n, seeds=10,
                            K=round(float(np.mean(Ks)), 4), K_sd=round(float(np.std(Ks)), 4),
                            f1=round(float(np.mean(f1s)), 4),
                            sec=round(time.time() - t0, 1))
            save(name, res)
            r = res[key]
            log('  [E18] lam=%.1f n=%-5d K=%.3f(%.3f)  F1=%.3f' %
                (lam, n, r['K'], r['K_sd'], r['f1']))
    return res


# ------------------------------------------------------------------ E20
class BICScore:
    """Decomposable Gaussian BIC for hill-climbing search."""
    def __init__(self, P):
        self.P = P; self.n, self.d = P.shape; self.cache = {}

    def local(self, j, pa):
        key = (j, pa)
        if key in self.cache:
            return self.cache[key]
        y = self.P[:, j]
        if pa:
            Z = self.P[:, list(pa)]
            beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
            r = y - Z @ beta
        else:
            r = y - y.mean()
        rss = float(r @ r) + 1e-12
        s = self.n * np.log(rss / self.n) + np.log(self.n) * (len(pa) + 1)
        self.cache[key] = s
        return s

    def total(self, W):
        return sum(self.local(j, tuple(np.where(W[:, j] != 0)[0])) for j in range(self.d))


def acyclic(W):
    d = W.shape[0]
    A = (W != 0).astype(float)
    return np.allclose(np.linalg.matrix_power(A, d), 0)


def hc_bic(P, max_iter=200):
    sc = BICScore(P)
    d = P.shape[1]
    W = np.zeros((d, d))
    cur = sc.total(W)
    for _ in range(max_iter):
        best = (cur, None)
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                Wn = W.copy()
                Wn[i, j] = 0.0 if W[i, j] else 1.0
                if not acyclic(Wn):
                    continue
                s = sc.total(Wn)
                if s < best[0] - 1e-8:
                    best = (s, Wn)
        if best[1] is None:
            break
        cur, W = best
    return W


def e20():
    name = 'e20_ges'
    res = load(name)
    for lam in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        key = 'd30_lam%.1f' % lam
        if key in res:
            continue
        t0 = time.time()
        f1s, sz = [], []
        for sd in range(5):
            X, W, tree, C = gen(NSPEC, D30, KED, lam, sd)
            P = pic_of(X, tree)
            Wh = hc_bic(P)
            est = (Wh != 0) | (Wh.T != 0)
            f1s.append(prf(est, W)[2]); sz.append(int(est.sum() // 2))
        res[key] = dict(lam=lam, seeds=5, f1=round(float(np.mean(f1s)), 3),
                        sd=round(float(np.std(f1s)), 3),
                        skeleton=round(float(np.mean(sz)), 1),
                        sec=round(time.time() - t0, 1))
        save(name, res)
        log('  [E20] %-11s HC-BIC F1=%.3f(%.3f) edges=%.1f  (%ds)'
            % (key, res[key]['f1'], res[key]['sd'], res[key]['skeleton'], res[key]['sec']))
    return res


# ------------------------------------------------------------------ E19
def e19():
    """Rerun the E5 and E11 grids at their published seeds to record precision/recall."""
    name = 'e19_precision_recall'
    res = load(name)
    for d in (D30, D50):
        for lam in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
            key = 'dose_d%d_lam%.2f' % (d, lam)
            if key in res:
                continue
            pp, rr, ff, ss = [], [], [], []
            for sd in (0, 1, 2):
                X, W, _e, tree, _H, C = simulate(n=NSPEC, d=d, k=KED, lam=lam, noise='laplace', seed=sd)
                R = corr_scores(pic_of(X, tree))
                adj = topk_adj(R, KED)
                p, r_, f = prf(adj, W)
                pp.append(p); rr.append(r_); ff.append(f); ss.append(int(adj.sum() // 2))
            res[key] = dict(d=d, lam=lam, precision=round(float(np.mean(pp)), 4),
                            recall=round(float(np.mean(rr)), 4), f1=round(float(np.mean(ff)), 4),
                            skeleton=round(float(np.mean(ss)), 1))
            save(name, res)
            log('  [E19] %-17s P=%.3f R=%.3f F1=%.3f edges=%.1f' %
                (key, res[key]['precision'], res[key]['recall'], res[key]['f1'],
                 res[key]['skeleton']))
    # the causal-learn methods on the E11 grid
    from methods_unified import pc_skeleton, fci_pattern
    for lam in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        for mname in ('pc', 'fci'):
            key = 'fam_d30_lam%.1f_%s' % (lam, mname)
            if key in res:
                continue
            pp, rr, ff, ss = [], [], [], []
            for sd in (0, 1, 2):
                X, W, _e, tree, _H, C = simulate(n=NSPEC, d=D30, k=KED, lam=lam, noise='laplace', seed=sd)
                P = pic_of(X, tree)
                adj = pc_skeleton(P) if mname == 'pc' else fci_pattern(P)[0]
                p, r_, f = prf(adj, W)
                pp.append(p); rr.append(r_); ff.append(f); ss.append(int(adj.sum() // 2))
            res[key] = dict(lam=lam, method=mname, precision=round(float(np.mean(pp)), 4),
                            recall=round(float(np.mean(rr)), 4), f1=round(float(np.mean(ff)), 4),
                            skeleton=round(float(np.mean(ss)), 1))
            save(name, res)
            log('  [E19] %-24s P=%.3f R=%.3f F1=%.3f edges=%.1f' %
                (key, res[key]['precision'], res[key]['recall'], res[key]['f1'],
                 res[key]['skeleton']))
    return res


# ------------------------------------------------------------------ E22
def e22():
    name = 'e22_seed_spread'
    res = load(name)
    for lam in (0.0, 1.0, 2.0, 3.0, 4.0):
        key = 'd30_lam%.1f' % lam
        if key in res:
            continue
        t0 = time.time()
        cp, nt = [], []
        for sd in range(10):
            X, W, _e, tree, _H, C = simulate(n=NSPEC, d=D30, k=KED, lam=lam, noise='laplace', seed=sd)
            P = pic_of(X, tree)
            cp.append(f1_topk(corr_scores(P), W, KED))
            E = run_notears(P)
            if E is not None:
                nt.append(f1_topk(np.abs(E), W, KED))
        res[key] = dict(lam=lam, seeds=10,
                        corr_mean=round(float(np.mean(cp)), 3),
                        corr_sd=round(float(np.std(cp, ddof=1)), 3),
                        corr_ci=round(float(1.96 * np.std(cp, ddof=1) / np.sqrt(len(cp))), 3),
                        notears_mean=round(float(np.mean(nt)), 3) if nt else None,
                        notears_sd=round(float(np.std(nt, ddof=1)), 3) if nt else None,
                        notears_ci=round(float(1.96 * np.std(nt, ddof=1) / np.sqrt(len(nt))), 3) if nt else None,
                        sec=round(time.time() - t0, 1))
        save(name, res)
        r = res[key]
        log('  [E22] lam=%.1f corr=%.3f+-%.3f  notears=%s  (%ds)'
            % (lam, r['corr_mean'], r['corr_ci'], r['notears_mean'], r['sec']))
    return res


BLOCKS = dict(E14=e14, E15=e15, E16=e16, E17=e17, E18=e18, E19=e19, E20=e20, E22=e22)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='E14,E15,E16,E17,E18,E19,E20,E22')
    a = ap.parse_args()
    t0 = time.time()
    for k in a.only.split(','):
        k = k.strip()
        if k in BLOCKS:
            log('=== %s ===' % k)
            BLOCKS[k]()
    log('TOTAL %.1f min' % ((time.time() - t0) / 60))
