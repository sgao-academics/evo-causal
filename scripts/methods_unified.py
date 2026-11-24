# -*- coding: utf-8 -*-
"""One implementation of every method, one decision rule, one significance level.

Why this module exists
----------------------
An audit of the first revision found four defects that would each have been fatal if a
reviewer had found them first:

1.  FCI node labels in causal-learn are X1..Xd, i.e. ONE-based. The experiment code did
        i = int(name.lstrip('X'))
    and used i directly as a row index of a zero-based matrix, so every FCI edge was
    shifted by one position and any edge touching Xd was dropped by the i >= d guard.
    DONE HERE: labels are converted with a single helper that subtracts one.

2.  NOTEARS was run with w_threshold = 0.5, which (at d=30, k=10, lambda=0) left only 8
    non-zero entries. Because scoring keeps the top k entries, an estimate with fewer
    than k non-zeros cannot reach F1 = 1 however good the ranking is. The same paper
    also used 0.3 in one experiment and 0.5 in another.
    DONE HERE: every method returns a dense score, and a single top-k rule is applied.

3.  PC was run at alpha = 0.01 while FCI used causal-learn's default 0.05.
    DONE HERE: one constant, ALPHA, is used for both.

4.  PC was a hand-written skeleton phase capped at conditioning depth 3.
    DONE HERE: PC comes from causal-learn, the same package as FCI, at full depth.

Scoring
-------
Every method produces an undirected score matrix; `topk_adj` turns it into the k-edge
graph that is compared with the truth. Undirected scoring is deliberate: orientation is a
separate problem and mixing it in would confound the comparison between a method that
returns a DAG and one that returns a pattern.
"""
import numpy as np
from scipy import stats

ALPHA = 0.05          # one significance level for every constraint-based method


# ------------------------------------------------------------------ scoring
def topk_adj(score, k):
    """Undirected adjacency obtained by keeping the k largest off-diagonal |score|."""
    S = np.abs(np.asarray(score, dtype=float)).copy()
    d = S.shape[0]
    np.fill_diagonal(S, -np.inf)
    ti = np.triu_indices(d, 1)
    vals = S[ti]
    k = int(min(k, vals.size))
    keep = np.argsort(vals)[::-1][:k]
    adj = np.zeros((d, d), bool)
    adj[ti[0][keep], ti[1][keep]] = True
    return adj | adj.T


def symmetrise(adj):
    return (np.asarray(adj, bool) | np.asarray(adj, bool).T)


def f1_undirected(est, W):
    """F1 between an undirected estimate and the undirected skeleton of W."""
    est = symmetrise(est)
    T = (np.abs(W) > 0)
    T = T | T.T
    tp = float((est & T).sum())
    fp = float((est & ~T).sum())
    fn = float((~est & T).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


# ------------------------------------------------------------------ NOTEARS
def notears_score(X, lambda1=0.1, max_iter=100, h_tol=1e-8):
    """NOTEARS on X, returned as a dense score.

    w_threshold is 0.0 on purpose. Zheng et al. threshold the estimate to report a graph,
    but a threshold is a second decision rule on top of the one under study, and it caps
    the top-k score. The continuous estimate is the method's output; the rule is common
    to every method here.
    """
    import os, sys

    from notears_impl import notears_linear
    W = notears_linear(np.asarray(X, dtype=np.float64), lambda1=lambda1,
                       loss_type='l2', max_iter=max_iter, h_tol=h_tol,
                       w_threshold=0.0)
    return np.abs(W)


def notears_h(W):
    """Acyclicity residual h(W) for reporting; 0 means the estimate is a DAG."""
    import numpy as np
    import scipy.linalg as slin
    W = np.asarray(W, float)
    return float(np.trace(slin.expm(W * W)) - W.shape[0])


# ------------------------------------------------------------------ PC
def pc_skeleton(P, alpha=ALPHA):
    """Skeleton from causal-learn's PC. Returns an undirected bool adjacency.

    causal-learn labels nodes X1..Xd; `_label_to_index` is the only place where that
    convention is translated, so the off-by-one cannot reappear in a second file.
    """
    from causallearn.search.ConstraintBased.PC import pc
    P = np.asarray(P, dtype=float)
    cg = pc(P, alpha=alpha, indep_test='fisherz', stable=True, show_progress=False)
    G = cg.G
    d = P.shape[1]
    adj = np.zeros((d, d), bool)
    for e in G.get_graph_edges():
        i = _label_to_index(e.get_node1().get_name(), d)
        j = _label_to_index(e.get_node2().get_name(), d)
        if i is None or j is None:
            continue
        adj[i, j] = adj[j, i] = True
    return symmetrise(adj)


# ------------------------------------------------------------------ FCI
def _label_to_index(name, d):
    """causal-learn names nodes X1..Xd (one-based). Return the zero-based column index.

    node.get_name() returns something like 'X1'. Some versions return an int. Both are
    handled, and an out-of-range label returns None rather than being silently clamped,
    because a clamped label is what produced the original error.
    """
    s = str(name).strip()
    if s.startswith('X'):
        s = s[1:]
    try:
        idx = int(s) - 1
    except ValueError:
        return None
    return idx if 0 <= idx < d else None


def fci_pattern(P, alpha=ALPHA):
    """FCI from causal-learn. Returns (undirected skeleton, fraction of pair marks).

    The fraction counts adjacencies whose endpoints carry a circle mark or that are
    arrow-arrow: those are the marks FCI uses to say a confounder may be present.
    """
    import contextlib, io as _io
    from causallearn.search.ConstraintBased.FCI import fci
    P = np.asarray(P, dtype=float)
    d = P.shape[1]
    # causal-learn prints the recovered edges even with verbose=False; silence it so the
    # run log stays a record of our own output.
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        G, edges = fci(P, alpha=alpha, verbose=False, show_progress=False)
    adj = np.zeros((d, d), bool)
    marked = 0
    total = 0
    for e in edges:
        i = _label_to_index(e.get_node1().get_name(), d)
        j = _label_to_index(e.get_node2().get_name(), d)
        if i is None or j is None or i == j:
            continue
        adj[i, j] = adj[j, i] = True
        total += 1
        s1, s2 = str(e.get_endpoint1()), str(e.get_endpoint2())
        if ('CIRCLE' in s1) or ('CIRCLE' in s2) or ('ARROW' in s1 and 'ARROW' in s2):
            marked += 1
    frac = marked / total if total else 0.0
    return symmetrise(adj), frac
