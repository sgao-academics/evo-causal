# -*- coding: utf-8 -*-
"""Self-check.  Run as:  python -m pic_screen

Each check prints PASS or FAIL.  The point is that a reader can confirm the two behaviour-
changing claims (Grafen lengths make the tips contemporaneous; the screen separates a shared
axis from a single offending variable) without touching the manuscript's code.
"""
import sys
import numpy as np

from . import tree as T
from .screen import screen

UNBALANCED = '(((A:1,B:1):1,(C:1,D:1):1):1,(E:1,F:1):1,G:1);'
BALANCED = '(((t0,t1),(t2,t3)),((t4,t5),(t6,t7)),((t8,t9),(t10,t11)),((t12,t13),(t14,t15)),' \
           '((t16,t17),(t18,t19)),((t20,t21),(t22,t23)),((t24,t25),(t26,t27)),' \
           '((t28,t29),(t30,t31)));'

TRAITS = ['body mass', 'gestation', 'longevity', 'litter size', 'weaning', 'maturity']


def simulate(d=6, lam=0.0, seed=0, n_species=32):
    """Brownian traits across the tips of a balanced 32-tip tree, plus an optional latent axis.

    Rows are species and columns are traits, which is the shape the screen expects: the tree
    fixes the (species x species) covariance, a d-column matrix of values is drawn from it.
    """
    rng = np.random.default_rng(seed)
    root = T.parse_newick(BALANCED)
    tips = sorted(l.name for l in T.leaves(root))[:n_species]
    root = T.prune(T.parse_newick(BALANCED), set(tips))
    T.grafen_lengths(root)
    C = T.brownian_cov(root, tips)
    Lc = np.linalg.cholesky(C + 1e-10 * np.eye(len(tips)))
    X = Lc @ rng.normal(size=(len(tips), d))
    if lam:
        X = X + lam * rng.normal(size=(len(tips), 1))
    return X, BALANCED, tips


def main():
    ok = True

    def chk(name, cond, detail=''):
        nonlocal ok
        ok = ok and bool(cond)
        print('%-4s %-54s %s' % ('PASS' if cond else 'FAIL', name, detail))

    # 1 parsing
    r = T.parse_newick('((A:1,B:1):2,C:3);')
    chk('newick parses to three tips', len(T.leaves(r)) == 3,
        str(sorted(l.name for l in T.leaves(r))))

    # 2 grafen lengths are ultrametric, on a tree whose tips sit at different depths
    r = T.prune(T.parse_newick(UNBALANCED), set('ABCDEFG'))
    T.grafen_lengths(r)
    lo, hi = T.check_ultrametric(r)
    chk('Grafen lengths place every tip at distance 1',
        abs(lo - 1.0) < 1e-9 and abs(hi - 1.0) < 1e-9, '%.6f to %.6f' % (lo, hi))

    # 3 the depth-form slip is caught on that same tree
    r = T.prune(T.parse_newick(UNBALANCED), set('ABCDEFG'))
    h = T._heights(r)
    H = h[id(r)]
    for v in T.postorder(r):
        v.length = 0.0 if v.parent is None else (H - h[id(v)]) / float(H)
    lo, hi = T.check_ultrametric(r)
    chk('using depth as edge length breaks contemporaneity', hi - lo > 0.2,
        '%.2f to %.2f' % (lo, hi))

    # 4 screen refuses an unknown branch-length convention
    lab32 = sorted(l.name for l in T.leaves(T.parse_newick(BALANCED)))
    try:
        screen(np.random.default_rng(0).normal(size=(32, 6)), BALANCED, labels=lab32,
               tree='nonsense')
        raised = False
    except ValueError:
        raised = True
    chk('screen rejects an unknown branch-length convention', raised)

    # 5 no factor -> does not fire
    X, nw, sp = simulate(6, 0.0, seed=1)
    a = screen(X, nw, species=sp, labels=TRAITS, nsim=2000)
    chk('independent traits do not fire', a.p > 0.05, 'r1=%.2f p=%.2f' % (a.r1, a.p))

    # 6 a shared axis spread over the traits fires and is not repaired
    X2, nw2, sp2 = simulate(6, 3.0, seed=2)
    b = screen(X2, nw2, species=sp2, labels=TRAITS, nsim=2000)
    chk('a shared axis fires', b.p <= 0.05, 'r1=%.2f p=%.3g' % (b.r1, b.p))
    chk('the axis is shared, not owned by one trait', b.entropy > 0.85,
        'entropy=%.3f top=%s %.2f' % (b.entropy, b.top, b.top_share))

    # 7 one variable carrying the axis is reported as such: rebuild column 0 as a proxy for the
    # common part of the others, which is what "the offending variable" looks like in practice
    X3, nw3, sp3 = simulate(6, 0.0, seed=3)
    rng3 = np.random.default_rng(4)
    f = X3[:, 1:].mean(1)
    X3[:, 0] = 3.0 * f + 0.3 * rng3.normal(size=X3.shape[0])
    c = screen(X3, nw3, species=sp3, labels=TRAITS, nsim=2000)
    chk('a single dominant variable is named', c.top == TRAITS[0],
        'top=%s %.2f' % (c.top, c.top_share))

    print()
    print('ALL PASS' if ok else 'SOME CHECKS FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
