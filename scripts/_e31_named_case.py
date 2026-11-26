# -*- coding: utf-8 -*-
"""E31  Named case: can a life-history network that omits body mass be identified?

Design fixed before looking at any number.

  sample   one set of mammal species, complete on all six life-history traits; the SAME species
           are kept in every variant, so only the column set changes.  Changing the sample would
           change n and confound the comparison.
  tree     the same Upham synthesis subtree, Grafen lengths, in every variant.
  variants full (6 traits) / no_mass (drop body mass) / two placebos that drop a different
           column each (longevity, litter size).

Read-outs, all on the contrast correlation matrix R:

  r1        lambda_1 over the mean of the remaining eigenvalues  (the paper's screen)
  top_share squared loading of the largest single variable on the leading eigenvector.
           This is the absorbability of the leading direction: if one variable carries most of
           it, that variable can be deflated and the rest of the graph stays estimable.
  ent       normalised entropy of the squared loadings; 0 = one variable carries it all,
           1 = spread evenly, i.e. no variable can absorb it.
  edge_fit  |corr| between the observed off-diagonal entries and v_i v_j from the leading
           eigenvector.  Near 1 means the whole matrix is one factor's outer product.
  hub       which variable a top-d edge ranking collapses onto, before and after deflation.
"""
import os, io, json, sys, time
import numpy as np

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, S)
import fetch_real as FR
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'tool'))
from pic_screen import tree as PT   # one implementation, shared with the released tool

COLS = FR.BLOCKS['life_history']
MASS = '5-1_AdultBodyMass_g'

VARIANTS = [
    ('full', []),
    ('no_mass', [MASS]),
    ('no_longevity', ['17-1_MaxLongevity_m']),
    ('no_litter', ['15-1_LitterSize']),
]


def load():
    rows = FR.read_pantheria()
    with io.open(FR.MATCH, encoding='utf-8') as f:
        match = json.load(f)
    ok = [(sp, v) for sp, v in rows if match.get(sp) and all(c in v for c, _ in COLS)]
    species = [sp for sp, _ in ok]
    X = np.array([[v[c] for c, _ in COLS] for _, v in ok], float)
    logged = []
    for j in range(X.shape[1]):
        col = X[:, j]
        if col.min() > 0 and col.max() / max(col.min(), 1e-12) > 20:
            X[:, j] = np.log10(col)
            logged.append(COLS[j][0])
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    return species, X, match, logged


def principle_tree(species, match):
    """Prune to the species, collapse the degree-one ancestors that the synthesis tree carries,
    then assign Grafen lengths.  Collapsing is not cosmetic: the node-count convention pays one
    step per ancestor, so redundant nodes stretch some tips more than others.  It also keeps
    this script and the released tool on the same numbers."""
    with io.open(FR.NEWICK_CLADE, encoding='utf-8') as f:
        tmpl = f.read()
    frames = [sp.replace(' ', '_') for sp in species]
    want = set(frames)
    have = set(l.name for l in PT.leaves(PT.parse_newick(tmpl)))
    idx = [i for i, f in enumerate(frames) if f in want and f in have]
    root = PT.prune(PT.parse_newick(tmpl), set(frames[i] for i in idx))
    PT.grafen_lengths(root)
    lo, hi = PT.check_ultrametric(root)
    assert hi - lo < 1e-9, 'tips not contemporaneous: %.6f to %.6f' % (lo, hi)
    C = PT.brownian_cov(root, [frames[i] for i in idx])
    return idx, C


def whiten(X, C):
    L = np.linalg.cholesky(C + 1e-10 * np.eye(C.shape[0]))
    return np.linalg.solve(L, X)


def corr_of(M):
    M = np.asarray(M, float)
    M = M - M.mean(0)
    sd = M.std(0)
    sd[sd < 1e-12] = 1.0
    return np.corrcoef(M / sd, rowvar=False)


def hub(R, k):
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
    return [int(x) for x in deg], int(o[0]), float(deg[o[0]] / max(2.0 * k, 1.0)), int(o[1])


def readouts(R, labels):
    w, v = np.linalg.eigh(R)
    o = np.argsort(w)[::-1]
    w = w[o]
    u = v[:, o[0]]
    u = u * np.sign(u[np.argmax(np.abs(u))])
    sh = u ** 2 / (u ** 2).sum()
    d = len(w)
    so = np.argsort(sh)[::-1]
    iu = np.triu_indices(d, 1)
    pred = np.outer(u, u)[iu]
    obs = (R - np.diag(np.diag(R)))[iu]
    ent = float(-(sh * np.log(sh + 1e-12)).sum() / np.log(d))
    deg, top, share, second = hub(R, d)
    # deflated
    Rd = R - w[0] * np.outer(u, u)
    wd = np.linalg.eigvalsh(Rd)[::-1]
    degd, topd, shared, secondd = hub(Rd, d)
    return dict(
        n=int(R.shape[0]), d=int(d),
        w=[round(float(x), 4) for x in w],
        r1=round(float(w[0] / w[1:].mean()), 4),
        r2=round(float(w[1] / w[2:].mean()), 4) if d > 2 else None,
        top=labels[so[0]], top_share=round(float(sh[so[0]]), 4),
        second=labels[so[1]], second_share=round(float(sh[so[1]]), 4),
        entropy=round(ent, 4),
        edge_fit=round(float(np.corrcoef(np.abs(pred), np.abs(obs))[0, 1]), 4),
        shares={labels[j]: round(float(sh[j]), 4) for j in range(d)},
        deg=deg, hub=labels[top], hub_share=round(share, 4), hub2=labels[second],
        defl_r1=round(float(wd[0] / wd[1:].mean()), 4),
        defl_hub=labels[topd], defl_share=round(shared, 4), defl_deg=degd,
    )


def main():
    t0 = time.time()
    species, X, match, logged = load()
    print('complete-case species on %d traits: %d   (logged: %s)' % (len(COLS), len(species), logged))
    idx, C = principle_tree(species, match)
    X = X[idx]
    species = [species[i] for i in idx]
    frames = [sp.replace(' ', '_') for sp in species]
    print('in tree: %d   (%.0fs)' % (len(species), time.time() - t0))

    P = whiten(X, C)
    base = corr_of(P)
    print('contrast r1 on full 6: %.3f' % (np.linalg.eigvalsh(base)[-1] /
                                           np.linalg.eigvalsh(base)[:-1].mean()))

    out = {}
    for name, drop in VARIANTS:
        keep = [j for j, (c, _) in enumerate(COLS) if c not in drop]
        labels = [COLS[j][1] for j in keep]
        R = corr_of(P[:, keep])
        r = readouts(R, labels)
        out[name] = r
        print('\n=== %s   d=%d ===' % (name, r['d']))
        print('  r1=%6.2f  r2=%6.2f   top=%s (share %.2f)  entropy=%.3f  edge_fit=%.3f'
              % (r['r1'], r['r2'], r['top'], r['top_share'], r['entropy'], r['edge_fit']))
        print('  shares ' + '  '.join('%s=%.2f' % (k, v) for k, v in sorted(
            r['shares'].items(), key=lambda kv: -kv[1])))
        print('  hub (plain)   %s  share=%.2f  deg=%s' % (r['hub'], r['hub_share'], r['deg']))
        print('  after deflate r1=%.2f  hub=%s share=%.2f deg=%s'
              % (r['defl_r1'], r['defl_hub'], r['defl_share'], r['defl_deg']))
        print('  eigenvalues ' + ' '.join('%.2f' % x for x in r['w']))

    dst = os.path.join(os.path.dirname(S), 'results', 'e31_named_case.json')
    with io.open(dst, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('\nsaved', dst, '  total %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
