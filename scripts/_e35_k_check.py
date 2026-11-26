# -*- coding: utf-8 -*-
"""E35  Is Blomberg's K trustworthy on the corrected tree?

All the descriptive numbers in Table 3 are K on a Grafen-length tree, and they halved when the
length convention was fixed.  Before any of that goes into the manuscript, check the statistic
against its own definition: on the same tree, data simulated as Brownian motion must give K ~ 1,
and data simulated with no phylogenetic structure must give K ~ 0.

The old convention is run through the same two checks, which is what shows whether the change is
a correction or a shift.
"""
import os, sys, json, io
import numpy as np

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, S)

import phylo
import fetch_real as FR
from run_revision import blomberg_k

DATA = os.path.join(os.path.dirname(S), 'data')


def old_grafen(root):
    """The previous implementation, kept here only to be measured."""
    h = phylo.heights(root)
    H = h[id(root)] or 1
    for v in phylo.postorder(root):
        v.length = 0.0 if v.parent is None else (H - h[id(v)]) / float(H)
    return root


def build(convention):
    tmpl = io.open(FR.NEWICK_CLADE, encoding='utf-8').read()
    root = phylo.parse_newick(tmpl)
    rows = FR.read_pantheria()
    match = json.load(io.open(os.path.join(DATA, 'otol_match.json'), encoding='utf-8'))
    blk = FR_blocks = None
    import exp_real as E
    blk = E.prep_block('life_history', rows, match, tmpl, None)
    labels = [E.otol_label(s, match[s]) for s in blk['species']]
    root, kept = phylo.prune(phylo.parse_newick(tmpl), set(labels))
    kept = [l for l in kept]
    if convention == 'fixed':
        root = phylo.collapse_unary(root)
        phylo.grafen_lengths(root)
    else:
        old_grafen(root)
    C = phylo.brownian_cov(root, kept)
    C = C / np.mean(np.diag(C))
    return root, kept, C


def tips(root):
    d = []
    for l in phylo.leaves(root):
        x, v = 0.0, l
        while v.parent is not None:
            x += v.length
            v = v.parent
        d.append(x)
    return np.array(d)


def main():
    for conv in ('fixed', 'old'):
        root, labels, C = build(conv)
        t = tips(root)
        n = len(labels)
        rng = np.random.default_rng(5)
        L = np.linalg.cholesky(C + 1e-10 * np.eye(n))
        kb = [blomberg_k(L @ rng.normal(size=n), C, nsim=200, seed=100 + i) for i in range(12)]
        ki = [blomberg_k(rng.normal(size=n), C, nsim=200, seed=200 + i) for i in range(12)]
        print('%-6s n=%d  tip distance %.4f to %.4f (sd %.4f)'
              % (conv, n, t.min(), t.max(), t.std()))
        print('        K for Brownian data   median %.2f  [%.2f, %.2f]'
              % (np.median(kb), min(kb), max(kb)))
        print('        K for iid data        median %.2f  [%.2f, %.2f]'
              % (np.median(ki), min(ki), max(ki)))
        print()


if __name__ == '__main__':
    main()
