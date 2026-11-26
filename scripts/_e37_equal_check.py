# -*- coding: utf-8 -*-
"""E37  Does the equal-length convention leave the tips contemporaneous?

Setting every edge to length one makes root-to-tip distance equal to a tip's depth, and a pruned
synthesis tree is unbalanced, so the tips are not equidistant.  If so, the equal-length variant is
not a contrasts computation at all and must not be described as a sensitivity of one.
"""
import io, json, os, sys
import numpy as np

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, S)

import phylo
import exp_real as E
import fetch_real as FR

DATA = os.path.join(os.path.dirname(S), 'data')


def tipdist(root):
    out = []
    for l in phylo.leaves(root):
        x, v = 0.0, l
        while v.parent is not None:
            x += v.length
            v = v.parent
        out.append(x)
    return np.array(out)


def main():
    tmpl = io.open(FR.NEWICK_CLADE, encoding='utf-8').read()
    rows = FR.read_pantheria()
    match = json.load(io.open(os.path.join(DATA, 'otol_match.json'), encoding='utf-8'))
    for name in ('life_history', 'ecology', 'range_climate'):
        blk = E.prep_block(name, rows, match, tmpl, None)
        blk['block'], blk['match'] = name, match
        labels = [E.otol_label(s, match[s]) for s in blk['species']]
        root, kept = phylo.prune(phylo.parse_newick(tmpl), set(labels))
        root = phylo.collapse_unary(root)
        phylo.equal_lengths(root)
        d = tipdist(root)
        print('%-14s equal lengths: tip distance %.3f to %.3f (spread %.3f of %.3f)'
              % (name, d.min(), d.max(), d.max() - d.min(), d.max()))
        root2, kept2 = phylo.prune(phylo.parse_newick(tmpl), set(labels))
        root2 = phylo.collapse_unary(root2)
        phylo.grafen_lengths(root2)
        d2 = tipdist(root2)
        print('%-14s grafen        : tip distance %.6f to %.6f'
              % ('', d2.min(), d2.max()))


if __name__ == '__main__':
    main()
