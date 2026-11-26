# -*- coding: utf-8 -*-
"""Export the named case as a plain example set, and check the tool against e31.

Writes tool/example/life_history.tsv (species x six traits) and life_history.nwk (the pruned
synthesis subtree, topology only; the tool assigns Grafen lengths itself).  Then screens it and
compares against results/e31_named_case.json, so the standalone tool and the manuscript code
cannot drift apart.
"""
import os, io, json, sys
import numpy as np

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, S)
sys.path.insert(0, os.path.join(os.path.dirname(S), 'tool'))

import fetch_real as FR
import phylo
import _e31_named_case as E
import pic_screen

OUT = os.path.join(os.path.dirname(S), 'tool', 'example')


def to_newick(node):
    """Tips keep their names; internal labels are dropped (the screen only needs the topology,
    and the synthesis tree carries placeholder labels that are not worth propagating)."""
    if not node.children:
        return (node.name or 'unnamed').replace("'", '')
    inner = ','.join(to_newick(c) for c in node.children)
    return '(' + inner + ')'


def main():
    os.makedirs(OUT, exist_ok=True)
    species, X, match, logged = E.load()
    idx, C = E.principle_tree(species, match)
    X = X[idx]
    species = [species[i] for i in idx]
    frames = [sp.replace(' ', '_') for sp in species]

    with io.open(FR.NEWICK_CLADE, encoding='utf-8') as f:
        tmpl = f.read()
    root = phylo.parse_newick(tmpl)
    root, _ = phylo.prune(root, set(frames))
    nw = to_newick(root) + ';'

    tsv = os.path.join(OUT, 'life_history.tsv')
    with io.open(tsv, 'w', encoding='utf-8', newline='') as f:
        f.write('species\t' + '\t'.join(lab for _, lab in E.COLS) + '\n')
        for i, sp in enumerate(frames):
            f.write(sp + '\t' + '\t'.join('%.6g' % v for v in X[i]) + '\n')
    nwk = os.path.join(OUT, 'life_history.nwk')
    with io.open(nwk, 'w', encoding='utf-8') as f:
        f.write(nw)
    print('wrote %s (%d species x %d traits)' % (tsv, len(frames), X.shape[1]))
    print('wrote %s (%.1f KB)' % (nwk, os.path.getsize(nwk) / 1024.0))

    res = pic_screen.screen(X, nw, species=frames,
                            labels=[lab for _, lab in E.COLS], nsim=4000)
    print()
    print(res)
    print()
    for k, v in res.share.items():
        print('  %-30s %.3f' % (k, v))

    ref = json.load(io.open(os.path.join(os.path.dirname(S), 'results',
                                         'e31_named_case.json'), encoding='utf-8'))['full']
    print()
    print('e31 full   r1=%.2f  top=%s %.2f  deflated=%.2f'
          % (ref['r1'], ref['top'], ref['top_share'], ref['defl_r1']))
    print('tool       r1=%.2f  top=%s %.2f  deflated=%.2f'
          % (res.r1, res.top, res.top_share, res.deflated_r1))
    same = abs(res.r1 - ref['r1']) < 0.01 and res.top == ref['top']
    print('AGREE' if same else 'MISMATCH')

    summary = dict(n=res.n, d=res.d, r1=res.r1, p=res.p, threshold=res.threshold,
                   top=res.top, top_share=res.top_share, entropy=res.entropy,
                   deflated_r1=res.deflated_r1, deflated_p=res.deflated_p,
                   hub=res.hub, share=res.share, verdict=res.verdict)
    with io.open(os.path.join(OUT, 'life_history_screen.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print('saved', os.path.join(OUT, 'life_history_screen.json'))


if __name__ == '__main__':
    main()
