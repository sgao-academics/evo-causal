# -*- coding: utf-8 -*-
"""E34  Regenerate the five real-data blocks with the corrected tree conventions.

phylo.grafen_lengths used to give every edge the depth (H - h)/H instead of the increment
(h_parent - h)/H, so tips were not contemporaneous and the whitened rows were not contrasts.
exp_real.power_lengths had a second defect: it differenced the depths the wrong way round and
clamped to 1e-6, so every rho in {0.5, 2} produced a tree of constant branch length.

Both are fixed at the source.  This script recomputes every published block and prints the
old and new values side by side, so nothing in the manuscript changes without a recorded
before-and-after.  Outputs go to *_fixed.json; e26_real.json and e27_birds.json are left
untouched until the numbers have been read.

Run: python _e34_rerun.py
"""
import os, sys, json, time
import numpy as np

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, S)

import phylo
import exp_real as E

DATA = os.path.join(os.path.dirname(S), 'data')


def check_tree(root, labels):
    """Tips must all sit at the same distance from the root, or the whiten is not a contrast."""
    d = {}
    for l in phylo.leaves(root):
        x, v = 0.0, l
        while v.parent is not None:
            x += v.length
            v = v.parent
        d[l.name] = x
    vals = np.array([d.get(k, np.nan) for k in labels])
    return float(np.nanmin(vals)), float(np.nanmax(vals)), int(np.sum(~np.isfinite(vals)))


def mammal_blocks():
    res = {}
    root_template = open(os.path.join(DATA, 'otol_clade.nwk'), encoding='utf-8').read() \
        if os.path.exists(os.path.join(DATA, 'otol_clade.nwk')) else None
    import fetch_real as FR
    root_template = open(FR.NEWICK_CLADE, encoding='utf-8').read()
    rows = FR.read_pantheria()
    match = json.load(open(os.path.join(DATA, 'otol_match.json'), encoding='utf-8'))
    for name in FR.BLOCKS:
        t0 = time.time()
        blk = E.prep_block(name, rows, match, root_template, None)
        if blk is None:
            continue
        blk['block'] = name
        blk['match'] = match
        o = E.block_stats(blk, root_template, verbose=False)
        if o is None:
            continue
        o['variants'] = E.variants(blk, root_template)
        res[name] = o
        print('  %-14s n=%4d d=%2d  r1=%.2f  K=%.2f  (%.0fs)'
              % (name, o['n'], o['d'], o['contrast']['r'][0], o['K_mean'], time.time() - t0))
    return res


def bird_blocks():
    res = {}
    blocks = json.load(open(os.path.join(DATA, 'bird_blocks.json'), encoding='utf-8'))
    match = json.load(open(os.path.join(DATA, 'otol_match_birds.json'), encoding='utf-8'))
    tree = open(os.path.join(DATA, 'otol_aves.nwk'), encoding='utf-8').read()
    for name, b in blocks.items():
        t0 = time.time()
        blk = dict(species=b['species'], X=np.array(b['X'], float), cols=b['cols'],
                   labels=b['labels'], logged=b['logged'], dropped=b['dropped'],
                   match=match, block=name)
        o = E.block_stats(blk, tree, verbose=False)
        if o is None:
            continue
        o['clade'] = 'Aves'
        o['variants'] = E.variants(blk, tree)
        res[name] = o
        print('  %-12s n=%4d d=%2d  r1=%.2f  K=%.2f  (%.0fs)'
              % (name, o['n'], o['d'], o['contrast']['r'][0], o['K_mean'], time.time() - t0))
    return res


def show(old, new, tag):
    print()
    print('=' * 96)
    print('%-13s %5s %5s | %14s | %18s | %16s | %10s'
          % ('block', 'n', 'd', 'K  old -> new', 'r1  old -> new', 'r3  old -> new',
             'r1 after defl'))
    print('-' * 96)
    for k, o in new.items():
        p = old.get(k, {})
        print('%-13s %5d %5d | %5.2f -> %5.2f  | %5.2f -> %5.2f  p=%7.5f | %5.2f -> %5.2f | %5.2f -> %5.2f'
              % (k, o['n'], o['d'],
                 p.get('K_mean', float('nan')), o['K_mean'],
                 p.get('contrast', {}).get('r', [float('nan')])[0], o['contrast']['r'][0],
                 o['contrast']['p'][0],
                 p.get('contrast', {}).get('r', [0, 0, float('nan')])[2], o['contrast']['r'][2],
                 p.get('r_after_deflation', float('nan')), o['r_after_deflation']))
    print()
    print('%-13s %-26s %-26s' % ('block', 'rho sensitivity old ?', 'rho 0.5 / 1.0 / 2.0 / equal'))
    for k, o in new.items():
        v = o['variants']
        print('%-13s %-26s %s' % (k, '',
                                  ' / '.join('%.2f' % v[t]['r'] for t in
                                             ('grafen_0.5', 'grafen_1.0', 'grafen_2.0', 'equal'))))
    print()


def main():
    print('mammal blocks')
    new_m = mammal_blocks()
    print('bird blocks')
    new_b = bird_blocks()

    old_m = E.load('e26_real')
    old_b = E.load('e27_birds')
    show(old_m, new_m, 'mammal')
    show(old_b, new_b, 'bird')

    E.save('e26_real_fixed', new_m)
    E.save('e27_birds_fixed', new_b)
    print('saved results/e26_real_fixed.json and results/e27_birds_fixed.json')


if __name__ == '__main__':
    main()
