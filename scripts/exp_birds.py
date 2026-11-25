# -*- coding: utf-8 -*-
"""E27  The same measurement on a second clade, from a second trait database.

Everything here is exp_real.py's protocol applied to birds: EltonTraits 1.0 traits,
the Open Tree of Life Aves subtree, Grafen branch lengths, the contrast correlation
matrix, the exact null, the tetrad test and the hub.

The point is not more data. It is that the mammal anchor uses one class and one trait
compilation, so "real comparative matrices land in the failure region" could be a property
of PanTHERIA. Birds are a separate radiation with a separate compilation.

Run: python exp_birds.py
"""
import os, sys, json
import numpy as np

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(S), 'scripts'))
sys.path.insert(0, S)

import exp_real as E

DATA = os.path.join(os.path.dirname(S), 'data')


def main():
    res = E.load('e27_birds')
    blocks = json.load(open(os.path.join(DATA, 'bird_blocks.json'), encoding='utf-8'))
    match = json.load(open(os.path.join(DATA, 'otol_match_birds.json'), encoding='utf-8'))
    tree = open(os.path.join(DATA, 'otol_aves.nwk'), encoding='utf-8').read()

    for name, b in blocks.items():
        if name in res and 'contrast' in res[name]:
            E.log('skip', name)
            continue
        blk = dict(species=b['species'], X=np.array(b['X'], float), cols=b['cols'],
                   labels=b['labels'], logged=b['logged'], dropped=b['dropped'],
                   match=match, block=name)
        o = E.block_stats(blk, tree)
        if o is None:
            continue
        o['clade'] = 'Aves'
        o['variants'] = E.variants(blk, tree)
        res[name] = o
        E.save('e27_birds', res)
        E.log('%-12s n=%4d d=%2d K=%.2f [%.2f,%.2f]  contrast r1=%.2f p=%.1e '
              'r2=%.2f p=%.2f  tetrad %.2f (size %.2f)  hub=%s'
              % (name, o['n'], o['d'], o['K_mean'], o['K_min'], o['K_max'],
                 o['contrast']['r'][0], o['contrast']['p'][0],
                 o['contrast']['r'][1], o['contrast']['p'][1],
                 o['tetrad']['reject'], o['tetrad']['size'],
                 o['hub']['top_label']))


if __name__ == '__main__':
    main()
