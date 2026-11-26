# -*- coding: utf-8 -*-
"""E36  Where does Blomberg's K sit when there is no phylogenetic structure at all?

Table 3 reports K per trait, and the manuscript reads those numbers against a band.  A band only
means something if the statistic's own floor on the same tree is known: on a tree with a given
shape, independent traits do not score zero.  This computes, per block and on the corrected
lengths, the K of iid data and of Brownian data drawn on that block's own covariance, and then
counts how many of the observed traits are indistinguishable from the iid floor.

Run: python _e36_kfloor.py
"""
import os, sys, json, io
import numpy as np

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, S)

import phylo
import exp_real as E
import fetch_real as FR
from run_revision import blomberg_k

DATA = os.path.join(os.path.dirname(S), 'data')
REPS = 40
NSIM = 200


def cov_of(blk, tree_text):
    X, labels, C, species = E.trimmed(blk, tree_text)
    C = C / np.mean(np.diag(C))
    return X, labels, C, species


def main():
    out = {}
    root_text = io.open(FR.NEWICK_CLADE, encoding='utf-8').read()
    rows = FR.read_pantheria()
    match = json.load(io.open(os.path.join(DATA, 'otol_match.json'), encoding='utf-8'))
    res_m = E.load('e26_real')
    res_b = E.load('e27_birds')
    bird_blocks = json.load(io.open(os.path.join(DATA, 'bird_blocks.json'), encoding='utf-8'))
    bird_match = json.load(io.open(os.path.join(DATA, 'otol_match_birds.json'), encoding='utf-8'))
    bird_tree = io.open(os.path.join(DATA, 'otol_aves.nwk'), encoding='utf-8').read()

    jobs = []
    for name in FR.BLOCKS:
        if name not in res_m:
            continue
        blk = E.prep_block(name, rows, match, root_text, None)
        blk['block'], blk['match'] = name, match
        jobs.append((name, blk, root_text, res_m[name]))
    for name, b in bird_blocks.items():
        if name not in res_b:
            continue
        blk = dict(species=b['species'], X=np.array(b['X'], float), cols=b['cols'],
                   labels=b['labels'], logged=b['logged'], dropped=b['dropped'],
                   match=bird_match, block=name)
        jobs.append((name, blk, bird_tree, res_b[name]))

    for name, blk, tree, res in jobs:
        X, labels, C, species = cov_of(blk, tree)
        n = len(labels)
        L = np.linalg.cholesky(C + 1e-10 * np.eye(n))
        rng = np.random.default_rng(3)
        k_iid = [blomberg_k(rng.normal(size=n), C, nsim=NSIM, seed=500 + i) for i in range(REPS)]
        k_bm = [blomberg_k(L @ rng.normal(size=n), C, nsim=NSIM, seed=900 + i)
                for i in range(REPS)]
        floor = float(np.quantile(k_iid, 0.95))
        obs = res['K']
        below = sum(1 for k in obs if k <= floor)
        out[name] = dict(n=n, d=len(obs), iid_median=round(float(np.median(k_iid)), 3),
                         iid_q95=round(floor, 3), bm_median=round(float(np.median(k_bm)), 3),
                         obs_mean=round(float(np.mean(obs)), 3),
                         obs_min=round(float(np.min(obs)), 3),
                         obs_max=round(float(np.max(obs)), 3),
                         at_floor=below)
        print('%-14s n=%4d d=%2d | iid K %.2f (q95 %.2f) | BM K %.2f | obs %.2f [%.2f, %.2f] '
              '| %d/%d at or below the iid 95th'
              % (name, n, len(obs), np.median(k_iid), floor, np.median(k_bm),
                 np.mean(obs), np.min(obs), np.max(obs), below, len(obs)))

    tot = sum(v['d'] for v in out.values())
    at = sum(v['at_floor'] for v in out.values())
    print()
    print('across %d traits: %d at or below the iid 95th percentile of their own tree'
          % (tot, at))
    with io.open(os.path.join(os.path.dirname(S), 'results', 'e36_kfloor.json'), 'w',
                 encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
