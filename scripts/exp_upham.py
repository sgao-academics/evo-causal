# -*- coding: utf-8 -*-
"""E31  The mammal blocks on a dated topology, with the two effects separated.

Why this stage exists.  The empirical section uses the Open Tree of Life synthesis topology
with Grafen branch lengths, which carries no dates.  A reviewer who works with dated
phylogenies will ask whether the leading direction is a property of the traits or of the tree
the analysis happened to use.  Varying Grafen's rho on one topology does not answer that: it
varies the lengths of one tree and nothing else.

Two things can move the statistic, and they are usually confounded in this kind of check:
the topology, and the branch-length convention that turns it into a covariance.  Changing to a
dated tree changes both at once, and it also changes which species survive the join.  So the
join is done once, on the species present in the trait block and on both trees, and then three
computations are run on exactly that set:

    A  Open Tree topology  + Grafen lengths      (what the manuscript reports)
    B  Upham topology      + Grafen lengths      (difference from A is the topology)
    C  Upham topology      + dated lengths       (difference from B is the dated convention)

The tree is the completed MCC tree of Upham, Esselstyn and Jetz (2019), 5,911 species,
node-dated, distributed with the paper's code (n8uphem/MamPhy_v1).  It is used as an
alternative and not as a better tree: Upham et al. warn that species without DNA are imputed
within genus and family constraints, so the completed tree is not meant for single-tree
inference.  That caveat bears on the lengths of imputed branches; the question asked here is
only how far the statistic moves when the tree changes.

Each of the three is compared with the same sparse-graph reference as Table 3 (same rule for k,
5,000 draws, one null per block because n and d are shared).

Run: python exp_upham.py [--probe]
"""
import io
import json
import os
import re
import sys

import numpy as np

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(S), 'scripts'))
sys.path.insert(0, S)

import phylo
import exp_real as E
from fetch_real import read_pantheria, BLOCKS, NEWICK_CLADE
from exp_holm import graph_null_spec, NDRAW

DATA = os.path.join(os.path.dirname(S), 'data')
TREE = os.path.join(DATA, 'upham_mammals_mcc.tre')
OUT = os.path.join(os.path.dirname(S), 'results', 'e31_upham.json')
OTOL = io.open(NEWICK_CLADE, encoding='utf-8').read()


def log(*a):
    print(*a, flush=True)


def species_of(label):
    """Genus_species_FAMILY_ORDER -> Genus_species."""
    parts = label.split('_')
    return '_'.join(parts[:2]) if len(parts) >= 2 else label


def read_nexus_tree(path):
    """Return (newick with binomial tip labels, map id -> binomial) from the Upham NEXUS file.

    The file holds a Translate block of numbered tips labelled Genus_species_FAMILY_ORDER and
    one BEAST tree carrying [&...] annotations; both are handled here so the ordinary Newick
    parser can be reused.  Tips keep the binomial only, because the trait blocks are keyed by
    binomial and the family and order suffixes would change the join key.
    """
    s = io.open(path, encoding='utf-8', errors='ignore').read()
    i = s.find('Translate')
    j = s.find(';', i)
    names = {}
    for m in re.finditer(r'(\d+)\s+([A-Za-z][^\s,;]*)', s[i:j]):
        names[m.group(1)] = species_of(m.group(2))
    k = s.find('=', s.find('tree ', j))
    t = s[k + 1:]
    t = t[:t.rfind(')') + 1]
    t = re.sub(r'\[[^\]]*\]', '', t)
    t = re.sub(r'(?<=[(,])(\d+)(?=[:,)])', lambda m: names.get(m.group(1), m.group(1)), t)
    return t, names


def collapse_unary(root):
    """A pruned tree can leave the root with one child; fold that child into the root."""
    while len(root.children) == 1:
        c = root.children[0]
        c.length = (c.length or 0.0) + (root.length or 0.0)
        c.parent = None
        root = c
    return root


def corr_of(M):
    M = np.asarray(M, float)
    M = M - M.mean(0)
    sd = M.std(0)
    sd[sd < 1e-12] = 1.0
    return np.corrcoef(M / sd, rowvar=False)


def lead_ratio(R):
    w = np.linalg.eigvalsh(R)[::-1]
    return float(w[0] / max(w[1:].mean(), 1e-12)), w


def save_json(obj):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(obj, io.open(OUT, 'w', encoding='utf-8'), indent=1)


def main():
    probe = '--probe' in sys.argv
    newick, upham_names = read_nexus_tree(TREE)
    root0 = phylo.parse_newick(newick)
    depths, _ = phylo.tip_depths(root0)
    dm = np.array(list(depths.values()), float)
    log('dated tree: %d tips, ultrametric spread %.2e of %.1f Myr, %d binomials'
        % (len(dm), float(dm.max() - dm.min()), float(dm.max()), len(set(upham_names.values()))))
    upham_sp = set(upham_names.values())

    rows = read_pantheria()
    match = json.load(io.open(os.path.join(DATA, 'otol_match.json'), encoding='utf-8'))
    res = {}
    if os.path.exists(OUT):
        try:
            res = json.load(io.open(OUT, encoding='utf-8'))
        except Exception:
            res = {}

    blocks = ['life_history'] if probe else list(BLOCKS)
    for name in blocks:
        blk = E.prep_block(name, rows, match, None, None)
        row_of = {sp: i for i, sp in enumerate(blk['species'])}
        both = [sp for sp in blk['species']
                if sp.replace(' ', '_') in upham_sp and match.get(sp)]
        d = len(blk['cols'])
        log('%-14s complete-case %4d   on both trees %4d   d=%d'
            % (name, len(blk['species']), len(both), d))
        if len(both) < 20:
            log('  too few species on both trees, skipped')
            continue

        blk2 = dict(blk)
        blk2['species'] = both
        blk2['X'] = blk['X'][[row_of[sp] for sp in both]]
        blk2['match'] = match
        row2 = {sp: i for i, sp in enumerate(both)}

        # ---- A  Open Tree topology + Grafen lengths, on exactly this species set
        Xa, la, Ca, sa = E.trimmed(blk2, OTOL, 'grafen', 1.0)
        Ca = Ca / Ca.diagonal().max()
        Ra = corr_of(E.whiten(Xa, Ca))
        rA, wA = lead_ratio(Ra)

        # ---- B, C  Upham topology, same species set, two length conventions
        labels_up = [sp.replace(' ', '_') for sp in sa]
        X_up = blk2['X'][[row2[sp] for sp in sa]]
        out = dict(n=int(len(sa)), d=int(d), n_open_tree=int(len(blk['species'])),
                   join=round(len(sa) / max(1, len(blk['species'])), 3),
                   open_tree_grafen=round(rA, 3))
        for conv in ('grafen', 'dated'):
            root = phylo.parse_newick(newick)
            root, _ = phylo.prune(root, set(labels_up))
            root = collapse_unary(root)
            if conv == 'grafen':
                phylo.grafen_lengths(root)
            C = phylo.brownian_cov(root, labels_up)
            C = C / C.diagonal().max()
            r, _ = lead_ratio(corr_of(E.whiten(X_up, C)))
            out['upham_' + conv] = round(r, 3)
            log('   %-9s + %-6s  n=%4d  r1=%.3f' % ('Upham', conv, len(sa), r))

        # ---- why C can fall: a dated tree has very short terminal branches, and a contrast
        # across two close relatives is a small difference in which a literature-compiled
        # trait's measurement error is large relative to the signal.  That is not a claim that
        # can be asserted; it is a claim that makes a prediction, which is what this ladder
        # tests.  Adding delta^2 * I to the contrast covariance is the standard tip-error term:
        # at delta = 0 the model is pure Brownian motion and at large delta the whitening is
        # switched off, so r1 must climb from the dated value to the unwhitened value if the
        # attenuation is error and must not move if it is a property of the traits.
        root_d = phylo.parse_newick(newick)
        root_d, _ = phylo.prune(root_d, set(labels_up))
        root_d = collapse_unary(root_d)
        Cd = phylo.brownian_cov(root_d, labels_up)
        Cd = Cd / Cd.diagonal().max()
        raw = lead_ratio(corr_of(X_up))[0]
        ladder = {}
        for delta in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0):
            Cx = Cd + (delta ** 2) * np.eye(len(sa))
            ladder['%.2f' % delta] = round(lead_ratio(corr_of(E.whiten(X_up, Cx)))[0], 3)
        out['tip_error_ladder'] = ladder
        out['unwhitened'] = round(raw, 3)
        log('   tip-error ladder ' + ' '.join('%s:%.2f' % (k2, v) for k2, v in ladder.items())
            + '  | unwhitened %.2f' % raw)

        # ---- one reference for this block: n and d are shared by all three
        kref = max(2, int(round(d / 3.0)))
        reps = 200 if probe else NDRAW
        nul = graph_null_spec(len(sa), d, kref, reps=reps)[:, 0]
        null = dict(k=int(kref), mean=round(float(nul.mean()), 3),
                    sd=round(float(nul.std()), 3), draws=int(len(nul)))
        for key in ('open_tree_grafen', 'upham_grafen', 'upham_dated'):
            r = out[key]
            out['p_' + key] = round(float((1 + np.sum(nul >= r)) / (len(nul) + 1)), 5)
        out['null'] = null
        out['delta_topology'] = round(out['upham_grafen'] - out['open_tree_grafen'], 3)
        out['delta_dates'] = round(out['upham_dated'] - out['upham_grafen'], 3)
        res[name] = out
        save_json(res)
        log('   A %.2f (p=%.4f) | B %.2f (p=%.4f) | C %.2f (p=%.4f) | ref %.2f+-%.2f'
            % (rA, out['p_open_tree_grafen'], out['upham_grafen'], out['p_upham_grafen'],
               out['upham_dated'], out['p_upham_dated'], null['mean'], null['sd']))

    if not probe:
        log('written', OUT)


if __name__ == '__main__':
    main()
