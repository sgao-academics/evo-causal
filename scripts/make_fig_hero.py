# -*- coding: utf-8 -*-
"""The named case, as one figure.

Four panels, in reading order:

  a  the 337 mammal species on the synthesis tree, with body mass as a strip beside the tips.
     The strip is patchy rather than dotted, which is the phylogenetic signal the whitening is
     meant to remove.
  b  the 6x6 contrast correlation matrix: one deep block, no isolated pair.
  c  the null distribution of the leading ratio, with the observed value and the value left
     after subtracting the leading component.  Both sit outside the null.
  d  where the leading direction sits: six squared loadings, largest 27%.

Everything is read from results/e31_named_case.json, results/e31b_null.json or recomputed from
the exported example, so the figure cannot drift from the numbers in the text.
"""
import os, io, json, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, 'results')
FIG = os.path.join(ROOT, 'figures')
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'tool'))

from palette import PAL, INK, GREY, TICK, GRID, FRAME, RAMP, make_cmap
from pic_screen import tree as PT

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 8,
    'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': INK, 'axes.linewidth': 0.7,
    'xtick.color': TICK, 'ytick.color': TICK, 'axes.labelcolor': INK,
    'text.color': INK, 'legend.frameon': False,
    'xtick.major.size': 2.4, 'ytick.major.size': 2.4,
    'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
})

LILAC, MIST, VIOLET, ORCHID = PAL['lilac'], PAL['mist'], PAL['violet'], PAL['orchid']
SHORT = {
    'adult body mass (g)': 'body mass',
    'age at sexual maturity (d)': 'maturity',
    'maximum longevity (mo)': 'longevity',
    'weaning age (d)': 'weaning',
    'gestation length (d)': 'gestation',
    'litter size': 'litter size',
}


def load_example():
    p = os.path.join(ROOT, 'tool', 'example')
    rows = [r.rstrip('\n').split('\t') for r in
            io.open(os.path.join(p, 'life_history.tsv'), encoding='utf-8') if r.strip()]
    labels = rows[0][1:]
    sp = [r[0] for r in rows[1:]]
    X = np.array([[float(x) for x in r[1:]] for r in rows[1:]])
    nw = io.open(os.path.join(p, 'life_history.nwk'), encoding='utf-8').read()
    return sp, labels, X, nw


def tree_xy(root, x0=0.0):
    """Horizontal tree: x is the number of nodes from the root, y an even spread over the tips.

    Node count rather than branch length: with 337 tips the accumulated lengths differ enough
    between lineages that the drawing reads as a web instead of a tree, and the panel is here to
    show the shape of the sample, not the timing.
    """
    tips = [l for l in PT.leaves(root) if l.name]
    order = {l.name: i for i, l in enumerate(sorted(tips, key=lambda v: v.name))}
    xs, ys = {}, {}

    def walk(v, x):
        xs[id(v)] = x
        if not v.children:
            ys[id(v)] = order[v.name]
            return
        for c in v.children:
            walk(c, x + 1.0)
        ys[id(v)] = float(np.mean([ys[id(c)] for c in v.children]))

    walk(root, x0)
    segs = [(xs[id(v)], ys[id(v)], xs[id(c)], ys[id(c)])
            for v in PT.postorder(root) for c in v.children]
    return segs, xs, ys, order


def panel_tree(ax, sp, X, nw, labels):
    """Traits for the 337 species in phylogenetic order.

    A drawing of the tree at this size is 337 hairlines and reads as texture rather than as a
    tree, so the ordering carries the phylogeny and the values carry the signal: the columns
    break into blocks, and a block is what makes a direction shared by relatives instead of by
    ecology.
    """
    root = PT.prune(PT.parse_newick(nw), set(sp))
    order = []

    def dfs(v):
        if not v.children:
            order.append(v.name)
            return
        for c in v.children:
            dfs(c)

    dfs(root)
    idx = [sp.index(nm) for nm in order if nm in sp]
    M = np.array(X)[idx]
    div = LinearSegmentedColormap.from_list('cf', [PAL['mist'], '#FFFFFF', PAL['orchid']])
    im = ax.imshow(M, aspect='auto', cmap=div, vmin=-2.6, vmax=2.6,
                   interpolation='nearest')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([SHORT.get(l, l) for l in labels], rotation=38, ha='right',
                       fontsize=6.2, color=TICK)
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis='x', length=0, pad=2)
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_ylabel('species, in phylogenetic order', fontsize=6.6, color=TICK)
    # no colour bar: the panel is qualitative (columns break into blocks) and the caption states
    # that colour is in standard deviations.  A bar here reads as belonging to panel b.
    ax.text(-0.10, 1.14, 'a', transform=ax.transAxes, fontsize=10, fontweight='bold')


def panel_matrix(ax, R, labels):
    order = [i for i in np.argsort(-np.abs(R).sum(1))]
    Ro = R[np.ix_(order, order)]
    lab = [SHORT.get(labels[i], labels[i]) for i in order]
    cmap = make_cmap()
    im = ax.imshow(Ro, cmap=cmap, vmin=0, vmax=1)
    cb = plt.colorbar(im, ax=ax, fraction=0.038, pad=0.03, ticks=[0, 0.5, 1])
    cb.ax.tick_params(labelsize=6, length=1.6, pad=1.5, colors=TICK)
    cb.outline.set_visible(False)
    ax.set_xticks(range(len(lab)))
    ax.set_yticks(range(len(lab)))
    ax.set_xticklabels(lab, rotation=42, ha='right', fontsize=6.4, color=TICK)
    ax.set_yticklabels(lab, fontsize=6.4, color=TICK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(lab), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(lab), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.2)
    ax.tick_params(which='minor', length=0)
    ax.text(-0.02, 1.10, 'b', transform=ax.transAxes, fontsize=10, fontweight='bold')


def panel_null(ax, null, r1, rd, thr):
    ax.hist(null, bins=np.linspace(null.min(), null.max(), 30), color=LILAC,
            edgecolor='white', linewidth=0.3, density=True)
    ax.axvline(r1, color=INK, lw=1.4)
    ax.axvline(rd, color=ORCHID, lw=1.4, ls=(0, (2.2, 1.4)))
    y = ax.get_ylim()[1]
    ax.set_ylim(0, y * 1.45)
    ax.annotate('as observed  %.2f' % r1, xy=(r1, y * 0.52), xytext=(r1 + 0.10, y * 1.16),
                fontsize=6.5, color=INK, va='center')
    ax.annotate('$\\lambda_1$ subtracted\n%.2f' % rd, xy=(rd, y * 0.50),
                xytext=(rd + 0.10, y * 0.86), fontsize=6.5, color=ORCHID, va='center',
                ha='left')
    ax.annotate('95%% null  %.2f' % thr, xy=(thr, y * 0.55),
                xytext=(thr + 0.13, y * 0.16), fontsize=6.2, color=TICK, va='center',
                ha='left',
                arrowprops=dict(arrowstyle='-', color=GREY, lw=0.6,
                                shrinkA=0, shrinkB=1))
    ax.set_xlabel('leading ratio  $\\lambda_1/\\overline{\\lambda}_{2..d}$', fontsize=7)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlim(null.min() - 0.12, r1 + 0.45)
    ax.text(-0.02, 1.06, 'c', transform=ax.transAxes, fontsize=10, fontweight='bold')


def panel_share(ax, share, labels):
    items = sorted(share.items(), key=lambda kv: -kv[1])
    names = [SHORT.get(k, k) for k, _ in items]
    vals = [v for _, v in items]
    ypos = np.arange(len(vals))[::-1]
    ax.barh(ypos, vals, height=0.62, color=MIST, edgecolor='none')
    for y, v in zip(ypos, vals):
        ax.text(v + 0.008, y, '%.0f%%' % (v * 100), va='center', fontsize=6.5, color=TICK)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=6.6, color=TICK)
    ax.set_xlim(0, 0.40)
    ax.set_xlabel('squared loading on the leading direction', fontsize=7)
    ax.tick_params(length=0)
    ax.spines['left'].set_visible(False)
    ax.axvline(1.0 / len(vals), color=GREY, lw=0.7, ls=(0, (1.6, 1.6)))
    ax.text(-0.02, 1.10, 'd', transform=ax.transAxes, fontsize=10, fontweight='bold')


def main():
    sp, labels, X, nw = load_example()
    res = json.load(io.open(os.path.join(RES, 'e31_named_case.json'), encoding='utf-8'))['full']
    nul = json.load(io.open(os.path.join(RES, 'e31b_null.json'), encoding='utf-8'))
    d = len(labels)
    n = len(sp)
    rng = np.random.default_rng(11)
    null = np.array([np.linalg.eigvalsh(np.corrcoef(rng.normal(size=(n, d)), rowvar=False))[::-1]
                     for _ in range(4000)])
    null_r1 = null[:, 0] / null[:, 1:].mean(1)
    thr = float(np.quantile(null_r1, 0.95))

    # contrast correlation from the same whitening the tool uses
    root = PT.prune(PT.parse_newick(nw), set(sp))
    PT.grafen_lengths(root)
    C = PT.brownian_cov(root, sp)
    L = np.linalg.cholesky(C + 1e-10 * np.eye(n))
    P = np.linalg.solve(L, X)
    R = np.corrcoef(P, rowvar=False)

    fig = plt.figure(figsize=(159 / 25.4, 5.05))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.12, 1.0],
                          height_ratios=[1.0, 0.82, 1.05], wspace=0.30, hspace=0.62)
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 1])

    panel_tree(ax1, sp, X, nw, labels)
    panel_matrix(ax2, R, labels)
    panel_null(ax3, null_r1, res['r1'], res['defl_r1'], thr)
    panel_share(ax4, res['shares'], labels)

    for ext in ('pdf', 'png'):
        out = os.path.join(FIG, 'fig_hero.%s' % ext)
        fig.savefig(out, dpi=400 if ext == 'png' else None, bbox_inches='tight',
                    facecolor='white')
        print('wrote', out)
    print('r1=%.3f  thr=%.2f  deflated=%.2f' % (res['r1'], thr, res['defl_r1']))


if __name__ == '__main__':
    main()
