# -*- coding: utf-8 -*-
"""Figure 1: the design, and the exactness of the correction.

Four panels in reading order, no decorative containers. (a) a random ultrametric tree on which an
unmeasured factor evolves; (b) the covariance that factor implies; (c) and (d) the same transform
applied under the model PIC assumes and under OU covariance. Panels (c) and (d) are adjacent on
purpose: the claim of the figure is the contrast between a matrix with nothing off its diagonal
and one that keeps structure there.

Encoding choices, and why
------------------------
* The diagonal is drawn as a grey band and the colour scale covers the OFF-diagonal only. Drawn
  the other way round -- one scale over the whole matrix -- the residual in (d) is 8 per cent of
  the diagonal and simply disappears, which would hide the very result the panel exists for.
* (c) and (d) share one scale, fixed at the largest absolute off-diagonal entry of (d), read from
  the result file. (c) is therefore white: that is what "diagonal to machine precision" looks like.
* Both matrices are divided by their mean diagonal before display, so the number printed on the
  page and the colour on the page are the same quantity. Raw M_OU has entries above unity on its
  diagonal, and displaying it unscaled puts the matrix and the printed maximum on two scales.
* Frames carry the reading: mist where the assumed model holds, orchid where it does not, neutral
  for the input. No arrow, no card, no in-figure headline -- the caption carries all of it.

Every number is read from results/e1_whitening_n12.json and scripts/_fig1_arrays.npz.
"""
import os, json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, 'results')
FIG = os.path.join(ROOT, 'figures')
os.makedirs(FIG, exist_ok=True)

from palette import PAL, INK, GREY, TICK, FRAME, make_cmap
from run_revision import build_ultrametric_tree

# Drawn at the actual text width of the manuscript (A4 minus 1 in margins = 159 mm).
FIGW = 159 / 25.4
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 8,
    'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': INK, 'axes.linewidth': 0.7,
    'xtick.color': INK, 'ytick.color': INK, 'axes.labelcolor': INK,
    'text.color': INK, 'legend.frameon': False,
})


# --------------------------------------------------------------------------- tree geometry
def tree_segments(root):
    """Left-to-right phylogram: tips evenly spaced down the page, branches to the right.

    Even spacing is the point. Laid out from branch lengths alone the twelve tips of this tree
    land in two clumps and the drawing reads as a smudge; spaced by index every tip is separated
    by the same distance and the topology stays exact, because each vertical connector sits at
    the parent's height rather than at the child's.
    """
    segs, tips = [], []
    counter = [0]

    def rec(node, x):
        if not node.children:
            y = counter[0]
            counter[0] += 1
            tips.append((x, y))
            return y
        ys = [rec(c, x + c.branch_length) for c in node.children]
        y = float(np.mean(ys))
        segs.append([(x, min(ys)), (x, max(ys))])          # vertical connector at parent height
        for c, yy in zip(node.children, ys):
            segs.append([(x, yy), (x + c.branch_length, yy)])
        return y

    y_root = rec(root, 0.0)
    return segs, tips, y_root


# --------------------------------------------------------------------------- helpers
def off_diagonal(M):
    """Return M with its diagonal masked, so the colour scale can be spent on the rest."""
    A = np.array(M, dtype=float)
    np.fill_diagonal(A, np.nan)
    return np.ma.masked_invalid(A)


def matrix_panel(ax, data, edge, vmin, vmax, cmap):
    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper', interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(edge)
        s.set_linewidth(0.9)


def label(ax, s, dx=0.0, dy=0.0):
    """Panel letter in figure coordinates, just above the axes' top-left corner.

    Placed inside the column rather than outside it: the matrices sit close together and there
    is no gutter wide enough for a letter, but there is a clear band above each matrix.
    """
    fig = ax.figure
    fig.canvas.draw()
    bb = ax.get_position()
    fig.text(bb.x0 + dx, bb.y1 + dy, s, fontsize=10, fontweight='bold',
             va='bottom', ha='left', color=INK)


# --------------------------------------------------------------------------- figure
def main():
    e1 = json.load(open(os.path.join(RES, 'e1_whitening_n12.json'), encoding='utf-8'))
    arr = np.load(os.path.join(HERE, '_fig1_arrays.npz'))
    V, M_BM, M_OU = arr['V'], arr['M_BM'], arr['M_OU']
    n = e1['n']
    max_bm = e1['BM']['max_offdiag']
    max_ou = e1['OU_alpha3']['max_offdiag']

    # one scale for both transformed panels, set by the larger of the two residuals
    vmax_off = max(max_bm, max_ou, 1e-6)

    cmap_off = make_cmap()
    cmap_off.set_bad(FRAME)          # masked diagonal: a neutral band, not a value

    fig = plt.figure(figsize=(FIGW, 3.16))
    # the tree needs only a third of what the matrices need: give it 0.88 of a column and every
    # millimetre saved there goes straight into the three matrices, which are the evidence
    gs = fig.add_gridspec(1, 4, width_ratios=[0.88, 1.04, 1.04, 1.04],
                          left=0.018, right=0.996, top=0.905, bottom=0.205,
                          wspace=0.165)

    # ---------------------------------------------------------------- (a) the process
    ax = fig.add_subplot(gs[0, 0])
    tree, total_h = build_ultrametric_tree(n, seed=e1['tree_seed'])
    segs, tips, y_root = tree_segments(tree)
    for (p, q) in segs:
        ax.plot([p[0] / total_h, q[0] / total_h], [p[1], q[1]],
                color=TICK, lw=0.7, solid_capstyle='round', zorder=2)
    for (x, y) in tips:
        ax.plot([x / total_h], [y], 'o', ms=2.2, color=PAL['mist'],
                mec='white', mew=0.35, zorder=3)
    ax.plot([0], [y_root], 'o', ms=3.8, color=PAL['orchid'], mec='white', mew=0.55, zorder=4)
    ax.annotate('', xy=(-0.012, y_root), xytext=(-0.145, y_root),
                arrowprops=dict(arrowstyle='-|>', color=PAL['orchid'], lw=1.0,
                                mutation_scale=7))
    ax.text(-0.155, y_root + 0.45, '$L$', color=PAL['orchid'], fontsize=8.5,
            ha='right', va='bottom')
    ax.set_xlim(-0.20, 1.02)
    ax.set_ylim(n + 0.85, -1.05)
    ax.axis('off')
    # the two facts a reader needs, parked in the bands the drawing leaves empty, so neither
    # can land on a branch. Placed in axes coordinates: the letter top-left, the model top-right,
    # the tip count bottom-left.
    ax.text(0.0, 1.0, 'a', transform=ax.transAxes, fontsize=10, fontweight='bold',
            ha='left', va='top', color=INK)
    ax.text(1.0, 1.0, '$L\\sim\\mathrm{BM}(\\mathbf{V})$', transform=ax.transAxes, fontsize=7,
            color=PAL['orchid'], ha='right', va='top')
    ax.text(0.0, 0.0, '%d tips' % n, transform=ax.transAxes, fontsize=6.5,
            color=GREY, ha='left', va='bottom')

    # ---------------------------------------------------------------- (b) the covariance
    ax = fig.add_subplot(gs[0, 1])
    matrix_panel(ax, V, FRAME, 0.0, 1.0, make_cmap())
    ax.text(0.5, -0.045, '$V_{ij}=h(\\mathrm{MRCA})\\,/\\,h(\\mathrm{root})$',
            transform=ax.transAxes, ha='center', va='top', fontsize=7, color=INK)
    ax.text(0.0, 1.015, 'b', transform=ax.transAxes, fontsize=10, fontweight='bold',
            ha='left', va='bottom', color=INK)
    ax.text(0.125, 1.038, 'covariance $\\mathbf{V}$', transform=ax.transAxes, ha='left',
            va='bottom', fontsize=7.5, color=INK, fontweight='bold')

    # ---------------------------------------------------------------- (c) under the model
    ax = fig.add_subplot(gs[0, 2])
    M_bm = M_BM / np.mean(np.diag(M_BM))
    matrix_panel(ax, off_diagonal(M_bm), PAL['mist'], 0.0, vmax_off, cmap_off)
    ax.text(0.5, -0.045, 'max$|$off-diag$|$ = %.3f' % max_bm,
            transform=ax.transAxes, ha='center', va='top', fontsize=7.5, color=INK)
    ax.text(0.0, 1.015, 'c', transform=ax.transAxes, fontsize=10, fontweight='bold',
            ha='left', va='bottom', color=INK)
    ax.text(0.125, 1.038, 'Brownian', transform=ax.transAxes, ha='left', va='bottom',
            fontsize=7.5, color=PAL['mist'], fontweight='bold')

    # ---------------------------------------------------------------- (d) under OU
    ax = fig.add_subplot(gs[0, 3])
    M_ou = M_OU / np.mean(np.diag(M_OU))
    matrix_panel(ax, off_diagonal(M_ou), PAL['orchid'], 0.0, vmax_off, cmap_off)
    ax.text(0.5, -0.045, 'max$|$off-diag$|$ = %.3f' % max_ou,
            transform=ax.transAxes, ha='center', va='top', fontsize=7.5, color=INK)
    ax.text(0.0, 1.015, 'd', transform=ax.transAxes, fontsize=10, fontweight='bold',
            ha='left', va='bottom', color=INK)
    ax.text(0.125, 1.038, 'OU, $\\alpha=3$', transform=ax.transAxes, ha='left', va='bottom',
            fontsize=7.5, color=PAL['orchid'], fontweight='bold')

    # ---------------------------------------------------------------- scale note
    # positioned from the drawn boxes rather than from a guessed figure coordinate
    fig.canvas.draw()
    b_c = fig.axes[2].get_position()
    b_d = fig.axes[3].get_position()
    fig.text((b_c.x0 + b_d.x1) / 2.0, b_c.y0 - 0.135,
             '$\\mathbf{V}$: 0–1        $\\mathbf{AVA}^{\\top}$: $|$off-diag$|$ 0–%.3f, '
             'diag grey' % vmax_off,
             fontsize=6.9, color=GREY, ha='center', va='bottom')

    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, 'fig1_design.' + ext),
                    dpi=400 if ext == 'png' else None, facecolor='white',
                    bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print('fig1 ok  n=%d  BM=%.5f  OU3=%.5f  vmax=%.5f' % (n, max_bm, max_ou, vmax_off))


if __name__ == '__main__':
    main()
