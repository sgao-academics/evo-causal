# -*- coding: utf-8 -*-
"""Figure 3: three arms that reach the same K by different routes, and what each does to F1
and AUROC.

Redesigned 2026-11-25. What the old version got wrong:

1. The iid arm was drawn as five triangles stacked vertically at K = 0.079. Five markers read
   as five different datasets on the x axis, but they are five confounding loads that all land
   on the same K -- that identity is the whole point of the arm. It is now a vertical range bar
   with end caps: one K, a spread in F1 caused by lambda. The bar says "K did not move, F1 did".
2. Every point carries K_sd on the horizontal axis. The tree arm has K_sd up to 0.30, so its
   points are not measurements of a sharp abscissa at all; that is evidence for the paper's own
   claim, and the old figure hid it.
3. The F1 = 0.8 line is gone, as in Fig. 2: nothing in the paper defines 0.8, and the derived
   chance level k/M is the reference that has a source.
4. Panel (b) gains the 0.5 chance line, and both panels now span 0 to 1 so the two panels can
   be compared without rescaling them in the reader's head.
5. Colours follow the house meaning rather than being chosen per figure: moss green is the
   control arm, violet the third category, orchid the arm that has no tree in it. Blue is
   reserved for "this work" (the PIC analysis of Fig. 2) and is not reused here for a confounder.

The d = 50 iid bar is offset 0.013 to the right so that the two overlapping ranges stay
readable; the vertical extent is unchanged and the offset is declared in the caption.

One figure, one script: running this file writes figures/fig3_dissociation.pdf and .png and
touches no other figure. Every number is read from results/*.json, so the figure cannot
disagree with the tables or with the text.
"""
import os, json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, 'results')
FIG = os.path.join(ROOT, 'figures')
os.makedirs(FIG, exist_ok=True)

try:
    from palette import PAL, INK, GREY, GRID, TICK, FRAME
except Exception:
    PAL = {'mist': '#4D6EAF', 'peri': '#7A9AC3', 'lilac': '#B0B9CB',
           'orchid': '#D0A0B6', 'moss': '#8FB578', 'violet': '#9A5A9F',
           'indigo': '#3F3A5B'}
    INK, GREY, GRID, TICK, FRAME = '#3F3A5B', '#8A93A6', '#EDF1F7', '#5B6478', '#C9D2E0'

FIGW = 159 / 25.4
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

C_SIG = PAL['moss']      # signal without a confounder -- the control arm
C_TREE = PAL['violet']   # confounder that evolves on the tree
C_IID = PAL['orchid']    # confounder with no phylogenetic structure
K_TRUE = 10
IID_OFFSET = 0.013


def load(n):
    p = os.path.join(RES, n + '.json')
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else {}


def arm_series(res, arm, dd, field):
    xs, ys, es = [], [], []
    for k, v in res.items():
        if v.get('arm') == arm and v.get('d') == dd:
            xs.append(v['K'])
            ys.append(v[field])
            es.append(v['K_sd'])
    o = np.argsort(xs)
    return np.array(xs)[o], np.array(ys)[o], np.array(es)[o]


def draw_iid(ax, x, y, ls, filled):
    """One K, a range of scores: a vertical bar with end caps, not five stacked markers."""
    if not len(x):
        return
    xc = float(np.mean(x)) + (0.0 if ls == '-' else IID_OFFSET)
    ax.plot([xc, xc], [y.min(), y.max()], color=C_IID, lw=1.9, ls=ls,
            solid_capstyle='butt', zorder=3)
    for yy in (y.min(), y.max()):
        ax.plot([xc - 0.013, xc + 0.013], [yy, yy], color=C_IID, lw=1.2, ls=ls,
                solid_capstyle='butt', zorder=3)
    ax.plot([xc], [y.max()], marker='^', ms=4.2, color=C_IID,
            mfc=C_IID if filled else 'white', mew=0.9, ls='none', zorder=4)


def main():
    res = load('e14_dissociation')
    if not res:
        print('e14_dissociation.json missing')
        return

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(FIGW, 2.95))

    for ax, field, ylab in ((ax_a, 'f1_pic', '$F_1$ under top-$k$'),
                            (ax_b, 'auroc', 'AUROC')):
        # tree first so that the circles of the signal arm sit on top where the two coincide
        for dd, ls, filled in ((30, '-', True), (50, '--', False)):
            for arm, col, mk, ms, lw in (('tree', C_TREE, 's', 3.2, 1.15),
                                         ('signal', C_SIG, 'o', 3.7, 1.3)):
                x, y, e = arm_series(res, arm, dd, field)
                if not len(x):
                    continue
                # the spread of K is drawn first and light, so the curves stay on top: the
                # uncertainty is evidence, not the subject of the panel
                ax.errorbar(x, y, xerr=e, fmt='none', ecolor=col, elinewidth=0.5,
                            capsize=0, alpha=0.42, zorder=2)
                ax.plot(x, y, ls=ls, marker=mk, ms=ms, lw=lw, color=col,
                        mfc=col if filled else 'white', mew=0.85, zorder=4)
            x, y, _ = arm_series(res, 'iid', dd, field)
            draw_iid(ax, x, y, ls, filled)

        ax.set_ylabel(ylab)
        ax.set_ylim(-0.03, 1.06)
        ax.set_xlim(-0.010, 0.960)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])

    chance_a = K_TRUE / (30 * 29 / 2.0)
    ax_a.axhline(chance_a, color=GREY, lw=0.6, ls=(0, (1.2, 2.0)), zorder=1)
    ax_a.text(0.940, chance_a + 0.022, 'chance', fontsize=6.0, color=GREY,
              ha='right', va='bottom')
    ax_b.axhline(0.5, color=GREY, lw=0.6, ls=(0, (1.2, 2.0)), zorder=1)
    ax_b.text(0.940, 0.478, 'chance', fontsize=6.0, color=GREY,
              ha='right', va='top')

    # the one thing a reader cannot infer from a vertical bar: what moved along it
    ax_a.text(0.120, 0.078, '$\\lambda = 1 \\to 6$', fontsize=6.4, color=C_IID,
              ha='left', va='center')

    fig.text(0.53, 0.150, "measured Blomberg's $K$", ha='center', va='center',
             fontsize=8, color=INK)

    for ax, s in zip((ax_a, ax_b), 'ab'):
        ax.text(-0.150, 1.005, s, transform=ax.transAxes, fontsize=9.5, fontweight='bold',
                ha='left', va='bottom', color=INK)

    # two legends: mixing "which arm" and "which d" in one block scrambles the reading order,
    # because matplotlib fills a legend by columns and the rows then pair unrelated entries
    arms = [
        Line2D([], [], ls='none', marker='o', ms=3.7, color=C_SIG,
               label='signal, no confounder'),
        Line2D([], [], ls='none', marker='s', ms=3.2, color=C_TREE,
               label='confounder on the tree'),
        Line2D([], [], ls='none', marker='^', ms=4.0, color=C_IID,
               label='confounder, no tree ($K$ fixed)'),
    ]
    styles = [
        Line2D([], [], ls='-', lw=1.2, color=GREY, label='$d = 30$'),
        Line2D([], [], ls='--', lw=1.2, color=GREY, label='$d = 50$'),
    ]
    fig.legend(handles=arms, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.058),
               frameon=False, handlelength=1.4, columnspacing=2.6, handletextpad=0.5,
               fontsize=7)
    fig.legend(handles=styles, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.000),
               frameon=False, handlelength=1.9, columnspacing=2.0, handletextpad=0.5,
               fontsize=7)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.885, bottom=0.300, wspace=0.315)

    fig.savefig(os.path.join(FIG, 'fig3_dissociation.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG, 'fig3_dissociation.png'), dpi=400, bbox_inches='tight')
    plt.close(fig)
    print('fig3_dissociation written')


if __name__ == '__main__':
    main()
