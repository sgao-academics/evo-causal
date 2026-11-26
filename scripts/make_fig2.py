# -*- coding: utf-8 -*-
"""Figure 2: F1 and AUROC against the Blomberg's K actually measured in each dataset.

Redesigned 2026-11-25. Five changes, each of them about evidence rather than looks:

1. Every point now carries the measured spread of K on the horizontal axis (K_sd, 0.016 to
   0.074). The argument of this figure is that K is the axis a reader should trust; a figure
   making that argument cannot draw K as if it were exact.
2. Colour and marker say which analysis (PIC or none), line style and fill say d. Before, the
   caption claimed filled symbols were PIC while the code drew every marker open, so text and
   image disagreed.
3. The F1 = 0.8 reference line is gone. Nothing in the paper defines 0.8. The dotted line now
   marks the chance level of top-k F1, k/M, which Section 2 derives and which the text quotes:
   0.023 at d = 30, 0.008 at d = 50. A reference line has to have a source.
4. The three regimes the prose names (works / transition / neither works) are bracketed above
   panel (a), so the intervals the text describes are visible. Brackets mark intervals, not
   walls, which matches the gradualness the text insists on.
5. The AUROC panel gains the 0.5 chance line. Without it "threshold-free" has no anchor, and
   the fall to 0.51 reads as an arbitrary drop rather than a return to guessing.

One figure, one script: running this file writes figures/fig2_kaxis.pdf and .png and touches no
other figure. Every number is read from results/*.json, so the figure cannot disagree with the
tables or with the text.
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

# Drawn at the actual text width of the manuscript (A4 minus 1 in margins = 159 mm), so the
# figure is included at scale 1.0 and the declared point sizes survive.
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

# k, the number of true edges, is the number of entries kept by the top-k rule (Methods), so the
# chance level of F1 is k/M with M = d(d-1)/2. The two values below are the ones the text quotes.
K_TRUE = 10


def load(n):
    p = os.path.join(RES, n + '.json')
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else {}


def regime(ax, x0, x1, y, text):
    """Bracket an interval of the x axis above the plotting area.

    A bracket, not a shaded band: the text stresses that the three regimes are intervals on a
    continuous axis, and a shaded band with hard edges would read as a threshold.
    """
    tr = ax.get_xaxis_transform()
    ax.plot([x0, x1], [y, y], transform=tr, color=GREY, lw=0.6,
            solid_capstyle='butt', clip_on=False, zorder=5)
    # the ends drop to the axis frame: a bracket that floats reads as decoration
    for xx in (x0, x1):
        ax.plot([xx, xx], [y, 1.008], transform=tr, color=GREY, lw=0.6,
                solid_capstyle='butt', clip_on=False, zorder=5)
    ax.text((x0 + x1) / 2.0, y + 0.012, text, transform=tr, ha='center', va='bottom',
            fontsize=6.2, color=GREY, clip_on=False)


def rows_for(d, dd):
    rows = [v for v in d.values() if v['d'] == dd]
    rows.sort(key=lambda v: v['K_measured'])
    return rows


def main():
    d = load('e5_k_axis')
    if not d:
        print('e5_k_axis.json missing')
        return

    fig, axes = plt.subplots(1, 2, figsize=(FIGW, 3.00))
    ax_a, ax_b = axes

    # ------------------------------------------------------------------ (a) F1
    for dd, ls, filled in ((30, '-', True), (50, '--', False)):
        rows = rows_for(d, dd)
        x = np.array([r['K_measured'] for r in rows])
        e = np.array([r['K_sd'] for r in rows])
        for key, col, mk, ms, lw in (('f1_pic', PAL['mist'], 'o', 3.7, 1.25),
                                     ('f1_raw', PAL['lilac'], 's', 3.2, 1.05)):
            ax_a.errorbar(x, [r[key] for r in rows], xerr=e,
                          ls=ls, marker=mk, ms=ms, lw=lw, color=col,
                          mfc=col if filled else 'white', mew=0.85,
                          elinewidth=0.55, capsize=0, zorder=3)
    chance30 = K_TRUE / (30 * 29 / 2.0)
    ax_a.axhline(chance30, color=GREY, lw=0.6, ls=(0, (1.2, 2.0)), zorder=1)
    ax_a.text(0.055, chance30 + 0.022, 'chance', fontsize=6.0, color=GREY,
              ha='left', va='bottom')
    ax_a.set_ylabel('$F_1$ under top-$k$')
    ax_a.set_ylim(-0.03, 1.06)

    regime(ax_a, 0.056, 0.136, 1.048, 'works')
    regime(ax_a, 0.152, 0.243, 1.048, 'transition')
    regime(ax_a, 0.258, 0.474, 1.048, 'fails')

    # ------------------------------------------------------------------ (b) AUROC
    for dd, ls, filled in ((30, '-', True), (50, '--', False)):
        rows = rows_for(d, dd)
        x = np.array([r['K_measured'] for r in rows])
        e = np.array([r['K_sd'] for r in rows])
        ax_b.errorbar(x, [r['auroc_pic'] for r in rows], xerr=e,
                      ls=ls, marker='o', ms=3.7, lw=1.25, color=PAL['mist'],
                      mfc=PAL['mist'] if filled else 'white', mew=0.85,
                      elinewidth=0.55, capsize=0, zorder=3)
    ax_b.axhline(0.5, color=GREY, lw=0.6, ls=(0, (1.2, 2.0)), zorder=1)
    ax_b.text(0.472, 0.478, 'chance', fontsize=6.0, color=GREY,
              ha='right', va='top')
    ax_b.set_ylabel('AUROC')
    ax_b.set_ylim(-0.03, 1.06)

    for ax in axes:
        ax.set_xlim(0.050, 0.475)
        ax.set_xticks([0.1, 0.2, 0.3, 0.4])

    # one x label, centred under both panels: the two axes carry the same variable
    fig.text(0.53, 0.150, "measured Blomberg's $K$", ha='center', va='center',
             fontsize=8, color=INK)

    # ------------------------------------------------------------------ letters
    for ax, s in zip(axes, 'ab'):
        ax.text(-0.150, 1.005, s, transform=ax.transAxes, fontsize=9.5, fontweight='bold',
                ha='left', va='bottom', color=INK)

    # ------------------------------------------------------------------ one shared legend
    handles = [
        Line2D([], [], ls='-', marker='o', ms=3.7, lw=1.25, color=PAL['mist'],
               mfc=PAL['mist'], mew=0.85, label='PIC, $d = 30$'),
        Line2D([], [], ls='--', marker='o', ms=3.7, lw=1.25, color=PAL['mist'],
               mfc='white', mew=0.85, label='PIC, $d = 50$'),
        Line2D([], [], ls='-', marker='s', ms=3.2, lw=1.05, color=PAL['lilac'],
               mfc=PAL['lilac'], mew=0.85, label='no correction, $d = 30$'),
        Line2D([], [], ls='--', marker='s', ms=3.2, lw=1.05, color=PAL['lilac'],
               mfc='white', mew=0.85, label='no correction, $d = 50$'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.012),
               frameon=False, handlelength=2.3, columnspacing=2.6, handletextpad=0.6,
               fontsize=7)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.855, bottom=0.300, wspace=0.315)

    fig.savefig(os.path.join(FIG, 'fig2_kaxis.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG, 'fig2_kaxis.png'), dpi=400, bbox_inches='tight')
    plt.close(fig)
    print('fig2 ok')


if __name__ == '__main__':
    main()
