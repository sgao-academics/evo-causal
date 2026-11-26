# -*- coding: utf-8 -*-
"""Figure 4: the rank-one screen, and the deflation it motivates.

Redesigned 2026-11-25. What the old version got wrong:

1. Panel (b) had no error bars, although results/e16_deflate.json carries plain_sd and
   deflate_sd for every point and the spread reaches 0.20 at d = 50, lambda = 1.5. Without them
   "the deflated rule returns 0.81 against 0.51" reads as a sharper separation than ten seeds
   support, and the reader cannot tell where the two rules actually meet.
2. The caption said the triangles were "offset horizontally for legibility". Nothing in the
   code offset anything: the iid arm simply has fewer levels (1, 2, 4) than the tree arm
   (0.5, 1, 2, 3, 4, 6). A caption may not describe a transformation the figure does not perform.
3. Panel (b) carried an F1 = 0.8 line that the paper never defines, the same defect as in
   Figs. 2 and 3. It now marks the chance level k/M instead.
4. Colour carried d (blue = 30, green = 50), which collided with Fig. 3, where green is the
   signal arm. Colour now carries the rule -- grey for the analysis as it stands, blue for the
   deflated one -- and d is carried by line style and fill, as in Figs. 2 and 3.
5. The signal arm of (a) is the evidence that r and K point in opposite directions, but on a
   log axis spanning 2 to 300 it is flattened into a straight line. The lower edge of the null
   band is now drawn as a solid rule so that the fall below it is visible: r drops from 3.76 to
   2.37 at d = 30, which is where the claim that a rule in K alone would refuse data the search
   can handle comes from.

One figure, one script: running this file writes figures/fig4_remedy.pdf and .png and touches no
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

C_SIG = PAL['moss']      # a per-trait Brownian component, no confounder -- the control arm
C_TREE = PAL['violet']   # confounder that evolves on the tree
C_IID = PAL['orchid']    # confounder with no phylogenetic structure
LILAC = PAL['lilac']     # the analysis as it stands -- baseline, de-emphasised
K_TRUE = 10


def load(n):
    p = os.path.join(RES, n + '.json')
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else {}


def main():
    sc = load('e15_screen')
    scb = load('e15b_screen_signal')
    de = load('e16_deflate')
    if not sc or not de:
        print('missing e15/e16')
        return

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(FIGW, 2.95))

    # ------------------------------------------------------------------ (a) the screen
    q05 = min(sc['null_d30']['q05'], sc['null_d50']['q05'])
    q95 = max(sc['null_d30']['q95'], sc['null_d50']['q95'])
    ax_a.axhspan(q05, q95, color=LILAC, alpha=0.42, lw=0, zorder=1)
    for yy in (q05, q95):
        ax_a.axhline(yy, color=LILAC, lw=0.85, zorder=2)

    arms = (('tree', 's', C_TREE, 'confounder on the tree', False),
            ('iid', '^', C_IID, 'confounder, no tree', False),
            ('signal', 'o', C_SIG, 'signal only', True))
    for d, ls, filled in ((30, '-', True), (50, '--', False)):
        for arm, mk, col, name, from_sig in arms:
            src = scb if from_sig else sc
            lv, rr = [], []
            for k, v in src.items():
                if not isinstance(v, dict) or v.get('d') != d:
                    continue
                if not from_sig and v.get('latent') != arm:
                    continue
                lv.append(v['level'])
                rr.append(v['r'])
            if not lv:
                continue
            o = np.argsort(lv)
            ax_a.plot(np.array(lv)[o], np.array(rr)[o], ls=ls, marker=mk, ms=3.4, lw=1.15,
                      color=col, mfc=col if filled else 'white', mew=0.85, zorder=4)

    ax_a.set_yscale('log')
    ax_a.set_ylim(2.0, 330)
    # explicit ticks: the decade labels alone leave the 2-10 stretch, which is where the null
    # band and the whole signal arm live, without a single readable reference
    ax_a.set_yticks([3, 10, 30, 100])
    ax_a.set_yticklabels(['3', '10', '30', '100'])
    ax_a.set_xlim(-0.28, 6.40)
    ax_a.set_xticks([0, 1, 2, 3, 4, 6])
    ax_a.set_xlabel('confounding load $\\lambda$')
    ax_a.set_ylabel('leading / rest eigenvalue')
    # inside the band, at the right, where neither arm has a point
    ax_a.text(6.32, (q05 * q95) ** 0.5, '95% null', fontsize=6.0, color=GREY,
              ha='right', va='center')

    handles_a = [Line2D([], [], ls='none', marker=mk, ms=3.4, color=col, label=name)
                 for arm, mk, col, name, _ in arms]
    ax_a.legend(handles=handles_a, loc='upper left', fontsize=6.2, handletextpad=0.45,
                labelspacing=0.3, borderpad=0.25, handlelength=1.0)

    # ------------------------------------------------------------------ (b) the deflation
    for d, ls, filled in ((30, '-', True), (50, '--', False)):
        rows = [(v['lam'], v['plain'], v['plain_sd'], v['deflate'], v['deflate_sd'])
                for v in de.values()
                if isinstance(v, dict) and v.get('d') == d and 'deflate' in v]
        if not rows:
            continue
        rows.sort()
        lam = np.array([r[0] for r in rows])
        for i, col in ((1, LILAC), (3, PAL['mist'])):
            y = np.array([r[i] for r in rows])
            e = np.array([r[i + 1] for r in rows])
            ax_b.errorbar(lam, y, yerr=e, fmt='none', ecolor=col, elinewidth=0.5,
                          capsize=0, alpha=0.38, zorder=2)
            ax_b.plot(lam, y, ls=ls, marker='o', ms=3.6, lw=1.25, color=col,
                      mfc=col if filled else 'white', mew=0.85, zorder=4)

    chance = K_TRUE / (30 * 29 / 2.0)
    ax_b.axhline(chance, color=GREY, lw=0.6, ls=(0, (1.2, 2.0)), zorder=1)
    ax_b.text(-0.05, chance * 2.6, 'chance', fontsize=6.0, color=GREY, ha='left', va='bottom')
    ax_b.set_xlim(-0.28, 6.40)
    ax_b.set_xticks([0, 1, 2, 3, 4, 6])
    ax_b.set_ylim(-0.03, 1.05)
    ax_b.set_xlabel('confounding load $\\lambda$')
    ax_b.set_ylabel('$F_1$ under top-$k$')

    handles_b = [
        Line2D([], [], ls='-', marker='o', ms=3.6, lw=1.25, color=LILAC,
               mfc=LILAC, mew=0.85, label='as is, $d = 30$'),
        Line2D([], [], ls='--', marker='o', ms=3.6, lw=1.25, color=LILAC,
               mfc='white', mew=0.85, label='as is, $d = 50$'),
        Line2D([], [], ls='-', marker='o', ms=3.6, lw=1.25, color=PAL['mist'],
               mfc=PAL['mist'], mew=0.85, label='component removed, $d = 30$'),
        Line2D([], [], ls='--', marker='o', ms=3.6, lw=1.25, color=PAL['mist'],
               mfc='white', mew=0.85, label='component removed, $d = 50$'),
    ]
    fig.legend(handles=handles_b, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.012),
               frameon=False, handlelength=2.3, columnspacing=2.6, handletextpad=0.6,
               fontsize=7)

    for ax, s in zip((ax_a, ax_b), 'ab'):
        ax.text(-0.150, 1.005, s, transform=ax.transAxes, fontsize=9.5, fontweight='bold',
                ha='left', va='bottom', color=INK)

    fig.subplots_adjust(left=0.078, right=0.985, top=0.885, bottom=0.300, wspace=0.315)
    fig.savefig(os.path.join(FIG, 'fig4_remedy.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG, 'fig4_remedy.png'), dpi=400, bbox_inches='tight')
    plt.close(fig)
    print('fig4_remedy written')


if __name__ == '__main__':
    main()
