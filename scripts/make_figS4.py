# -*- coding: utf-8 -*-
"""Figure S4: phylogenetic path analysis in two candidate regimes. A zero-height bar is a silent
failure, not a small effect.

Data source is results/e17_ppa_order.json and nothing else. The earlier version of this figure
read results/e7_ppa.json, which is a different run: three seeds instead of five, candidates
enumerated over the generating order (1,024) instead of all orders (29,281 and 1,141). Its values
therefore disagreed with Table 3 at lambda=0 (0.667 against 0.578) and with the contrast column
at lambda=2 (0.722 against 0.800). The figure now shows the same three columns the table reports,
minus the order-supplied column, which is left in the table because it comes from that other run.
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

try:
    from palette import PAL, INK, GREY
except Exception:
    PAL = {'mist': '#4D6EAF', 'peri': '#7A9AC3', 'lilac': '#B0B9CB',
           'orchid': '#D0A0B6', 'moss': '#8FB578'}
    INK, GREY = '#222222', '#8A8A8A'

# 159.2 mm: the figure is included at \textwidth, so this is the width it is printed at
FIGW = 159 / 25.4
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': INK, 'axes.linewidth': 0.7,
    'xtick.color': INK, 'ytick.color': INK, 'axes.labelcolor': INK,
    'text.color': INK, 'legend.frameon': False,
})


def rows(regime):
    d = json.load(open(os.path.join(RES, 'e17_ppa_order.json'), encoding='utf-8'))
    r = [v for k, v in d.items() if isinstance(v, dict) and v.get('regime') == regime]
    return sorted(r, key=lambda v: v['lam'])


def letter(ax, s, dx=-0.17, dy=1.09):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight='bold',
            va='top', ha='left')


def main():
    full = rows('full')
    restr = rows('restricted')
    lam = [r['lam'] for r in full]
    w = 0.30
    xl = np.arange(len(lam))

    fig, axes = plt.subplots(1, 2, figsize=(FIGW, 2.44), gridspec_kw={'wspace': 0.32})
    ax = axes[0]
    b1 = ax.bar(xl - w / 2, [r['ppa_f1'] for r in full], w * 0.92, color=PAL['moss'],
                label='PPA, every order admitted')
    b2 = ax.bar(xl + w / 2, [r['ppa_f1'] for r in restr], w * 0.92, color=PAL['lilac'],
                label='PPA, at most three edges')
    # the contrast rule sits just outside the pair, so the comparison is read across a group
    # rather than the point being hidden between two bars
    ln, = ax.plot(xl + 0.50, [r['pic_f1'] for r in full], '-D', ms=3.6, lw=1.0,
                  color=PAL['mist'], mfc=PAL['mist'], mec='white', mew=0.6,
                  label='contrasts + correlation', clip_on=False)
    ax.set_xlabel(r'latent load $\lambda$')
    ax.set_ylabel(r'$F_1$')
    ax.set_xticks(xl); ax.set_xticklabels(['%g' % v for v in lam])
    ax.set_xlim(-0.55, len(lam) - 0.32)
    ax.set_ylim(0, 1.02)

    ax = axes[1]
    ax.bar(xl - w / 2, [r['ppa_exact'] for r in full], w * 0.92, color=PAL['moss'],
           label='every order admitted')
    ax.bar(xl + w / 2, [r['ppa_exact'] for r in restr], w * 0.92, color=PAL['lilac'],
           label='at most three edges')
    # label only what the bar cannot show by itself: the one non-zero rate, and every zero on
    # the restricted regime, which is the silent failure the panel is about
    for i, r in enumerate(full):
        if r['ppa_exact'] > 0:
            ax.text(xl[i] - w / 2, r['ppa_exact'] + 0.012, '%.2f' % r['ppa_exact'],
                    ha='center', va='bottom', fontsize=6.2, color=INK, fontweight='bold')
    for i, r in enumerate(restr):
        ax.text(xl[i] + w / 2, 0.012, '%.2f' % r['ppa_exact'], ha='center', va='bottom',
                fontsize=6.2, color=INK)
    ax.set_xlabel(r'latent load $\lambda$')
    ax.set_ylabel('exact DAG recovery rate')
    ax.set_xticks(xl); ax.set_xticklabels(['%g' % v for v in lam])
    ax.set_xlim(-0.62, len(lam) - 0.38)
    ax.set_ylim(0, 1.02)
    ax.text(0.5, 0.97, 'a zero-height bar is a silent failure',
            transform=ax.transAxes, ha='center', va='top', fontsize=6.3, color=GREY)

    # legend order follows the reading order on the page: green bar, grey bar, blue point
    fig.legend([b1, b2, ln], [b1.get_label(), b2.get_label(), ln.get_label()],
               loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.105),
               frameon=False, handlelength=1.8, columnspacing=1.6,
               handletextpad=0.5, fontsize=7)
    letter(axes[0], 'a'); letter(axes[1], 'b')
    # left/right fixed so the tight bounding box lands on the printed width; see make_figS1.py
    fig.subplots_adjust(left=0.075, right=0.985, top=0.905, bottom=0.245)
    fig.savefig(os.path.join(FIG, 'figS4_ppa.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG, 'figS4_ppa.png'), dpi=400, bbox_inches='tight')
    plt.close(fig)
    print('figS4_ppa ok')


if __name__ == '__main__':
    main()
