# -*- coding: utf-8 -*-
"""Figure S1: the evaluation rule isolated. No confounding is present and no correction is
applied, so the only variable is the number of traits.

One figure, one script: running this file writes figures/figS1_rule.pdf and .png and touches no
other figure. Every number is read from results/e10_threshold.json, so the figure cannot disagree
with the tables or with the text.

Two things this version does that the previous one did not.

1. Colour now carries the rule and nothing else: orchid = the 99th-percentile rule, mist = top-k.
   The old version painted the "true edges" line moss in panel (a) and the top-k curve moss in
   panel (b), so one colour stood for the truth in one panel and for a rule in the other. A single
   two-entry legend now serves both panels, because the y-axis says which quantity is shown.

2. The percentile rule turns out to be *exactly* deterministic: its admitted count is
   0.01*d(d-1)/2 by construction, and since the correlation ranking at lambda = 0 is essentially
   perfect the ten true edges are always inside the kept set, so F1 = 2*min(k,k99)/(k+k99) with no
   seed-to-seed spread at all (sd = 0.0000 at every d, three seeds). The curve that does move is
   top-k, by +-0.058 at the two smallest d. Only that curve therefore carries an error bar; putting
   a zero-length bar on the other would suggest a measurement where there is arithmetic.
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
    PAL = {'mist': '#4D6EAF', 'orchid': '#D0A0B6'}
    INK, GREY = '#3F3A5B', '#8A93A6'

C_RULE = PAL['orchid']      # the 99th-percentile rule -- the thing under indictment
C_TOPK = PAL['mist']        # top-k, the choice the paper makes

# Drawn at the actual text width of the manuscript (A4 minus 1 in margins = 159 mm), so the
# figure is included at scale 1.0 and the declared point sizes survive.
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


def load(n):
    p = os.path.join(RES, n + '.json')
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else {}


def letter(ax, s, dx=-0.135, dy=1.075):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight='bold',
            va='top', ha='left')


def main():
    d = load('e10_threshold')
    rows = sorted(d.values(), key=lambda v: v['d'])
    x = np.array([r['d'] for r in rows], float)
    k99 = np.array([r['k_p99'] for r in rows])
    ktr = np.array([r['k_true'] for r in rows])
    f99 = np.array([r['f1_p99'] for r in rows])
    ftk = np.array([r['f1_topk'] for r in rows])

    # the seed spread, recomputed per seed because e10_threshold.json holds means only;
    # scripts/e10_threshold_seeds.py writes this file and checks the means against that one
    seeds = load('e10_threshold_seeds')
    sd_tk = np.array([seeds['d%d' % int(v)]['f1_topk_sd'] for v in x]) if seeds else np.zeros_like(x)
    sd_99 = np.array([seeds['d%d' % int(v)]['f1_p99_sd'] for v in x]) if seeds else np.zeros_like(x)

    fig, axes = plt.subplots(1, 2, figsize=(FIGW, 2.42), gridspec_kw={'wspace': 0.30})
    # left/right are set so the content spans the full 159 mm canvas: with the default margins
    # the tight bounding box came out 136 mm wide, and \includegraphics[width=\textwidth] then
    # blew the figure up by 1.17, so the 8 pt labels were printed at 9.4 pt
    fig.subplots_adjust(left=0.062, right=0.987, top=0.900, bottom=0.255)

    # ---- (a) how many edges each rule admits ----------------------------------------------
    ax = axes[0]
    ax.plot(x, k99, '-', marker='s', ms=3.4, lw=1.15, color=C_RULE,
            mfc=C_RULE, mew=0.7, label='99th percentile', zorder=4)
    # top-k admits exactly k edges -- the same number as the truth, by construction
    ax.plot(x, ktr, '-', marker='o', ms=3.4, lw=1.15, color=C_TOPK,
            mfc=C_TOPK, mew=0.7, label='top-$k$', zorder=4)
    ax.set_xlabel('number of traits $d$')
    ax.set_ylabel('edges admitted')
    ax.set_xticks([30, 50, 100, 150, 200])
    ax.set_xlim(15, 208)
    ax.set_ylim(0, 215)
    ax.set_yticks([0, 50, 100, 150, 200])
    # the flat line is the truth as well as the top-k count; say it once, in grey
    ax.text(118, 22, 'ten true edges', fontsize=6.2, color=GREY, ha='left', va='bottom')
    ax.text(150, 150, '$0.01\\,d(d-1)/2$', fontsize=6.4, color=C_RULE,
            ha='right', va='bottom', rotation=20)

    # ---- (b) what that does to F1 ---------------------------------------------------------
    ax = axes[1]
    # no error bar on this curve: its spread is exactly zero, and a zero-length bar on a line
    # that is arithmetic rather than measured would imply a measurement that does not exist
    ax.plot(x, f99, '-', marker='s', ms=3.4, lw=1.15, color=C_RULE,
            mfc=C_RULE, mew=0.7, zorder=4)
    ax.errorbar(x, ftk, yerr=sd_tk, fmt='none', ecolor=C_TOPK, elinewidth=0.55,
                capsize=1.6, capthick=0.55, zorder=2)
    ax.plot(x, ftk, '-', marker='o', ms=3.4, lw=1.15, color=C_TOPK,
            mfc=C_TOPK, mew=0.7, zorder=4)
    ax.set_xlabel('number of traits $d$')
    ax.set_ylabel('$F_1$')
    ax.set_ylim(-0.04, 1.10)
    ax.set_xticks([30, 50, 100, 150, 200])
    ax.set_xlim(15, 208)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # the honest value: 0.9995 at d=30 and 0.9999 at d=50 are not 1.000
    # 0.93 not 0.97: at 0.97 the closing italic d sat on the right spine
    ax.text(0.93, 0.055, 'AUROC $\\geq 0.9995$ at every $d$', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=6.3, color=GREY)

    letter(axes[0], 'a')
    letter(axes[1], 'b')

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.035),
               frameon=False, handlelength=2.0, columnspacing=3.2,
               handletextpad=0.55, fontsize=7)

    fig.savefig(os.path.join(FIG, 'figS1_rule.pdf'), bbox_inches='tight', pad_inches=0.02)
    fig.savefig(os.path.join(FIG, 'figS1_rule.png'), dpi=400, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print('figS1_rule ok')
    for r in rows:
        print('  d=%3d  k99=%5.0f  k_true=%2.0f  F1@p99=%.4f  F1@topk=%.4f+-%.4f  AUROC=%.4f'
              % (r['d'], r['k_p99'], r['k_true'], r['f1_p99'], r['f1_topk'],
                 seeds.get('d%d' % int(r['d']), {}).get('f1_topk_sd', 0.0), r['auroc']))


if __name__ == '__main__':
    main()
