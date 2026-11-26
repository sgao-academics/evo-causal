# -*- coding: utf-8 -*-
"""The real-data figure: where five published comparative matrices sit.

Redesigned 2026-11-25. What the old version got wrong:

1. Panel (a) drew the null as a single grey tick, the mean of a sparse random DAG. The paper
   quotes "2.28 +- 0.40" and results/e30_holm.json carries null_sd for every block, so the
   spread existed and was not shown. A reference with no width makes a ratio of 3.99 look
   further outside it than it is, and makes "sits below the reference" unreadable.
2. The caption said "0.20 is what an even spread over ten edges would give". That number cannot
   be right: the five blocks have d = 6, 12, 10, 9, 6, and a top-d ranking keeps d edges, so an
   even spread gives 1/d, which is 0.167 at d = 6 and 0.083 at d = 12. Nothing gives 0.20. The
   panel now shades the interval 1/d actually spans (0.083 to 0.167), which is what the reader
   needs in order to judge by how much the ranking concentrates.
3. The legend mixed five long labels into three columns, which pairs them by column and scrambles
   the reading order. Labels are now short and sit on one row; the mammal or bird provenance
   moves into the caption.

Every number is read from results/e26_real.json, results/e27_birds.json and results/e30_holm.json,
so the figure cannot disagree with the table or with the text.
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
    from palette import PAL, INK, GREY, TICK
except Exception:
    PAL = {'mist': '#4D6EAF', 'peri': '#7A9AC3', 'lilac': '#B0B9CB',
           'orchid': '#D0A0B6', 'moss': '#8FB578', 'violet': '#9A5A9F',
           'indigo': '#3F3A5B'}
    INK, GREY, TICK = '#3F3A5B', '#8A93A6', '#5B6478'

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

LILAC = PAL['lilac']
ORDER = ['life_history', 'range_climate', 'bird_diet', 'bird_niche', 'ecology']
YLBL = {'life_history': 'life history', 'range_climate': 'range and climate',
        'bird_diet': 'bird diet', 'bird_niche': 'bird niche', 'ecology': 'ecology'}
COLOR = {'life_history': PAL['moss'], 'range_climate': PAL['orchid'],
         'bird_diet': PAL['peri'], 'bird_niche': PAL['violet'], 'ecology': PAL['mist']}


def load(n):
    p = os.path.join(RES, n + '.json')
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else {}


def main():
    H = load('e30_holm')
    M = load('e26_real')
    B = load('e27_birds')
    blocks = H.get('blocks', {})
    keys = [k for k in ORDER if k in blocks]
    if not keys:
        print('e30_holm missing')
        return

    fig, axs = plt.subplots(1, 3, figsize=(FIGW, 2.75),
                            gridspec_kw=dict(width_ratios=[1.32, 1.00, 0.78], wspace=0.30))

    # ---------------------------------------------------------------- (a) r1 against the reference
    ax = axs[0]
    for i, k in enumerate(keys):
        b = blocks[k]
        m = b['null_mean'][0]
        sd = b['null_sd'][0]
        # the reference with its width: a bare tick would hide how far 3.99 really is
        ax.plot([m - sd, m + sd], [i, i], color=LILAC, lw=2.6,
                solid_capstyle='butt', zorder=1)
        ax.plot([m], [i], marker='|', ms=6.5, mew=1.0, color=GREY, zorder=2)
        ax.plot([b['obs'][0]], [i], 'o', ms=6.2, mfc=COLOR[k], mec='white', mew=0.75, zorder=3)
        ph = H['holes'].get('%s:r1' % k, {})
        # the adjusted probability sits in a fixed column, not next to each point: parked beside
        # the marker it read as part of the position, and a reader takes a dot at 3.29 with a
        # label starting at 3.47 for a dot at 3.47
        ax.annotate('%.3f' % ph.get('holm', np.nan), (4.28, i), va='center', ha='left',
                    fontsize=6.6, color=INK if ph.get('survives') else GREY,
                    fontweight='bold' if ph.get('survives') else 'normal')
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([YLBL[k] for k in keys], fontsize=7.2)
    ax.set_ylim(len(keys) - 0.45, -0.75)
    ax.set_xlabel('leading ratio $r_1$')
    ax.set_xlim(0.7, 5.15)
    ax.set_xticks([1, 2, 3, 4])
    ax.text(4.28, -0.62, 'Holm', fontsize=6.4, color=GREY, ha='left', va='center')
    ax.axvline(1.0, color=LILAC, lw=0.7, ls=(0, (2, 2)), zorder=1)
    ax.text(-0.185, 1.005, 'a', transform=ax.transAxes, fontsize=9.5, fontweight='bold',
            ha='left', va='bottom', color=INK)

    # ---------------------------------------------------------------- (b) the three ratios
    ax = axs[1]
    ks = np.array([1, 2, 3])
    nullm = np.array([blocks[k]['null_mean'] for k in keys]).mean(0)
    # the dashed line is a mean over five blocks, so it gets no error bar: averaging five
    # standard deviations and hanging them on the mean would read as a confidence interval of
    # the average, which is not what the data are
    ax.plot(ks, nullm, '--', color=LILAC, lw=1.5, zorder=2)
    for k in keys:
        ax.plot(ks, blocks[k]['obs'], '-o', ms=3.8, lw=1.15, color=COLOR[k],
                mfc=COLOR[k], mec='white', mew=0.55, zorder=3)
    ax.set_xticks(ks)
    ax.set_xticklabels([r'$r_1$', r'$r_2$', r'$r_3$'])
    ax.set_xlim(0.78, 3.22)
    ax.set_ylim(0.95, 4.95)
    ax.set_ylabel('ratio')
    ax.set_xlabel('index of the leading ratio')
    ax.text(-0.135, 1.005, 'b', transform=ax.transAxes, fontsize=9.5, fontweight='bold',
            ha='left', va='bottom', color=INK)

    # ---------------------------------------------------------------- (c) what the ranking sees
    ax = axs[2]
    dfs = []
    for i, k in enumerate(keys):
        src = (M if k in M else B).get(k, {})
        dfs.append(src.get('d', 0))
        share = src.get('hub', {}).get('share', np.nan)
        ax.barh(i, share, height=0.55, color=COLOR[k], edgecolor='white', linewidth=0.6,
                zorder=2)
        ax.text(share + 0.014, i, '%.2f' % share, va='center', ha='left', fontsize=6.6,
                color=INK)
    # a top-d ranking keeps d edges, so an even spread puts 1/d at one trait; d differs per block
    lo, hi = min(1.0 / x for x in dfs if x), max(1.0 / x for x in dfs if x)
    ax.axvspan(lo, hi, color=LILAC, alpha=0.30, lw=0, zorder=1)
    for xx in (lo, hi):
        ax.axvline(xx, color=LILAC, lw=0.9, zorder=1)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([])
    ax.set_ylim(len(keys) - 0.45, -0.95)
    ax.set_xlim(0, 0.46)
    # the band is too narrow to carry a legible label inside it, so it is named by the axis
    ax.set_xlabel('edges at one trait\n(grey band: $1/d$)')
    ax.set_xticks([0, 0.2, 0.4])
    ax.text(-0.315, 1.005, 'c', transform=ax.transAxes, fontsize=9.5, fontweight='bold',
            ha='left', va='bottom', color=INK)

    handles = [Line2D([], [], ls='-', marker='o', ms=3.8, lw=1.15, color=COLOR[k],
                      mfc=COLOR[k], mec='white', mew=0.55, label=YLBL[k]) for k in keys]
    fig.legend(handles=handles, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.030),
               frameon=False, handlelength=1.5, columnspacing=1.6, handletextpad=0.45,
               fontsize=7)

    fig.subplots_adjust(left=0.115, right=0.985, top=0.885, bottom=0.315)
    fig.savefig(os.path.join(FIG, 'fig5_real.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG, 'fig5_real.png'), dpi=400, bbox_inches='tight')
    plt.close(fig)
    print('fig5_real written')


if __name__ == '__main__':
    main()
