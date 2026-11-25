# -*- coding: utf-8 -*-
"""The real-data figure: where five published comparative matrices sit.

Reads results/e26_real.json, results/e27_birds.json and results/e30_holm.json only, so the
figure cannot disagree with the table or with the text.

(a) the leading ratio against the reference an ordinary sparse causal graph produces (grey
    tick), with the Holm-adjusted p-value beside the observed value (bold when it survives);
(b) the three leading ratios against the same reference for each index: a line that rises from
    the first to the second index is a matrix whose excess is not a single direction;
(c) the share of admitted edges incident to the variable the ranking collapses onto.
Colours identify blocks throughout and are named once, under the figure.
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
           'orchid': '#D0A0B6', 'moss': '#8FB578', 'violet': '#9A5A9F'}
    INK, GREY = '#222222', '#8A8A8A'

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

LILAC = PAL.get('lilac', '#B0B9CB')
ORDER = ['life_history', 'range_climate', 'bird_diet', 'bird_niche', 'ecology']
SHORT = {'life_history': 'life history (mammals)',
         'range_climate': 'range and climate (mammals)',
         'bird_diet': 'bird diet', 'bird_niche': 'bird niche',
         'ecology': 'ecology (mammals)'}
YLBL = {'life_history': 'life history',
        'range_climate': 'range and climate',
        'bird_diet': 'bird diet', 'bird_niche': 'bird niche',
        'ecology': 'ecology'}
COLOR = {'life_history': PAL['moss'], 'range_climate': PAL['orchid'],
         'bird_diet': PAL['peri'], 'bird_niche': PAL['violet'],
         'ecology': PAL['mist']}


def load(n):
    p = os.path.join(RES, n + '.json')
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else {}


def letter(ax, s, dx=-0.15, dy=1.09):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight='bold',
            va='top', ha='left')


def main():
    H = load('e30_holm')
    M = load('e26_real'); B = load('e27_birds')
    blocks = H.get('blocks', {})
    keys = [k for k in ORDER if k in blocks]

    fig, axs = plt.subplots(1, 3, figsize=(FIGW, 2.62),
                            gridspec_kw=dict(width_ratios=[1.35, 1.02, 0.80], wspace=0.34))
    ax = axs[0]

    # ---------------------------------------------------------------- (a) r1 against the reference
    for i, k in enumerate(keys):
        b = blocks[k]
        ax.plot([b['null_mean'][0]], [i], marker='|', ms=13, mew=1.3, color=LILAC, zorder=2)
        ax.plot([b['obs'][0]], [i], 'o', ms=6, mfc=COLOR[k], mec='white', mew=0.7, zorder=3)
        ph = H['holes'].get('%s:r1' % k, {})
        ax.annotate('%.2f' % ph.get('holm', np.nan),
                    (max(b['obs'][0], b['null_mean'][0]) + 0.16, i),
                    va='center', ha='left', fontsize=6.6,
                    color=INK if ph.get('survives') else GREY,
                    fontweight='bold' if ph.get('survives') else 'normal')
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([YLBL[k] for k in keys], fontsize=7)
    ax.set_ylim(len(keys) - 0.45, -0.75)
    ax.set_xlabel(r'leading ratio $r_1$')
    ax.set_xlim(0.8, 4.9)
    ax.axvline(1.0, color=LILAC, lw=0.7, ls=(0, (2, 2)), zorder=1)
    letter(ax, 'a')

    # ---------------------------------------------------------------- (b) the three ratios
    ax = axs[1]
    ks = np.array([1, 2, 3])
    nullm = np.array([blocks[k]['null_mean'] for k in keys]).mean(0)
    for k in keys:
        b = blocks[k]
        ax.plot(ks, b['obs'], '-o', ms=3.8, lw=1.1, color=COLOR[k], mfc=COLOR[k],
                mec='white', mew=0.5, label=SHORT[k])
    ax.plot(ks, nullm, '--', color=LILAC, lw=1.4, zorder=1)
    ax.set_xticks(ks)
    ax.set_xticklabels([r'$r_1$', r'$r_2$', r'$r_3$'])
    ax.set_xlim(0.75, 3.25)
    ax.set_ylim(1.0, 4.85)
    ax.set_ylabel('ratio')
    ax.set_xlabel('index of the leading ratio')
    letter(ax, 'b')

    # ---------------------------------------------------------------- (c) what the ranking sees
    ax = axs[2]
    for i, k in enumerate(keys):
        src = (M if k in M else B).get(k, {})
        share = src.get('hub', {}).get('share', np.nan)
        ax.barh(i, share, height=0.55, color=COLOR[k], edgecolor='white', linewidth=0.6)
        ax.text(share + 0.016, i, '%.2f' % share, va='center', ha='left', fontsize=6.6,
                color=INK)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([])
    ax.set_ylim(len(keys) - 0.45, -0.75)
    ax.set_xlim(0, 0.47)
    ax.set_xlabel('edges at one trait')
    ax.set_xticks([0, 0.2, 0.4])
    letter(ax, 'c', dx=-0.30)

    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.21),
               frameon=False, handlelength=1.6, columnspacing=1.4,
               handletextpad=0.5, fontsize=6.8)

    out = os.path.join(FIG, 'fig8_real.pdf')
    fig.savefig(out, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(out.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', pad_inches=0.02)
    print('wrote', out)


if __name__ == '__main__':
    main()
