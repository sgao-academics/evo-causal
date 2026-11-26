# -*- coding: utf-8 -*-
"""Figure S3: the nonlinear additive-noise family against the two linear rules it could in
principle replace.

Two panels, one axis each, because the family's two metrics say different things: the
threshold-free ranking is flat and the top-k accuracy is low and also flat. Plotting F1 alone
would suggest the family is simply worse; the AUROC panel shows why, which is that the ranking
is insensitive to the confounder and the top-k rule cannot exploit it.

The two rules are measured on ten seeds (results/e23_families10.json) and the family on five
(results/e29_resit.json); the caption states both. NOTEARS enters the AUROC panel from
results/e23b_notears_auroc.json, which recomputes the same grid and asserts its correlation
column against E23 before the numbers are used.
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

try:
    from palette import PAL, INK, GREY
except Exception:
    PAL = {'mist': '#4D6EAF', 'peri': '#7A9AC3', 'lilac': '#B0B9CB',
           'orchid': '#D0A0B6', 'moss': '#8FB578', 'violet': '#9A5A9F'}
    INK, GREY = '#222222', '#8A8A8A'

# 124.2 mm = 0.78 \textwidth, the width the page actually gives this figure
FIGW = 159.2 * 0.78 / 25.4
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': INK, 'axes.linewidth': 0.7,
    'xtick.color': INK, 'ytick.color': INK, 'axes.labelcolor': INK,
    'text.color': INK, 'legend.frameon': False,
})


def series(f, field, sd=False):
    xs, ys, es = [], [], []
    for k in ('lam0.0', 'lam0.5', 'lam1.0', 'lam1.5', 'lam2.0', 'lam3.0'):
        if k in f:
            xs.append(f[k]['lam'])
            ys.append(f[k][field])
            if sd:
                es.append(f[k].get(field + '_sd', 0.0))
    o = np.argsort(xs)
    return np.array(xs)[o], np.array(ys)[o], np.array(es)[o] if es else None


def load(n):
    p = os.path.join(RES, n + '.json')
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else {}


def letter(ax, s, dx=-0.17, dy=1.09):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight='bold',
            va='top', ha='left')


def main():
    e23 = load('e23_families10')
    e29 = load('e29_resit')
    e23b = load('e23b_notears_auroc')

    fig, axes = plt.subplots(1, 2, figsize=(FIGW, 2.46), gridspec_kw={'wspace': 0.33})
    ax = axes[0]
    h = []
    x, y, s = series(e23, 'corr', sd=True)
    h.append(ax.errorbar(x, y, yerr=s, fmt='-o', ms=3.5, lw=1.05, capsize=1.8,
                         color=PAL['mist'], mfc=PAL['mist'], mec='white', mew=0.5,
                         label='correlation'))
    x, y, s = series(e23, 'notears', sd=True)
    h.append(ax.errorbar(x, y, yerr=s, fmt='-s', ms=3.3, lw=1.05, capsize=1.8,
                         color=PAL['moss'], mfc=PAL['moss'], mec='white', mew=0.5,
                         label='NOTEARS'))
    x, y, s = series(e29, 'f1', sd=True)
    h.append(ax.errorbar(x, y, yerr=s, fmt='-^', ms=3.9, lw=1.15, capsize=1.8,
                         color=PAL['orchid'], mfc=PAL['orchid'], mec='white', mew=0.5,
                         label='RESIT (nonlinear)'))
    ax.set_xlabel(r'confounding load $\lambda$')
    ax.set_ylabel(r'$F_1$')
    ax.set_ylim(0, 1.02)
    letter(ax, 'a')

    ax = axes[1]
    x, y, _ = series(e23, 'auroc')
    ax.plot(x, y, '-o', ms=3.5, lw=1.05, color=PAL['mist'], mfc=PAL['mist'],
            mec='white', mew=0.5, label='correlation')
    if e23b:
        x, y, _ = series(e23b, 'notears_auroc')
        ax.plot(x, y, '-s', ms=3.3, lw=1.05, color=PAL['moss'], mfc=PAL['moss'],
                mec='white', mew=0.5, label='NOTEARS')
    x, y, _ = series(e29, 'auroc')
    ax.plot(x, y, '-^', ms=3.9, lw=1.15, color=PAL['orchid'], mfc=PAL['orchid'],
            mec='white', mew=0.5, label='RESIT (nonlinear)')
    ax.set_xlabel(r'confounding load $\lambda$')
    ax.set_ylabel('AUROC')
    ax.set_ylim(0.68, 1.03)
    letter(ax, 'b')

    # one legend under both panels: inside panel (a) it sat on the correlation curve
    fig.legend(h, [a.get_label() for a in h], loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, 0.105), frameon=False, handlelength=1.8,
               columnspacing=1.8, handletextpad=0.5, fontsize=7)
    # left/right fixed so the tight bounding box lands on the printed width; see make_figS1.py
    fig.subplots_adjust(left=0.085, right=0.985, top=0.905, bottom=0.245)
    out = os.path.join(FIG, 'figS3_resit.pdf')
    fig.savefig(out, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(out.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print('figS3_resit ok')


if __name__ == '__main__':
    main()
