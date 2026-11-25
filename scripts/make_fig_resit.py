# -*- coding: utf-8 -*-
"""Figure S3: the nonlinear additive-noise family against the linear rules.

Reads results/e23_families10.json and results/e29_resit.json. Two panels, one axis each, because
the family's two metrics say different things: the threshold-free ranking is flat and the top-k
accuracy is low and also flat. Plotting F1 alone would suggest the family is simply worse; the
AUROC panel shows why, which is that the ranking is insensitive to the confounder and the top-k
rule cannot exploit it.
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
LILAC = PAL.get('lilac', '#B0B9CB')


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


def letter(ax, s, dx=-0.15, dy=1.08):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight='bold', va='top', ha='left')


def main():
    e23 = json.load(open(os.path.join(RES, 'e23_families10.json'), encoding='utf-8'))
    e29 = json.load(open(os.path.join(RES, 'e29_resit.json'), encoding='utf-8'))

    fig, axs = plt.subplots(1, 2, figsize=(FIGW * 0.72, 2.25),
                            gridspec_kw=dict(wspace=0.34))
    ax = axs[0]
    x, y, s = series(e23, 'corr', sd=True)
    ax.errorbar(x, y, yerr=s, fmt='-o', ms=3.8, lw=1.1, capsize=2, color=PAL['mist'],
                mfc=PAL['mist'], mec='white', mew=0.5, label='correlation')
    x, y, s = series(e23, 'notears', sd=True)
    ax.errorbar(x, y, yerr=s, fmt='-s', ms=3.6, lw=1.1, capsize=2, color=PAL['moss'],
                mfc=PAL['moss'], mec='white', mew=0.5, label='NOTEARS')
    x, y, s = series(e29, 'f1', sd=True)
    ax.errorbar(x, y, yerr=s, fmt='-^', ms=4.2, lw=1.2, capsize=2, color=PAL['orchid'],
                mfc=PAL['orchid'], mec='white', mew=0.5, label='RESIT (nonlinear)')
    ax.set_xlabel(r'confounding load $\lambda$')
    ax.set_ylabel(r'$F_1$')
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper right', handlelength=1.6, borderaxespad=0.2)
    letter(ax, 'a')

    ax = axs[1]
    x, y, _ = series(e23, 'auroc')
    ax.plot(x, y, '-o', ms=3.8, lw=1.1, color=PAL['mist'], mfc=PAL['mist'], mec='white',
            mew=0.5, label='correlation')
    x, y, _ = series(e29, 'auroc')
    ax.plot(x, y, '-^', ms=4.2, lw=1.2, color=PAL['orchid'], mfc=PAL['orchid'], mec='white',
            mew=0.5, label='RESIT (nonlinear)')
    ax.set_xlabel(r'confounding load $\lambda$')
    ax.set_ylabel('AUROC')
    ax.set_ylim(0.60, 1.02)
    ax.legend(loc='lower left', handlelength=1.6, borderaxespad=0.2)
    letter(ax, 'b')

    out = os.path.join(FIG, 'fig9_resit.pdf')
    fig.savefig(out, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(out.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', pad_inches=0.02)
    print('wrote', out)


if __name__ == '__main__':
    main()
