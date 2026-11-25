# -*- coding: utf-8 -*-
"""Figures 6 and 7 of the revision.

Fig. 6  the dissociation: performance against measured K for three data-generating arms that
        reach the same K by different routes.
Fig. 7  the remedy: the rank-one screen and the deflation it motivates.

Both read results/*.json only, so a figure cannot disagree with the tables.
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
    from palette import PAL, INK, GREY, GRID
except Exception:
    PAL = {'mist': '#4D6EAF', 'peri': '#7A9AC3', 'lilac': '#B0B9CB',
           'orchid': '#D0A0B6', 'moss': '#8FB578'}
    INK, GREY, GRID = '#222222', '#8A8A8A', '#E6E6E6'

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

C_SIG = PAL.get('moss', '#8FB578')      # signal without a confounder
C_TREE = PAL.get('mist', '#4D6EAF')     # confounder that evolves on the tree
C_IID = PAL.get('orchid', '#D0A0B6')    # confounder with no phylogenetic structure
LILAC = PAL.get('lilac', '#B0B9CB')     # baseline / reference colour


def load(n):
    p = os.path.join(RES, n + '.json')
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else {}


def letter(ax, s, dx=-0.16, dy=1.07):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight='bold',
            va='top', ha='left')


def figlegend(fig, ax, ncol, y=-0.04):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=ncol, bbox_to_anchor=(0.5, y),
               frameon=False, handlelength=1.8, columnspacing=1.6,
               handletextpad=0.5, fontsize=7)


def series(res, arm, d, field):
    xs, ys = [], []
    for k, v in res.items():
        if v.get('arm') == arm and v.get('d') == d:
            xs.append(v['K']); ys.append(v[field])
    o = np.argsort(xs)
    return np.array(xs)[o], np.array(ys)[o]


# ---------------------------------------------------------------- Fig 6
def fig6():
    res = load('e14_dissociation')
    if not res:
        print('e14_dissociation.json missing'); return
    fig, axes = plt.subplots(1, 2, figsize=(FIGW, 2.32))
    for ax, field, lab in zip(axes, ('f1_pic', 'auroc'),
                              ('$F_1$ under top-$k$', 'AUROC')):
        for d, ls in ((30, '-'), (50, '--')):
            for arm, col, mk, name in (('signal', C_SIG, 'o', 'signal, no confounder'),
                                       ('tree', C_TREE, 's', 'confounder on the tree'),
                                       ('iid', C_IID, '^', 'confounder, no tree')):
                x, y = series(res, arm, d, field)
                if len(x) == 0:
                    continue
                ax.plot(x, y, ls=ls, marker=mk, ms=3.4, lw=1.0, color=col,
                        mfc='white' if ls == '--' else col, mew=0.7,
                        label=('%s, $d=%d$' % (name, d)) if field == 'f1_pic' else None)
        ax.set_xlabel("Blomberg's $K$ (measured)")
        ax.set_ylabel(lab)
        ax.set_xlim(0, 0.85)
    axes[0].axhline(0.8, ls=':', lw=0.8, color=GREY)
    axes[0].set_ylim(0, 1.0)
    axes[1].set_ylim(0.45, 1.02)
    letter(axes[0], 'a'); letter(axes[1], 'b')
    figlegend(fig, axes[0], 2, y=-0.02)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.92, bottom=0.27, wspace=0.30)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG, 'fig6_dissociation.' + ext), dpi=300,
                    bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print('fig6_dissociation written')


# ---------------------------------------------------------------- Fig 7
def fig7():
    sc = load('e15_screen')
    de = load('e16_deflate')
    if not sc or not de:
        print('missing e15/e16'); return
    fig, axes = plt.subplots(1, 2, figsize=(FIGW, 2.32))

    # (a) screen: r by cell. Filled = d=30, hollow = d=50, one colour per arm.
    scb = load('e15b_screen_signal')
    ax = axes[0]
    for d in (30, 50):
        null = sc.get('null_d%d' % d, {})
        if null:
            ax.axhspan(null['q05'], null['q95'], color=LILAC, alpha=0.40, lw=0)
    lab_done = set()
    for d in (30, 50):
        for lat, mk, col, name in (('tree', 's', C_TREE, 'confounder on the tree'),
                                   ('iid', '^', C_IID, 'confounder, no tree'),
                                   ('signal', 'o', C_SIG, 'signal only')):
            src = scb if lat == 'signal' else sc
            lv, rr = [], []
            for k, v in src.items():
                if isinstance(v, dict) and v.get('d') == d:
                    if lat != 'signal' and v.get('latent') != lat:
                        continue
                    lv.append(v['level']); rr.append(v['r'])
            if not lv:
                continue
            o = np.argsort(lv)
            ax.plot(np.array(lv)[o], np.array(rr)[o], marker=mk, ms=3.2, lw=0.9,
                    color=col, mfc=(col if d == 30 else 'white'), mew=0.8,
                    label=name if d == 30 else None)
    ax.set_xlabel('confounding load $\\lambda$')
    ax.set_ylabel('leading / rest eigenvalue')
    ax.set_yscale('log')
    ax.set_ylim(2.2, 200)
    ax.set_xticks([0, 1, 2, 3, 4, 6])
    ax.legend(loc='upper left', fontsize=6.2, handletextpad=0.4, labelspacing=0.25)
    letter(ax, 'a')

    # (b) deflation
    ax = axes[1]
    for d, col in ((30, C_TREE), (50, C_SIG)):
        lam, pl, dl = [], [], []
        for k, v in de.items():
            if isinstance(v, dict) and v.get('d') == d and 'lam' in v:
                lam.append(v['lam']); pl.append(v['plain']); dl.append(v['deflate'])
        if not lam:
            continue
        o = np.argsort(lam)
        lam = np.array(lam)[o]
        ax.plot(lam, np.array(pl)[o], marker='o', ms=3.2, lw=1.0, color=col, mfc='white',
                mew=0.8, ls='-', label='$d=%d$, as is' % d)
        ax.plot(lam, np.array(dl)[o], marker='o', ms=3.2, lw=1.0, color=col, ls='-',
                label='$d=%d$, leading component removed' % d)
    ax.set_xlabel('confounding load $\\lambda$')
    ax.set_ylabel('$F_1$ under top-$k$')
    ax.set_ylim(0, 1.0)
    ax.axhline(0.8, ls=':', lw=0.8, color=GREY)
    letter(ax, 'b')
    figlegend(fig, axes[1], 2, y=-0.02)
    fig.subplots_adjust(left=0.095, right=0.985, top=0.92, bottom=0.30, wspace=0.34)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG, 'fig7_remedy.' + ext), dpi=300,
                    bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print('fig7_remedy written')


if __name__ == '__main__':
    fig6(); fig7()
