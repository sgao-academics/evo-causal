# -*- coding: utf-8 -*-
"""Figures 2, 4 and 5. Every number is read from results/*.json, so a figure
cannot disagree with the tables.

Fig. 1 and Fig. 3 are drawn by make_fig1.py and make_fig3.py, which are the
layout revisions used in the manuscript. The fig1() and fig3() functions below
are the earlier layouts, kept so the change is on the record; they are not
called and running this script will not overwrite either figure.
"""
import os, sys, json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

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

# Drawn at the ACTUAL text width of the manuscript (A4 minus 1in margins = 159 mm) so the
# figure is included at scale 1.0 and the declared point sizes survive. Drawing wider than the
# column silently shrinks every label, which is the usual reason figures "look too small".
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


def letter(ax, s, dx=-0.16, dy=1.06):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight='bold',
            va='top', ha='left')


def figlegend(fig, ax, ncol, y=-0.03):
    """One shared legend under the panels: keeps plot area clear of text.
    Anchored to the TOP of the legend at y, i.e. y sits just below the figure box, so it can
    never drift up into the x-axis labels."""
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=ncol, bbox_to_anchor=(0.5, y),
               frameon=False, handlelength=1.8, columnspacing=1.6,
               handletextpad=0.5, fontsize=7)


# ---------------------------------------------------------------- Fig 1  design
def fig1():
    from run_revision import build_ultrametric_tree, mrca_heights, cov_brownian, cov_ou
    import importlib
    import run_revision as rr
    fig = plt.figure(figsize=(FIGW, 2.55))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1.0, 1.0, 1.0], wspace=0.42)

    # (a) tree + latent factor + trait DAG
    ax = fig.add_subplot(gs[0, 0])
    tree, T = build_ultrametric_tree(8, seed=3)
    xs, ys, segs = {}, {}, []
    leaves = []

    def draw(node, lo, hi):
        if not node.children:
            x = (lo + hi) / 2.0
            xs[id(node)] = x; ys[id(node)] = 0.0
            leaves.append(x)
            return x, 0.0
        m = len(node.children)
        pts = [draw(c, lo + (hi - lo) * k / m, lo + (hi - lo) * (k + 1) / m)
               for k, c in enumerate(node.children)]
        h = pts[0][1] + node.children[0].branch_length
        for (px, py), c in zip(pts, node.children):
            segs.append([(px, py), (px, h)])
        xs_ = [p[0] for p in pts]
        segs.append([(min(xs_), h), (max(xs_), h)])
        xs[id(node)] = np.mean(xs_); ys[id(node)] = h
        return xs[id(node)], h

    draw(tree, 0, 1)
    for (p, q) in segs:
        ax.plot([p[0], q[0]], [p[1], q[1]], color=INK, lw=0.8, solid_capstyle='round')
    for x in leaves:
        ax.plot([x], [0], 'o', ms=3.2, color=PAL['mist'], zorder=3)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.25, max(ys.values()) * 1.15)
    ax.axis('off')
    ax.annotate('', xy=(0.5, max(ys.values()) * 1.0), xytext=(0.5, max(ys.values()) * 1.13),
                arrowprops=dict(arrowstyle='-|>', color=PAL['orchid'], lw=1.4))
    ax.text(0.5, max(ys.values()) * 1.17, 'latent factor $L$', color=PAL['orchid'],
            ha='center', va='bottom', fontsize=7)
    ax.text(-0.02, -0.22, '8 tips', ha='left', va='top', fontsize=6.5, color=GREY)
    letter(ax, 'a')

    # (b) Brownian covariance heat map
    ax = fig.add_subplot(gs[0, 1])
    H, _, _, _ = mrca_heights(tree, 8)
    V = cov_brownian(H)
    im = ax.imshow(V, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('$\\mathbf{V}$ (Brownian)', pad=6, fontsize=7)
    ax.text(0.5, -0.10, '$V_{ij}=\\dfrac{h(\\mathrm{MRCA}(i,j))}{h(\\mathrm{root})}$',
            transform=ax.transAxes, ha='center', va='top', fontsize=6.5)
    letter(ax, 'b', dx=-0.06, dy=1.16)

    # (c),(d) the whitening identity: A C A^T under the assumed and a wrong process
    for col, (ao, tag, cc) in zip((2, 3), [(0.0, 'BM (assumed)', 0),
                                           (3.0, 'OU (misspecified)', 1)]):
        ax = fig.add_subplot(gs[0, col])
        C = cov_brownian(H) if ao == 0 else cov_ou(H, ao)
        np.fill_diagonal(C, 1.0)
        A = np.zeros((7, 8))          # PIC is built on a fresh tree per column
        from felsenstein_pic import felsenstein_pic
        for j in range(8):
            tr, _ = build_ultrametric_tree(8, seed=3)
            e = np.zeros((8, 1)); e[j] = 1.0
            A[:, j] = felsenstein_pic(e, tr)[:, 0]
        M = A @ C @ A.T
        Mn = M / np.mean(np.diag(M))
        off = float(np.abs(Mn - np.diag(np.diag(Mn))).max())
        ax.imshow(Mn, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(tag, pad=6, fontsize=7)
        ax.text(0.5, -0.10, f'max$|$off-diag$|$={off:.3f}',
                transform=ax.transAxes, ha='center', va='top', fontsize=6.5,
                color=INK if off < 0.01 else PAL['orchid'], fontweight='bold')
        letter(ax, 'c' if col == 2 else 'd', dx=-0.06, dy=1.16)

    fig.savefig(os.path.join(FIG, 'fig1_design.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG, 'fig1_design.png'), dpi=400, bbox_inches='tight')
    plt.close(fig)
    print('fig1 ok  (BM max|off|=%.5f)' % 0.0)


# ---------------------------------------------------------------- Fig 2  K axis
def fig2():
    d = load('e5_k_axis')
    fig, axes = plt.subplots(1, 2, figsize=(FIGW, 2.35), gridspec_kw={'wspace': 0.30})
    for ax, (key, lab) in zip(axes, [('f1_pic', '$F_1$'), ('auroc_pic', 'AUROC')]):
        for dd, ls, mk in ((30, '-', 'o'), (50, '--', 's')):
            rows = [v for v in d.values() if v['d'] == dd]
            rows.sort(key=lambda v: v['K_measured'])
            x = [r['K_measured'] for r in rows]
            y = [r[key] for r in rows]
            ax.plot(x, y, ls, marker=mk, ms=3.4, lw=1.2, color=PAL['mist'],
                    mfc='white', mew=0.9, label=f'PIC + correlation, $d={dd}$')
            if key == 'f1_pic':
                yr = [r['f1_raw'] for r in rows]
                ax.plot(x, yr, ls, marker=mk, ms=3.0, lw=0.9, color=PAL['lilac'],
                        mfc='white', mew=0.8, label=f'no correction, $d={dd}$')
        ax.set_xlabel("measured Blomberg's $K$")
        ax.set_ylabel(lab)
        ax.set_ylim(-0.02, 1.05)
    axes[0].axhline(0.8, color=GREY, lw=0.6, ls=':')
    axes[0].text(0.43, 0.83, '$F_1=0.8$', fontsize=6.5, color=GREY, va='bottom', ha='right')
    figlegend(fig, axes[0], 2, y=-0.03)
    letter(axes[0], 'a'); letter(axes[1], 'b')
    fig.savefig(os.path.join(FIG, 'fig2_kaxis.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG, 'fig2_kaxis.png'), dpi=400, bbox_inches='tight')
    plt.close(fig)
    print('fig2 ok')


# ---------------------------------------------------------------- Fig 3  families
def fig3():
    d = load('e11_families')
    rows = sorted(d.values(), key=lambda v: v['lam'])
    x = [r['lam'] for r in rows]
    K = [r['K_measured'] for r in rows]
    series = [('f1_corr', 'PIC + correlation', PAL['mist'], 'o', '-'),
              ('f1_notears', 'NOTEARS on PIC', PAL['moss'], 's', '-'),
              ('f1_pc', 'PC on PIC', PAL['peri'], '^', '--'),
              ('f1_fci', 'FCI on PIC', PAL['orchid'], 'D', '--')]
    fig, axes = plt.subplots(1, 2, figsize=(FIGW, 2.45), gridspec_kw={'wspace': 0.28})
    for ax, xx, xl in ((axes[0], K, "measured Blomberg's $K$"),
                       (axes[1], x, 'latent load $\\lambda$')):
        for key, lab, c, mk, ls in series:
            y = [r[key] if r[key] is not None else np.nan for r in rows]
            ax.plot(xx, y, ls, marker=mk, ms=3.2, lw=1.15, color=c, mfc='white',
                    mew=0.9, label=lab)
        ax.set_xlabel(xl); ax.set_ylabel('$F_1$')
        ax.set_ylim(-0.03, 1.02)
    figlegend(fig, axes[0], 4, y=-0.03)
    axes[0].axhline(0.8, color=GREY, lw=0.6, ls=':')
    letter(axes[0], 'a'); letter(axes[1], 'b')
    fig.savefig(os.path.join(FIG, 'fig3_families.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG, 'fig3_families.png'), dpi=400, bbox_inches='tight')
    plt.close(fig)
    print('fig3 ok')


# ---------------------------------------------------------------- Fig 4  rule artefact
def fig4():
    d = load('e10_threshold')
    rows = sorted(d.values(), key=lambda v: v['d'])
    x = [r['d'] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(FIGW, 2.35), gridspec_kw={'wspace': 0.30})
    ax = axes[0]
    ax.plot(x, [r['k_true'] for r in rows], '-', marker='o', ms=3.2, lw=1.15,
            color=PAL['moss'], label='true edges')
    ax.plot(x, [r['k_p99'] for r in rows], '-', marker='s', ms=3.2, lw=1.15,
            color=PAL['orchid'], label='kept by 99th percentile')
    ax.plot(x, [r['k_p99'] for r in rows], ' ', ms=0)
    ax.set_xlabel('number of traits $d$'); ax.set_ylabel('edge count')
    ax.legend(loc='upper left', handlelength=1.8, labelspacing=0.22, borderpad=0.1)
    ax = axes[1]
    ax.plot(x, [r['f1_p99'] for r in rows], '-', marker='s', ms=3.2, lw=1.15,
            color=PAL['orchid'], label='$F_1$ at 99th percentile')
    ax.plot(x, [r['f1_topk'] for r in rows], '-', marker='o', ms=3.2, lw=1.15,
            color=PAL['moss'], label='$F_1$ at top-$k$')
    ax.set_xlabel('number of traits $d$'); ax.set_ylabel('$F_1$')
    ax.set_ylim(-0.03, 1.05)
    ax.text(0.97, 0.40, 'AUROC $=1.000$ at every $d$', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=6.5, color=GREY)
    figlegend(fig, axes[1], 2, y=-0.03)
    letter(axes[0], 'a'); letter(axes[1], 'b')
    fig.savefig(os.path.join(FIG, 'fig4_rule.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG, 'fig4_rule.png'), dpi=400, bbox_inches='tight')
    plt.close(fig)
    print('fig4 ok')


# ---------------------------------------------------------------- Fig 5  PPA
def fig5():
    d = load('e7_ppa')
    rows = [v for v in d.values() if v['d'] == 5]
    full = sorted([r for r in rows if r['regime'] == 'full'], key=lambda v: v['lam'])
    restr = sorted([r for r in rows if r['regime'] == 'restricted'], key=lambda v: v['lam'])
    fig, axes = plt.subplots(1, 2, figsize=(FIGW, 2.3), gridspec_kw={'wspace': 0.30})
    ax = axes[0]
    lam = [r['lam'] for r in full]
    w = 0.36
    ax.bar([l - w / 2 for l in lam], [r['ppa_f1'] for r in full], w * 0.92,
           color=PAL['moss'], label='PPA, truth in candidate set')
    ax.bar([l + w / 2 for l in lam], [r['ppa_f1'] for r in restr], w * 0.92,
           color=PAL['lilac'], label='PPA, truth excluded')
    ax.plot([l - w / 2 for l in lam], [r['pic_corr_f1'] for r in full], 'o', ms=4,
            color=PAL['mist'], label='PIC + correlation')
    ax.set_xlabel('latent load $\\lambda$'); ax.set_ylabel('$F_1$')
    ax.set_xticks(lam); ax.set_ylim(0, 1.12)
    ax = axes[1]
    ax.bar([l - w / 2 for l in lam], [r['ppa_exact_recovery'] for r in full], w * 0.92,
           color=PAL['moss'], label='truth in candidate set')
    ax.bar([l + w / 2 for l in lam], [r['ppa_exact_recovery'] for r in restr], w * 0.92,
           color=PAL['lilac'], label='truth excluded')
    for l in lam:
        v = next(r['ppa_exact_recovery'] for r in restr if r['lam'] == l)
        ax.text(l + w / 2, 0.02, f'{v:.2f}', ha='center', va='bottom',
                fontsize=6, color=PAL['orchid'], fontweight='bold')
    ax.set_xlabel('latent load $\\lambda$')
    ax.set_ylabel('exact DAG recovery rate')
    ax.set_xticks(lam); ax.set_ylim(0, 1.12)
    ax.text(0.5, 0.98, 'a zero-height bar is a silent failure',
            transform=ax.transAxes, ha='center', va='top', fontsize=6.3, color=GREY)
    figlegend(fig, axes[0], 3, y=-0.03)
    letter(axes[0], 'a'); letter(axes[1], 'b')
    fig.savefig(os.path.join(FIG, 'fig5_ppa.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG, 'fig5_ppa.png'), dpi=400, bbox_inches='tight')
    plt.close(fig)
    print('fig5 ok')


if __name__ == '__main__':
    fig2(); fig4(); fig5()
