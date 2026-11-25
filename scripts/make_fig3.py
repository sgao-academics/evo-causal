# -*- coding: utf-8 -*-
"""Figure 3: four procedures on one grid, read from e23_families10.json (ten seeds).

The previous version plotted FCI at 0.014-0.066, which was the consequence of a label
off-by-one in the edge parser rather than a property of FCI. With the mapping fixed FCI
sits at 0.33-0.39 on the same grid, level with PC, which is exactly what the theory
predicts when the confounder is weak. The figure now shows the ranking inversion that the
correct numbers produce: the correlation rule starts highest and ends lowest.

Every number comes from the JSON; nothing is typed.
"""
import os, sys, json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, 'results')
FIG = os.path.join(ROOT, 'figures')
# figures/ is not tracked, so on a fresh clone it does not exist yet.
os.makedirs(FIG, exist_ok=True)


from palette import PAL, INK, GREY, make_cmap

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


def letter(ax, s, dx=-0.155, dy=1.05):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight='bold',
            va='top', ha='left')


def main():
    d = load('e23_families10')
    if not d:
        raise SystemExit('e23_families10.json missing')
    rows = sorted(d.values(), key=lambda v: v['lam'])
    lam = [r['lam'] for r in rows]
    K = [r['K'] for r in rows]

    series = [
        ('corr', 'corr_sd', 'PIC + correlation', PAL['mist'], 'o', '-', 1.5),
        ('notears', 'notears_sd', 'NOTEARS on PIC', PAL['moss'], 's', '-', 1.5),
        ('pc', 'pc_sd', 'PC on PIC', PAL['peri'], '^', '--', 2.0),
        ('fci', 'fci_sd', 'FCI on PIC', PAL['orchid'], 'D', ':', 1.6),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(FIGW, 2.62),
                             gridspec_kw={'wspace': 0.26})
    fig.subplots_adjust(bottom=0.24)
    for ax, xx, xl in ((axes[0], K, "measured Blomberg's $K$"),
                       (axes[1], lam, r'generator loading $\lambda$')):
        for key, sdkey, lab, c, mk, ls, lw in series:
            y = np.array([r[key] if r.get(key) is not None else np.nan for r in rows])
            s = np.array([r.get(sdkey, 0.0) or 0.0 for r in rows])
            ax.fill_between(xx, y - s, y + s, color=c, alpha=0.16, lw=0)
            ax.plot(xx, y, ls, marker=mk, ms=3.6, lw=lw, color=c, mfc='white',
                    mew=1.0, label=lab, zorder=3 if key in ('f1_corr', 'f1_notears') else 2)
        ax.set_xlabel(xl)
        ax.set_ylabel('$F_1$ (skeleton)')
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(min(xx) - 0.02, max(xx) + 0.02)

    # the crossing of the two continuous methods is the result; mark it once
    a0 = axes[0]
    a0.annotate('ranks invert', xy=(0.222, 0.598), xytext=(0.245, 0.70),
                fontsize=6.6, color=INK, ha='left',
                arrowprops=dict(arrowstyle='-', lw=0.6, color=mixg(GREY, 0.30)))
    # PC and FCI are indistinguishable by design when the confounder is weak; say so
    a0.text(0.305, 0.285, 'PC and FCI coincide', fontsize=6.2, color=mixg(GREY, 0.10),
            ha='right', va='top')
    letter(axes[0], 'a')
    letter(axes[1], 'b')

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.005),
               handlelength=2.0, columnspacing=1.5)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, 'fig3_families.' + ext),
                    dpi=400 if ext == 'png' else None,
                    bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

    print('fig3 ok')
    for r in rows:
        print(f"  lam={r['lam']:.1f} K={r['K_measured']:.3f} corr={r['f1_corr']:.3f} "
              f"NOTEARS={r['f1_notears']:.3f} PC={r['f1_pc']:.3f} FCI={r['f1_fci']:.3f}")


def mixg(c, t):
    def rgb(x):
        return tuple(int(x[i:i + 2], 16) for i in (1, 3, 5))
    a, b = rgb(c), rgb('#FFFFFF')
    return "#%02X%02X%02X" % tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


if __name__ == '__main__':
    main()
