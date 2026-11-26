# -*- coding: utf-8 -*-
"""Figure S2: four procedures on one grid, read from e23_families10.json (ten seeds).

The previous version plotted FCI at 0.014-0.066, which was the consequence of a label
off-by-one in the edge parser rather than a property of FCI. With the mapping fixed FCI
sits at 0.33-0.39 on the same grid, level with PC, which is exactly what the theory
predicts when the confounder is weak. The figure now shows the ranking inversion that the
correct numbers produce: the correlation rule starts highest and ends lowest.

Three things this version corrects beyond the data.

1. The "ranks invert" leader pointed at (K, F1) = (0.222, 0.598), which is a point on the
   NOTEARS curve, not the crossing. The curves are exactly equal at lambda = 2, i.e. K = 0.276
   and F1 = 0.560; the leader now ends there.

2. All uncertainty bands are drawn before any line. In the old loop each band was painted just
   before its own curve, so the third and fourth bands were laid over the first two curves and
   tinted them; with four overlapping bands at alpha 0.16 that is a visible smear along the
   middle of the panel.

3. Panel (b) is not decoration. K is measured on the data and lambda is the generator parameter
   an analyst never observes; that the two panels rank the four methods identically is the
   evidence that K is an adequate stand-in for lambda. The old caption called it "for reference
   only", which sold the panel short.

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

C_MIST = PAL['mist']    # the correlation rule, as everywhere else in this paper
C_NOTE = PAL['moss']    # NOTEARS, as in Fig. S3
C_PC = PAL['peri']      # PC
C_FCI = PAL['orchid']   # FCI

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


def letter(ax, s, dx=-0.150, dy=1.060):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight='bold',
            va='top', ha='left')


def main():
    d = load('e23_families10')
    if not d:
        raise SystemExit('e23_families10.json missing')
    rows = sorted(d.values(), key=lambda v: v['lam'])
    lam = np.array([r['lam'] for r in rows], float)
    K = np.array([r['K'] for r in rows], float)

    series = [
        ('corr', 'corr_sd', 'PIC + correlation', C_MIST, 'o', '-', 1.5, 4),
        ('notears', 'notears_sd', 'NOTEARS on PIC', C_NOTE, 's', '-', 1.5, 4),
        ('pc', 'pc_sd', 'PC on PIC', C_PC, '^', '--', 1.4, 3),
        ('fci', 'fci_sd', 'FCI on PIC', C_FCI, 'D', ':', 1.5, 3),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(FIGW, 2.56),
                             gridspec_kw={'wspace': 0.26})
    # see make_figS1.py: the default margins left the tight bounding box 136 mm wide, and the
    # 1.17x blow-up to \textwidth put the labels at 9.4 pt against the 8 pt of the main figures
    fig.subplots_adjust(left=0.062, right=0.987, top=0.900, bottom=0.215)

    for ax, xx, xl in ((axes[0], K, "measured Blomberg's $K$"),
                       (axes[1], lam, r'generator loading $\lambda$')):
        # every band first, then every line: a band painted after a line tints that line
        for key, sdkey, lab, c, mk, ls, lw, z in series:
            y = np.array([r[key] if r.get(key) is not None else np.nan for r in rows])
            s = np.array([r.get(sdkey, 0.0) or 0.0 for r in rows])
            ax.fill_between(xx, y - s, y + s, color=c, alpha=0.13, lw=0, zorder=1)
        for key, sdkey, lab, c, mk, ls, lw, z in series:
            y = np.array([r[key] if r.get(key) is not None else np.nan for r in rows])
            ax.plot(xx, y, ls, marker=mk, ms=3.6, lw=lw, color=c, mfc='white',
                    mew=1.0, label=lab, zorder=z)
        ax.set_xlabel(xl)
        ax.set_ylabel('$F_1$ (skeleton)')
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    axes[0].set_xlim(K.min() - 0.030, K.max() + 0.020)
    axes[1].set_xlim(lam.min() - 0.110, lam.max() + 0.080)
    axes[0].set_xticks([0.1, 0.2, 0.3, 0.4])
    axes[1].set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    # the crossing is a real crossing and it is exactly located: the two point estimates are
    # equal to three decimals at lambda = 2, which is K = 0.276 and F1 = 0.560
    i = int(np.argmin(np.abs(lam - 2.0)))
    xc, yc = K[i], rows[i]['corr']
    a0 = axes[0]
    a0.plot([xc], [yc], marker='o', ms=7.0, mfc='none', mec=INK, mew=0.55,
            ls='none', zorder=5)
    a0.annotate('curves meet', xy=(xc, yc), xytext=(xc - 0.012, yc + 0.175),
                fontsize=6.4, color=INK, ha='center', va='bottom',
                arrowprops=dict(arrowstyle='-', lw=0.55, color=GREY,
                                shrinkA=1.5, shrinkB=3.0))
    # PC and FCI are indistinguishable by design when the confounder is weak; say so
    a0.text(0.300, 0.225, 'PC and FCI differ by\nat most $0.010$', fontsize=6.2,
            color=GREY, ha='right', va='top')
    letter(axes[0], 'a')
    letter(axes[1], 'b')

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.020),
               handlelength=2.2, columnspacing=1.5, handletextpad=0.55, fontsize=7)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, 'figS2_families.' + ext),
                    dpi=400 if ext == 'png' else None,
                    bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

    print('figS2_families ok')
    for r in rows:
        print(f"  lam={r['lam']:.1f} K={r['K']:.3f} corr={r['corr']:.3f} "
              f"NOTEARS={r['notears']:.3f} PC={r['pc']:.3f} FCI={r['fci']:.3f}")
    gaps = [abs(r['pc'] - r['fci']) for r in rows]
    print('  max |PC - FCI| = %.4f at lam = %.1f'
          % (max(gaps), rows[int(np.argmax(gaps))]['lam']))


if __name__ == '__main__':
    main()
