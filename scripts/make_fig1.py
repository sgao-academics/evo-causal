# -*- coding: utf-8 -*-
"""Figure 1 rebuilt in the conference-figure idiom, with numbers taken from the run.

What was wrong with the previous version, and what changed
---------------------------------------------------------
1.  The panel letter 'd' collided with the title "OU (misspecified)" and rendered as "dOU".
    Panel letters now sit inside the panel frame, on a different row from any title, so
    they cannot overlap.
2.  The tree was drawn for 8 tips while the text quoted the 40-tip numbers, so the figure
    said 0.044 and the text said 0.072. Both now use the 12-tip tree in
    results/e1_whitening_n12.json, and the number printed on each panel is read
    from that file rather than typed.
3.  Four free-floating panels gave no reading order. Three rounded zones with right-angle
    connectors make the order process -> covariance -> after contrasts explicit.

Drawing rules follow the house figure standard: rounded zone containers with a tinted
fill and same-hue edge, white nodes with a hairline edge, right-angle connectors in neutral
grey with a solid triangular head, no axes, no grid, no outer frame. Colours from
palette.py only.
"""
import os, sys, json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, 'figures')
RES = os.path.join(ROOT, 'results')
os.makedirs(FIG, exist_ok=True)




from palette import PAL, INK, GREY, FRAME, make_cmap
from run_revision import build_ultrametric_tree

WHITE = "#FFFFFF"


# ---------------------------------------------------------------- helpers
def _rgb(c):
    return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))


def mix(c1, c2, t):
    a, b = _rgb(c1), _rgb(c2)
    return "#%02X%02X%02X" % tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def tint(c, t):
    return mix(c, WHITE, t)


def shade(c, t):
    return mix(c, INK, t)


MUTED = mix(INK, WHITE, 0.42)


def rbox(ax, x, y, w, h, fc=WHITE, ec=FRAME, lw=1.0, r=0.9, ls='-', z=2):
    ax.add_patch(FancyBboxPatch((x + r, y + r), max(w - 2 * r, 1e-6), max(h - 2 * r, 1e-6),
                                boxstyle="round,pad=%.3f,rounding_size=%.3f" % (r, r),
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def zone(ax, x, y, w, h, label, color, sub=None):
    rbox(ax, x, y, w, h, fc=tint(color, 0.91), ec=mix(color, WHITE, 0.48), lw=1.1,
         r=1.7, z=1)
    ax.text(x + 1.7, y + h - 1.6, label, fontsize=8.2, fontweight='bold',
            color=shade(color, 0.28), va='top', ha='left', zorder=6)
    if sub:
        ax.text(x + 1.7, y + h - 3.9, sub, fontsize=6.4, color=MUTED,
                va='top', ha='left', zorder=6)


def arrow(ax, pts, lw=1.3, ms=9):
    color = mix(GREY, WHITE, 0.15)
    pts = [tuple(p) for p in pts]
    if len(pts) > 2:
        ax.plot([p[0] for p in pts[:-1]], [p[1] for p in pts[:-1]], color=color, lw=lw,
                solid_capstyle='round', zorder=3)
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle='-|>', mutation_scale=ms,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=3))


def tree_geom(tree):
    """Segments and tip positions in unit coordinates, height normalised to 0..1."""
    segs, tips = [], []

    def rec(node, lo, hi):
        if not node.children:
            x = (lo + hi) / 2.0
            tips.append((x, 0.0))
            return x, 0.0
        n = len(node.children)
        out = [rec(c, lo + (hi - lo) * k / n, lo + (hi - lo) * (k + 1) / n)
               for k, c in enumerate(node.children)]
        h = out[0][1] + node.children[0].branch_length
        for (cx, cy) in out:
            segs.append(((cx, cy), (cx, h)))
        xs = [o[0] for o in out]
        segs.append(((min(xs), h), (max(xs), h)))
        return float(np.mean(xs)), h

    rec(tree, 0.0, 1.0)
    hmax = max(max(s[0][1], s[1][1]) for s in segs) or 1.0
    segs = [((a[0], a[1] / hmax), (b[0], b[1] / hmax)) for a, b in segs]
    tips = [(x, 0.0) for x, _ in tips]
    return segs, tips


# ---------------------------------------------------------------- figure
def main():
    e1 = json.load(open(os.path.join(RES, 'e1_whitening_n12.json'), encoding='utf-8'))
    N = e1['n']
    off_bm = e1['BM']['max_offdiag']
    off_ou = e1['OU_alpha3']['max_offdiag']

    arr = np.load(os.path.join(HERE, '_fig1_arrays.npz'))
    V, M_BM, M_OU = arr['V'], arr['M_BM'], arr['M_OU']
    cmap = make_cmap()

    FIGW, FIGH = 159 / 25.4, 2.12
    XMAX, YMAX = 100.0, 32.4
    fig = plt.figure(figsize=(FIGW, FIGH))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, XMAX)
    ax.set_ylim(0, YMAX)
    ax.set_aspect('equal')
    ax.axis('off')

    y0, hz = 1.0, 30.4

    # ============================================================ zone 1
    zA = (0.6, 30.4)
    zone(ax, zA[0], y0, zA[1], hz, r'$\bf{1}$  The process', PAL['lilac'])

    tree, _ = build_ultrametric_tree(N, seed=e1['tree_seed'])
    segs, tips = tree_geom(tree)
    tx0, tx1 = zA[0] + 3.6, zA[0] + zA[1] - 3.6
    ty0, ty1 = y0 + 6.6, y0 + hz - 7.0
    for (a, b) in segs:
        ax.plot([tx0 + a[0] * (tx1 - tx0), tx0 + b[0] * (tx1 - tx0)],
                [ty0 + a[1] * (ty1 - ty0), ty0 + b[1] * (ty1 - ty0)],
                color=INK, lw=0.85, solid_capstyle='round', zorder=4)
    for (x, _) in tips:
        ax.plot([tx0 + x * (tx1 - tx0)], [ty0], 'o', ms=2.5, color=PAL['mist'],
                mec=WHITE, mew=0.45, zorder=5)

    cx = (tx0 + tx1) / 2
    ax.annotate('', xy=(cx, ty1), xytext=(cx, ty1 + 1.6),
                arrowprops=dict(arrowstyle='-|>', color=PAL['orchid'], lw=1.1,
                                mutation_scale=8))
    ax.text(cx, ty1 + 1.9, 'latent factor $L$', fontsize=6.6,
            color=shade(PAL['orchid'], 0.28), ha='center', va='bottom', zorder=6)

    ax.text(tx0, ty0 - 1.4, '%d tips' % N, fontsize=6.2, color=MUTED,
            ha='left', va='top', zorder=6)
    ax.text(tx1, ty0 - 1.4, r'$L_i \sim \mathrm{BM}(\mathbf{V})$', fontsize=6.2,
            color=MUTED, ha='right', va='top', zorder=6)
    ax.text(zA[0] + 1.5, y0 + 1.2, 'a', fontsize=8.6, fontweight='bold', color=INK,
            ha='left', va='bottom', zorder=7)

    # ============================================================ zone 2
    zB = (34.0, 21.4)
    zone(ax, zB[0], y0, zB[1], hz, r'$\bf{2}$  Covariance', PAL['peri'],
         'shared path to the MRCA')

    side = 13.2
    bx = zB[0] + (zB[1] - side) / 2
    by = y0 + 6.6
    ax.imshow(V, cmap=cmap, vmin=0, vmax=1, extent=(bx, bx + side, by, by + side),
              origin='upper', interpolation='nearest', zorder=3)
    rbox(ax, bx, by, side, side, fc='none', ec=mix(PAL['peri'], WHITE, 0.30), lw=0.8,
         r=0.4, z=4)
    ax.text(bx + side / 2, by - 1.3,
            r'$V_{ij}=h(\mathrm{MRCA}(i,j))\,/\,h(\mathrm{root})$',
            fontsize=6.4, color=INK, ha='center', va='top', zorder=5)
    ax.text(zB[0] + 1.5, y0 + 1.2, 'b', fontsize=8.6, fontweight='bold', color=INK,
            ha='left', va='bottom', zorder=7)

    # ============================================================ zone 3
    zC = (59.0, 40.4)
    zone(ax, zC[0], y0, zC[1], hz, r'$\bf{3}$  After contrasts', PAL['mist'],
         r'$\mathbf{A}\,\mathbf{C}\,\mathbf{A}^{\!\top}$, normalised; number = max$|$off-diag$|$')

    side2, gap = 12.0, 5.2
    tot = 2 * side2 + gap
    cx0 = zC[0] + (zC[1] - tot) / 2
    cy0 = y0 + 8.0
    panels = [(M_BM, r'Brownian', off_bm, 'c', PAL['mist']),
              (M_OU, r'OU, $\alpha=3$', off_ou, 'd', PAL['orchid'])]
    for k, (M, tag, off, letter, col) in enumerate(panels):
        x = cx0 + k * (side2 + gap)
        ax.imshow(M, cmap=cmap, vmin=0, vmax=1, extent=(x, x + side2, cy0, cy0 + side2),
                  origin='upper', interpolation='nearest', zorder=3)
        rbox(ax, x, cy0, side2, side2, fc='none', ec=mix(col, WHITE, 0.30), lw=0.8,
             r=0.4, z=4)
        # letter inside the frame, top-left: cannot collide with anything
        ax.text(x - 0.2, cy0 + side2 - 0.2, letter, fontsize=8.0, fontweight='bold',
                color=WHITE, ha='left', va='top', zorder=7,
                bbox=dict(boxstyle='circle,pad=0.16', fc=shade(col, 0.10), ec='none'))
        ax.text(x + side2 / 2, cy0 + side2 + 0.7, tag, fontsize=6.8, color=INK,
                ha='center', va='bottom', zorder=5)
        ax.text(x + side2 / 2, cy0 - 1.0, ('%.3f' % off), fontsize=9.0,
                fontweight='bold', color=shade(col, 0.22), ha='center', va='top', zorder=5)

    ax.text(zC[0] + 1.5, y0 + 1.2, '', fontsize=8.6, color=INK, ha='left', va='bottom')

    # ============================================================ connectors
    ym = y0 + hz / 2
    arrow(ax, [(zA[0] + zA[1] + 0.5, ym), (zB[0] - 0.5, ym)])
    arrow(ax, [(zB[0] + zB[1] + 0.5, ym), (zC[0] - 0.5, ym)])

    # No in-figure headline: the journal caption carries the title, and a headline here
    # would read as a second one directly above "Figure 1:".

    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, 'fig1_design.' + ext),
                    dpi=400 if ext == 'png' else None, facecolor=WHITE,
                    bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f'fig1 ok  n={N}  BM={off_bm:.5f}  OU3={off_ou:.5f}')


if __name__ == '__main__':
    main()
