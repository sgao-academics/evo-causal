"""
gen_fig3_method_landscape.py
Figure 3: Method Landscape — 2x2 panel: (A) head2head wins, (B) wall penetration,
(C) nonlinear robustness, (D) best-F1 summary.
Input: master_all.json + nonlinear_wall.json. Output: figures/fig3_method_landscape.pdf
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.transforms as mtransforms
matplotlib.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8, 'axes.titlesize': 9.5, 'axes.labelsize': 7.5,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 7,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'axes.spines.top': False, 'axes.spines.right': False,
})

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')  # portable
CKPT = os.path.join(ROOT, 'results', 'master_all.json')
CKPT_NL = os.path.join(ROOT, 'results', 'nonlinear_wall.json')
OUTDIR = os.path.join(ROOT, 'figures')
os.makedirs(OUTDIR, exist_ok=True)

with open(CKPT, 'r', encoding='utf-8') as f:
    ckpt = json.load(f)
with open(CKPT_NL, 'r', encoding='utf-8') as f:
    nl = json.load(f)

METHOD_COLORS = {
    'PIC+corr': '#2166ac', 'CT': '#b2182b', 'NOTEARS': '#4daf4a',
    'DAGMA': '#ff7f00', 'GOLEM': '#984ea3',
}

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.subplots_adjust(hspace=0.45, wspace=0.35, left=0.08, right=0.93, top=0.92, bottom=0.08)

# ============ Panel A: Head-to-head wins ============
ax = axes[0, 0]
h2h = ckpt['p5']
wins = {}
for k, v in h2h.items():
    wins[v.get('best', 'unknown')] = wins.get(v.get('best', 'unknown'), 0) + 1

methods_order = ['PIC+corr', 'NOTEARS', 'GOLEM', 'CT', 'DAGMA']
win_counts = [wins.get(m, 0) for m in methods_order]
bar_colors = [METHOD_COLORS[m] for m in methods_order]
bars = ax.bar(methods_order, win_counts, color=bar_colors,
              edgecolor='#333333', linewidth=0.5, width=0.55)

for bar, count in zip(bars, win_counts):
    if count > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{count}/15', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    else:
        ax.text(bar.get_x() + bar.get_width()/2, 0.2, '0', ha='center', va='bottom',
                fontsize=7.5, color='#999999')
ax.set_ylabel('Wins (out of 15)', fontsize=7.5)
ax.axhline(y=3, color='gray', linestyle='--', linewidth=0.8, alpha=0.35)
ax.text(0.01, 3.3, 'chance', fontsize=6, color='gray')
bars[0].set_edgecolor('#a50f15')
bars[0].set_linewidth(2)
ax.text(-0.12, 1.04, 'A', transform=ax.transAxes, fontsize=12, fontweight='bold')
ax.set_title('Head-to-head: PIC+corr wins 14/15', fontsize=9, fontweight='bold', pad=4)

# ============ Panel B: Wall penetration ============
ax = axes[0, 1]
b_data = nl['exp_b_penetration']
n_vals = [100, 200, 500, 1000, 2000, 5000]
a_vals = [0.01, 0.005, 0.001, 0.0001, 0.0]
pen_mat = np.zeros((len(n_vals), len(a_vals)))
for ni, nv in enumerate(n_vals):
    for ai, av in enumerate(a_vals):
        key = f'n={nv}_a={av}'
        pen_mat[ni, ai] = b_data.get(key, {}).get('pic_mean', 0)

p_cmap = LinearSegmentedColormap.from_list('pen',
    ['#f7f7f7', '#d1e5f0', '#92c5de', '#4393c3', '#2166ac', '#053061'], N=256)

im = ax.imshow(pen_mat, aspect='auto', cmap=p_cmap, vmin=0, vmax=0.25, origin='upper')
for ni in range(len(n_vals)):
    for ai in range(len(a_vals)):
        v = pen_mat[ni, ai]
        if v > 0.001:
            c = 'white' if v > 0.17 else '#333333'
            ax.text(ai, ni, f'{v:.3f}', ha='center', va='center', fontsize=6, fontweight='bold', color=c)

ax.set_xticks(range(len(a_vals)))
ax.set_xticklabels([r'$10^{-2}$', r'$5{\times}10^{-3}$', r'$10^{-3}$', r'$10^{-4}$', '0'],
                   rotation=45, ha='right', fontsize=5.5)
ax.set_yticks(range(len(n_vals)))
ax.set_yticklabels([str(n) for n in n_vals], fontsize=6)
ax.set_xlabel(r'Phylogenetic signal $\alpha$', fontsize=7.5)
ax.set_ylabel(r'Sample size $n$', fontsize=7.5)
ax.axhline(y=0.5, color='#a50f15', linestyle='--', linewidth=1.3, alpha=0.7)
ax.text(4.2, 0.3, 'PENETRATED', color='#a50f15', fontsize=6.5, fontweight='bold', ha='right')
ax.text(4.2, 1.3, 'WALL', color='#888888', fontsize=6.5, fontweight='bold', ha='right')
cbar_ax = fig.add_axes([0.94, 0.54, 0.012, 0.36])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label(r'F$_1$', fontsize=7.5, labelpad=3)
cbar.ax.tick_params(labelsize=6)
cbar.outline.set_linewidth(0.5)
ax.text(-0.12, 1.04, 'B', transform=ax.transAxes, fontsize=12, fontweight='bold')
ax.set_title('Wall penetration ($d=50$)', fontsize=9, fontweight='bold', pad=4)

# ============ Panel C: Nonlinear robustness ============
ax = axes[1, 0]
a_nl = nl['exp_a_nonlinear']
configs = [(30, 0.02), (50, 0.02), (100, 0.02), (30, 0.10), (50, 0.10)]
x_pos = np.arange(len(configs))
width = 0.22
gen_colors = {'linear': '#2166ac', 'mlp': '#4393c3', 'sigmoid': '#92c5de'}

for j, gen in enumerate(['linear', 'mlp', 'sigmoid']):
    f1s = []
    for d, a in configs:
        key = f'd={d}_a={a}_n=200_{gen}'
        v = a_nl.get(key, {}).get('pic_f1', 0)
        f1s.append(max(0, v) if v is not None and v >= 0 else 0)
    ax.bar(x_pos + (j - 1) * width, f1s, width, color=gen_colors[gen],
           edgecolor='#333333', linewidth=0.4, label=gen.title())

for j, gen in enumerate(['linear', 'mlp', 'sigmoid']):
    for i, (d, a) in enumerate(configs):
        key = f'd={d}_a={a}_n=200_{gen}'
        v = a_nl.get(key, {}).get('pic_f1', 0)
        if v is not None and v > 0.008:
            ax.text(x_pos[i] + (j-1)*width, v + 0.005, f'{v:.3f}', ha='center',
                   fontsize=5, rotation=90, va='bottom', color=gen_colors[gen])

x_labels = [f'd={d}\na={a:.02f}' for d, a in configs]
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, fontsize=6)
ax.set_ylabel(r'PIC+corr F$_1$', fontsize=7.5)
ax.legend(fontsize=6.5, ncol=3, loc='upper right', framealpha=0.8)
ax.text(-0.12, 1.04, 'C', transform=ax.transAxes, fontsize=12, fontweight='bold')
ax.set_title('Nonlinear generators: conclusion holds', fontsize=9, fontweight='bold', pad=4)

# ============ Panel D: Best-F1 summary ============
ax = axes[1, 1]
p2 = ckpt['p2']
p4 = ckpt.get('p4', {})

best_f1s = {
    'PIC+corr': max(v['pic_f1'] for v in p2.values()),
    'GOLEM':    max((v.get('golem_f1', 0) or 0 for v in p4.values()), default=0),
    'DAGMA':    max((v.get('dagma_f1', 0) or 0 for v in p4.values()), default=0),
    'NOTEARS':  max(v['note_f1'] for v in p2.values()),
    'CT':       max(v['ct_f1'] for v in p2.values()),
}
best_order = sorted(best_f1s, key=best_f1s.get, reverse=True)
best_vals = [best_f1s[m] for m in best_order]
best_colors = [METHOD_COLORS[m] for m in best_order]

bars = ax.barh(best_order, best_vals, color=best_colors,
               edgecolor='#333333', linewidth=0.5, height=0.55)
for bar, val in zip(bars, best_vals):
    ax.text(bar.get_width() + 0.007, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=7.5, fontweight='bold', color='#333333')
ax.set_xlabel(r'Best F$_1$ (40 phase-diagram configs)', fontsize=7.5)
ax.axvline(x=0.05, color='gray', linestyle='--', linewidth=0.8, alpha=0.35)
ax.set_xlim(0, max(best_vals) * 1.22)
ax.text(-0.12, 1.04, 'D', transform=ax.transAxes, fontsize=12, fontweight='bold')
ax.set_title('Best F$_1$ per method', fontsize=9, fontweight='bold', pad=4)

fig.suptitle('Figure 3 | Method comparison, wall boundaries, and robustness',
             fontsize=11, fontweight='bold', x=0.03, ha='left', y=0.995)

pdf_path = os.path.join(OUTDIR, 'fig3_method_landscape.pdf')
fig.savefig(pdf_path, dpi=300, facecolor='white', edgecolor='none',
            bbox_inches=mtransforms.Bbox.from_extents(0, 0, 12, 8))
plt.close()
print(f'fig3: {os.path.getsize(pdf_path)} bytes')
