"""
gen_fig2_ct_mechanism.py
Figure 2: CT Failure — 2x2 panel: (A) h(W) and edges vs alpha, (B) architecture ablation,
(C) convergence summary, (D) n-scaling.
Input: master_all.json. Output: figures/fig2_ct_mechanism.pdf
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
    'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 7.5,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'axes.spines.top': False, 'axes.spines.right': False,
})

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')  # portable
CKPT = os.path.join(ROOT, 'results', 'master_all.json')
OUTDIR = os.path.join(ROOT, 'figures')
os.makedirs(OUTDIR, exist_ok=True)

with open(CKPT, 'r', encoding='utf-8') as f:
    ckpt = json.load(f)

CT_COLOR = '#b2182b'; PIC_COLOR = '#2166ac'; H_COLOR = '#b2182b'
EDGE_COLOR = '#444444'

fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
fig.subplots_adjust(hspace=0.52, wspace=0.38, left=0.07, right=0.93, top=0.92, bottom=0.07)

# ============ Panel A ============
ax = axes[0, 0]
ct_trace = ckpt['p1']['ct_trace']
K_keys = sorted(ct_trace.keys(), key=lambda x: float(x.split('=')[1]))
alphas = [float(k.split('=')[1]) for k in K_keys]
h_vals  = [ct_trace[k]['h_plateau'] for k in K_keys]

ax2 = ax.twinx()
ax.plot(alphas, h_vals, 's-', color=H_COLOR, linewidth=2, markersize=6,
        markerfacecolor='white', markeredgewidth=1.5, label=r'$h(W)$ (left)')
ax2.plot(alphas, [ct_trace[k]['edges'] for k in K_keys], 'D--', color=EDGE_COLOR,
         linewidth=1.5, markersize=6, markerfacecolor='white', markeredgewidth=1.5,
         label='Edges (right)')
ax.set_xlabel(r'Phylogenetic signal $\alpha$', fontsize=7.5)
ax.set_ylabel(r'$h(W)$', fontsize=7.5, color=H_COLOR)
ax2.set_ylabel('Edges', fontsize=7.5, color=EDGE_COLOR)
ax.tick_params(axis='y', colors=H_COLOR, labelsize=6)
ax2.tick_params(axis='y', colors=EDGE_COLOR, labelsize=6)
ax2.set_ylim(-0.8, 2.5)
ax2.set_yticks([0])

# Single clean annotation
ax.annotate(r'Converges to ${\sim}2{\times}10^{-4}$', xy=(0.15, 2.7e-4),
            xytext=(0.05, 3.2e-4), arrowprops=dict(arrowstyle='->', color=H_COLOR, lw=1),
            fontsize=6.5, color=H_COLOR, fontweight='bold')
ax.annotate('Edges = 0', xy=(0.15, 0), xytext=(0.12, 1.8),
            arrowprops=dict(arrowstyle='->', color=EDGE_COLOR, lw=1),
            fontsize=6.5, color=EDGE_COLOR, fontweight='bold')

# Combined legend
lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labs1 + labs2, fontsize=6, loc='center right',
          framealpha=0.85, handlelength=1.2)

ax.text(-0.10, 1.02, 'A', transform=ax.transAxes, fontsize=12, fontweight='bold')
ax.set_title('DAG constraint converges, edges stay zero', fontsize=9, fontweight='bold', pad=5)

# ============ Panel B ============
ax = axes[0, 1]
arch = ckpt['p3']['arch']
arch_keys = sorted(arch.keys())
x_labels = [k.replace('dm=', 'd_m=').replace('_nl=', ', L=') for k in arch_keys]
colors_bars = ['#2166ac', '#4393c3', '#92c5de', '#d6604d', '#b2182b', '#67001f']

bars = ax.bar(range(len(arch_keys)), [arch[k]['f1_mean'] for k in arch_keys],
              color=colors_bars, edgecolor='#333333', linewidth=0.5, width=0.55)
ax.set_xticks(range(len(arch_keys)))
ax.set_xticklabels(x_labels, rotation=45, fontsize=5.5, ha='right')
ax.set_ylabel(r'F$_1$', fontsize=7.5)
ax.set_ylim(0, 0.06)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, 0.002, '0', ha='center', fontsize=5.5, color='#999999')
ax.text(-0.10, 1.02, 'B', transform=ax.transAxes, fontsize=12, fontweight='bold')
ax.set_title('Architecture variants (all F$_1{=}0$)', fontsize=9, fontweight='bold', pad=5)

# ============ Panel C ============
ax = axes[1, 0]
conv = ckpt['p1']['ct_convergence']
seeds = sorted(conv.keys(), key=lambda x: int(x.split('=')[1]))
h_1200 = [ct_trace.get('K=0.15', {}).get('h_plateau', 0)] * 3
h_3000 = [conv[s]['h'] for s in seeds]

x = np.arange(3)
w = 0.3
b1 = ax.bar(x - w/2, h_1200, w, color='#92c5de', edgecolor='#333333', linewidth=0.5,
            label=r'$h(W)$, 1,200 ep')
b2 = ax.bar(x + w/2, h_3000, w, color=H_COLOR, edgecolor='#333333', linewidth=0.5,
            label=r'$h(W)$, 3,000 ep')
ax.set_xticks(x)
ax.set_xticklabels(['seed 42', 'seed 123', 'seed 456'], fontsize=6.5)
ax.set_ylabel(r'$h(W)$', fontsize=7.5)
ax.legend(fontsize=6, loc='upper left', framealpha=0.8)

# Annotate values above bars
for bar, val in zip(b1, h_1200):
    ax.text(bar.get_x() + w/2, val * 1.3, f'{val:.1e}', ha='center',
            fontsize=5, rotation=90, va='bottom', color='#555555')
for bar, val in zip(b2, h_3000):
    ax.text(bar.get_x() + w/2, val * 1.3, f'{val:.1e}', ha='center',
            fontsize=5, rotation=90, va='bottom', color='#555555')

# Edge count note (inside plot, top)
ax.text(0.5, 0.82, 'Edges = 0 for all seeds', transform=ax.transAxes,
        ha='center', fontsize=6.5, color='#a50f15', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#fff5f5',
        edgecolor='#a50f15', alpha=0.85, linewidth=0.5))

ax.text(1.05, 1.02, 'C', transform=ax.transAxes, fontsize=12, fontweight='bold', ha='right')
ax.set_title('Longer training deepens $h(W)$ only', fontsize=9, fontweight='bold', pad=5)

# ============ Panel D ============
ax = axes[1, 1]
nsc = ckpt['p3']['n_scaling']
n_keys = sorted(nsc.keys(), key=lambda x: nsc[x]['n'])
n_vals   = [nsc[k]['n'] for k in n_keys]
pic_vals = [nsc[k]['pic_f1'] for k in n_keys]
ct_vals  = [nsc[k]['ct_f1'] for k in n_keys]

ax.plot(n_vals, pic_vals, 'o-', color=PIC_COLOR, linewidth=2, markersize=7,
        markerfacecolor='white', markeredgewidth=1.5, label='PIC+corr')
ax.plot(n_vals, ct_vals, 's--', color=CT_COLOR, linewidth=2, markersize=7,
        markerfacecolor='white', markeredgewidth=1.5, label='CT')
ax.set_xlabel('Sample size $n$', fontsize=7.5)
ax.set_ylabel(r'F$_1$', fontsize=7.5)
ax.set_xscale('log')
ax.legend(fontsize=6.5, loc='upper left', framealpha=0.8)
ax.set_ylim(-0.002, 0.035)
ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.5)
ax.annotate(r'$d{=}100,\alpha{=}0.15$', xy=(800, 0.0061), xytext=(250, 0.025),
            arrowprops=dict(arrowstyle='->', color='#a50f15', lw=1),
            fontsize=6.5, color='#a50f15', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
            edgecolor='#a50f15', alpha=0.8, linewidth=0.5))
ax.text(-0.10, 1.02, 'D', transform=ax.transAxes, fontsize=12, fontweight='bold')
ax.set_title('Sample size scaling', fontsize=9, fontweight='bold', pad=5)

fig.suptitle('Figure 2 | Causal Transformer failure analysis',
             fontsize=11, fontweight='bold', x=0.03, ha='left', y=0.995)

pdf_path = os.path.join(OUTDIR, 'fig2_ct_mechanism.pdf')
fig.savefig(pdf_path, dpi=300, facecolor='white', edgecolor='none',
            bbox_inches=mtransforms.Bbox.from_extents(0, 0, 12.5, 8.5))
plt.close()
print(f'fig2: {os.path.getsize(pdf_path)} bytes')
