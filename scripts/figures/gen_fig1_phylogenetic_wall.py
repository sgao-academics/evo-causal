"""
gen_fig1_phylogenetic_wall.py
Figure 1: The Phylogenetic Wall — 3x2 panel showing F1 across d x alpha for all 5 methods.
One script = one figure. Input: master_all.json. Output: figures/fig1_phylogenetic_wall.pdf
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8, 'axes.titlesize': 9.5, 'axes.labelsize': 7.5,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
})

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')  # portable
CKPT = os.path.join(ROOT, 'results', 'master_all.json')
OUTDIR = os.path.join(ROOT, 'figures')
os.makedirs(OUTDIR, exist_ok=True)

with open(CKPT, 'r', encoding='utf-8') as f:
    ckpt = json.load(f)

p2 = ckpt['p2']
p4 = ckpt.get('p4', {})
d_vals = [30, 50, 100, 150, 200]
a_vals = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
a_labels = ['0.02', '0.05', '0.10', '0.15', '0.20', '0.25', '0.30', '0.40']

METHOD_COLORS = {
    'PIC+corr': '#2166ac', 'CT': '#b2182b', 'NOTEARS': '#4daf4a',
    'DAGMA': '#ff7f00', 'GOLEM': '#984ea3',
}

# Checkpoint uses _K_ keys (design-time naming). Build lookup tables once.
def _fmt(d, a):
    """Format key as stored in checkpoint: d=30_K=0.02 etc."""
    return f'd={d}_K={a}'

# Pre-build matrices from checkpoint
pic_mat   = np.zeros((len(d_vals), len(a_vals)))
ct_mat    = np.zeros((len(d_vals), len(a_vals)))
note_mat  = np.zeros((len(d_vals), len(a_vals)))
dagma_mat = np.zeros((len(d_vals), len(a_vals)))
golem_mat = np.zeros((len(d_vals), len(a_vals)))

for di, d in enumerate(d_vals):
    for ai, a in enumerate(a_vals):
        key = _fmt(d, a)
        pic_mat[di, ai]   = p2.get(key, {}).get('pic_f1', 0)
        ct_mat[di, ai]    = p2.get(key, {}).get('ct_f1', 0)
        note_mat[di, ai]  = p2.get(key, {}).get('note_f1', 0)
        dagma_mat[di, ai] = p4.get(key, {}).get('dagma_f1', 0) or 0
        golem_mat[di, ai] = p4.get(key, {}).get('golem_f1', 0) or 0

# Verify data loaded (PIC d=30, α=0.02 should be 0.305)
assert pic_mat[0, 0] > 0.25, f'PIC data load failed: {pic_mat[0,0]:.4f}'
assert ct_mat[4, 0] > 0.02, f'CT data load failed: {ct_mat[4,0]:.4f}'

methods = [
    ('PIC+correlation',    pic_mat,   METHOD_COLORS['PIC+corr']),
    ('Causal Transformer', ct_mat,    METHOD_COLORS['CT']),
    ('NOTEARS',            note_mat,  METHOD_COLORS['NOTEARS']),
    ('DAGMA',              dagma_mat, METHOD_COLORS['DAGMA']),
    ('GOLEM',              golem_mat, METHOD_COLORS['GOLEM']),
]

# Heatmap colormap: white at zero, progressive red for positive F1
cmap = LinearSegmentedColormap.from_list('wall',
    ['#f7f7f7', '#fddbc7', '#f4a582', '#d6604d', '#b2182b', '#67001f'], N=256)

# Create figure with constrained_layout to avoid overlap
fig, axes = plt.subplots(2, 3, figsize=(14, 8.5),
                         gridspec_kw={'width_ratios': [1, 1, 1],
                                      'height_ratios': [1, 1]})

panel_labels = ['A', 'B', 'C', 'D', 'E']
vmax_global = 0.35

for idx, (name, mat, accent) in enumerate(methods):
    ax = axes[idx // 3, idx % 3]
    mat_masked = np.ma.masked_where(mat == 0, mat)
    im = ax.imshow(mat_masked, aspect='auto', cmap=cmap,
                   vmin=0, vmax=vmax_global, origin='upper')

    # Annotate non-zero cells
    for di in range(len(d_vals)):
        for ai in range(len(a_vals)):
            if mat[di, ai] > 0.001:
                color = 'white' if mat[di, ai] > 0.18 else '#222222'
                ax.text(ai, di, f'{mat[di, ai]:.3f}', ha='center', va='center',
                       fontsize=6, fontweight='bold', color=color)

    # Wall dashed line at alpha=0.10
    ax.axvline(x=2, ymin=0, ymax=1, color=accent, linestyle='--',
               linewidth=1.8, alpha=0.8)

    # WALL label
    ax.text(0.95, 0.05, 'WALL', transform=ax.transAxes,
            fontsize=7, fontweight='bold', color=accent, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=accent, alpha=0.85, linewidth=1))

    ax.set_xticks(range(len(a_vals)))
    ax.set_xticklabels(a_labels, rotation=45, ha='right', fontsize=6)
    ax.set_yticks(range(len(d_vals)))
    ax.set_yticklabels([str(d) for d in d_vals], fontsize=6)
    ax.set_xlabel(r'Phylogenetic signal $\alpha$', fontsize=7, labelpad=1)
    ax.set_ylabel(r'Dimension $d$', fontsize=7, labelpad=1)

    # Panel label (bold, top-left)
    ax.text(-0.08, 1.02, panel_labels[idx], transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='bottom', ha='left')

    # Title in method-specific color
    ax.set_title(name, fontsize=9, fontweight='bold', pad=4, color=accent)

# Hide empty 6th cell
axes[1, 2].set_visible(False)

# Shared colorbar
cbar_ax = fig.add_axes([0.92, 0.16, 0.012, 0.68])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label(r'F$_1$ score', fontsize=8.5, labelpad=4)
cbar.ax.tick_params(labelsize=6.5)
cbar.outline.set_linewidth(0.5)

# Suptitle
fig.suptitle('Figure 1 | The phylogenetic wall across five causal discovery methods',
             fontsize=11, fontweight='bold', x=0.03, ha='left', y=0.995)

plt.subplots_adjust(left=0.05, right=0.90, top=0.93, bottom=0.07,
                    hspace=0.42, wspace=0.32)

pdf_path = os.path.join(OUTDIR, 'fig1_phylogenetic_wall.pdf')
fig.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print(f'fig1: {os.path.getsize(pdf_path)} bytes')
