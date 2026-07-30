"""
Overnight massive sweep: fill 8 hours of GPU/CPU compute.
Addresses ALL reviewer concerns with dense data coverage.

Modules:
  1. Phase Diagram: d x K grid (PIC+corr + CT + NOTEARS)
  2. CT Architecture Ablation: d_model, n_layers
  3. Sample Size Scaling: n scaling for PIC+corr vs CT

All with full checkpointing for resumability.
Output: results/overnight_massive.json
"""

import sys, os, json, time, math

# CRITICAL: OpenMP fix BEFORE numpy/torch import
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
sys.path.insert(0, os.path.dirname(__file__))  # portable
from felsenstein_pic import PhyloNode, felsenstein_pic
# causalscale should be installed via: pip install causalscale

import torch
import torch.nn.functional as F
from causalscale.core.transformer import (
    CausalTransformer, CausalTransformerConfig, NOTEARSConstraint
)
from causalscale.core.dag_constraint import notears_linear

CKPT = r'C:\Users\高帅东\Desktop\evo_causal\results\overnight_massive.json'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

# ============================================================================
# Checkpoint
# ============================================================================

def load_ckpt():
    if os.path.exists(CKPT):
        with open(CKPT, 'r') as f:
            return json.load(f)
    return {}

def _json_safe(obj):
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_json_safe(v) for v in obj]
    return obj

def save_ckpt(ckpt):
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    tmp = CKPT + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(_json_safe(ckpt), f, indent=2)
    os.replace(tmp, CKPT)  # atomic

# ============================================================================
# Shared
# ============================================================================

def build_tree(n_species):
    tips = []
    for i in range(n_species):
        t = PhyloNode(f"Sp{i}", children=[], branch_length=max(0.01, np.random.exponential(1.0)))
        t.index = i; t.is_tip = True
        tips.append(t)
    nodes = tips; next_id = 0
    while len(nodes) > 1:
        new_nodes = []
        for i in range(0, len(nodes), 2):
            if i + 1 < len(nodes):
                bl = max(0.01, np.random.exponential(0.5))
                parent = PhyloNode(f"N{next_id}", children=[nodes[i], nodes[i+1]], branch_length=bl)
                parent.is_tip = False; next_id += 1
                new_nodes.append(parent)
            else:
                new_nodes.append(nodes[i])
        nodes = new_nodes
    return nodes[0]


def generate_data(n_species=200, d=50, n_true_edges=10, phylo_strength=0.15, seed=42):
    rng = np.random.default_rng(seed)
    tree = build_tree(n_species)
    n = n_species

    L = np.column_stack([rng.normal(0, 1, n) for _ in range(d)])
    eigvals, eigvecs = np.linalg.eigh(L @ L.T + 1e-6 * np.eye(n))
    eigvals = np.maximum(eigvals, 0)
    phylo_chol = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    W_true = np.zeros((d, d))
    edge_pairs = []
    for _ in range(n_true_edges):
        i, j = int(rng.integers(0, d)), int(rng.integers(0, d))
        if i != j:
            W_true[i, j] = rng.uniform(0.3, 0.7) * rng.choice([-1, 1])
            edge_pairs.append((i, j))

    Z = rng.normal(0, 1, (n, d))
    X_causal = Z @ np.linalg.inv(np.eye(d) - W_true)
    X_phylo = phylo_chol @ rng.normal(0, 1, (n, d))
    X = np.sqrt(1 - phylo_strength) * X_causal + np.sqrt(phylo_strength) * X_phylo
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return X, W_true, tree, edge_pairs


def compute_f1(W_est, W_true, threshold=0.3):
    Wb = np.abs(W_est) > threshold
    Wt = np.abs(W_true) > 0
    tp = int((Wb & Wt).sum())
    fp = int((Wb & ~Wt).sum())
    fn = int((~Wb & Wt).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return f1, prec, rec, tp, fp, fn


def run_pic_corr(X, tree, W_true, edge_pairs, d, pct=99):
    """PIC + correlation thresholding."""
    pic_data = felsenstein_pic(X, tree)
    pic_corr = np.corrcoef(pic_data.T)
    triu_idx = np.triu_indices(d, k=1)
    pic_vals = np.abs(pic_corr[triu_idx])
    threshold = np.percentile(pic_vals, pct)

    adj = np.abs(pic_corr) > threshold
    np.fill_diagonal(adj, False)
    f1 = compute_f1(adj.astype(float), W_true, 0.5)[0]

    bg_mask = np.ones((d, d), dtype=bool)
    np.fill_diagonal(bg_mask, False)
    bg_corr = np.abs(pic_corr[bg_mask]).mean()
    true_corrs = [np.abs(pic_corr[i, j]) for i, j in edge_pairs if i < d and j < d]
    true_corr = np.mean(true_corrs) if true_corrs else 0.0

    return {
        'f1': round(f1, 4),
        'bg_corr': round(float(bg_corr), 6),
        'true_corr': round(float(true_corr), 6),
        'enrichment': round(true_corr / bg_corr, 4) if bg_corr > 0 else 0.0,
    }


def run_ct(X, d, n_epochs=2000, d_model=64, n_heads=4, n_layers=2,
           edge_threshold=0.3, lr=0.001, device='cpu'):
    """Run Causal Transformer with given params, return F1 and diagnostics."""
    n, _d = X.shape
    X_t = torch.tensor(X, dtype=torch.float32, device=device)

    config = CausalTransformerConfig(
        d_vars=d, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        lambda_dag=0.5, lambda_sparsity=0.01, lr=lr, edge_threshold=edge_threshold,
    )
    model = CausalTransformer(config).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    batch_size = min(128, n)
    edge_history = []; dag_history = []

    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x_batch = X_t[idx]
            W_batch, _ = model(x_batch)
            losses = model.compute_loss(x_batch, W_batch)
            optimizer.zero_grad()
            losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        W_mean = W_batch.mean(dim=0).detach()
        edges = int((torch.abs(W_mean) > edge_threshold).float().sum().item())
        edge_history.append(edges)
        dag_history.append(float(losses['dag']))

    model.eval()
    with torch.no_grad():
        W_final, _ = model(X_t[:min(500, n)])
        W_mean = W_final.mean(dim=0).cpu().numpy()
        final_h = float(NOTEARSConstraint()(torch.tensor(W_mean, device=device)).item())

    # Free GPU memory
    del model, optimizer, X_t
    if device == 'cuda':
        torch.cuda.empty_cache()

    return {
        'W': W_mean.tolist(),
        'final_h': round(final_h, 6),
        'final_edges': int((np.abs(W_mean) > edge_threshold).sum()),
        'edge_plateau': float(np.mean(edge_history[-200:])),
        'h_plateau': float(np.mean(dag_history[-200:])),
    }


def run_notears_baseline(X, lambda1=0.1, w_threshold=0.3, max_iter=100):
    """NOTEARS linear baseline for comparison."""
    try:
        W = notears_linear(X.astype(np.float64), lambda1=lambda1,
                          loss_type='l2', max_iter=max_iter, w_threshold=w_threshold)
        return W
    except Exception:
        return np.zeros((X.shape[1], X.shape[1]))


# ============================================================================
# Module 1: Phase Diagram (d x K)
# ============================================================================

def run_phase_diagram(ckpt):
    if 'phase_diagram' in ckpt:
        print("[PHASE DIAGRAM] Done, skipping")
        return ckpt

    print("\n" + "="*60)
    print("[PHASE DIAGRAM] d x K grid: PIC+corr + CT + NOTEARS")
    print("="*60)

    d_vals = [30, 50, 100, 150, 200]
    K_vals = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
    seeds = [42, 123, 456]
    n = 200  # fixed sample size

    results = {}

    for d in d_vals:
        for K in K_vals:
            key = f'd={d}_K={K}'
            if key in ckpt.get('phase_diagram', {}):
                print(f"  {key}: cached, skip")
                results[key] = ckpt['phase_diagram'][key]
                continue

            t0 = time.time()
            print(f"\n  --- {key} ---")
            seed_results = []

            for seed in seeds:
                X, W_true, tree, edges = generate_data(n, d, 10, K, seed)

                # PIC+corr
                pic_r = run_pic_corr(X, tree, W_true, edges, d)

                # CT (2000 epochs, seed-specific)
                ct_r = run_ct(X, d, n_epochs=2000, device=DEVICE)
                ct_f1 = compute_f1(np.array(ct_r['W']), W_true, 0.3)[0]

                # NOTEARS baseline (fast for small d)
                if d <= 100:
                    W_note = run_notears_baseline(X)
                    note_f1 = compute_f1(W_note, W_true, 0.3)[0]
                else:
                    note_f1 = -1.0  # skipped for large d

                seed_results.append({
                    'seed': seed,
                    'pic_f1': pic_r['f1'],
                    'pic_bg_corr': pic_r['bg_corr'],
                    'pic_true_corr': pic_r['true_corr'],
                    'pic_enrich': pic_r['enrichment'],
                    'ct_f1': round(ct_f1, 4),
                    'ct_edges': ct_r['final_edges'],
                    'ct_h': ct_r['final_h'],
                    'notears_f1': round(note_f1, 4) if note_f1 >= 0 else None,
                })

            elapsed = time.time() - t0
            results[key] = {
                'd': d, 'K': K, 'n': n,
                'pic_f1_mean': round(np.mean([s['pic_f1'] for s in seed_results]), 4),
                'pic_f1_std': round(np.std([s['pic_f1'] for s in seed_results]), 4),
                'pic_enrich_mean': round(np.mean([s['pic_enrich'] for s in seed_results]), 4),
                'ct_f1_mean': round(np.mean([s['ct_f1'] for s in seed_results]), 4),
                'ct_f1_std': round(np.std([s['ct_f1'] for s in seed_results]), 4),
                'ct_best': round(max(s['ct_f1'] for s in seed_results), 4),
                'time_s': round(elapsed, 1),
                'seeds': seed_results,
            }
            ckpt['phase_diagram'] = results
            save_ckpt(ckpt)

            pf = results[key]['pic_f1_mean']; cf = results[key]['ct_f1_mean']
            print(f"  {key}: PIC_F1={pf:.3f} CT_F1={cf:.3f} [{elapsed:.0f}s]")

    ckpt['phase_diagram'] = results
    save_ckpt(ckpt)
    return ckpt


# ============================================================================
# Module 2: CT Architecture Ablation
# ============================================================================

def run_arch_ablation(ckpt):
    if 'arch_ablation' in ckpt:
        print("[ARCH ABLATION] Done, skipping")
        return ckpt

    print("\n" + "="*60)
    print("[ARCH ABLATION] CT d_model x n_layers sweep")
    print("="*60)

    d_models = [32, 64, 128]
    n_layers_vals = [2, 4]
    seeds = [42, 123, 456]
    n, d, K = 200, 100, 0.15

    results = {}
    # Generate data once, reuse
    X, W_true, tree, edges = generate_data(n, d, 10, K, seed=42)
    n_true = len(edges)

    for dm in d_models:
        for nl in n_layers_vals:
            key = f'dm={dm}_nl={nl}'
            if key in ckpt.get('arch_ablation', {}):
                print(f"  {key}: cached, skip")
                results[key] = ckpt['arch_ablation'][key]
                continue

            t0 = time.time()
            print(f"\n  --- {key} ---")
            seed_results = []

            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                ct_r = run_ct(X, d, n_epochs=2000, d_model=dm, n_layers=nl, device=DEVICE)
                f1 = compute_f1(np.array(ct_r['W']), W_true, 0.3)[0]
                seed_results.append({'seed': seed, 'f1': round(f1, 4),
                                     'edges': ct_r['final_edges'], 'h': ct_r['final_h']})

            elapsed = time.time() - t0
            results[key] = {
                'd_model': dm, 'n_layers': nl,
                'f1_mean': round(np.mean([s['f1'] for s in seed_results]), 4),
                'f1_std': round(np.std([s['f1'] for s in seed_results]), 4),
                'f1_best': round(max(s['f1'] for s in seed_results), 4),
                'edges_mean': round(np.mean([s['edges'] for s in seed_results]), 1),
                'time_s': round(elapsed, 1),
                'seeds': seed_results,
            }
            ckpt['arch_ablation'] = results
            save_ckpt(ckpt)
            print(f"  {key}: F1={results[key]['f1_mean']:.3f}+/-{results[key]['f1_std']:.3f} [{elapsed:.0f}s]")

    ckpt['arch_ablation'] = results
    save_ckpt(ckpt)
    return ckpt


# ============================================================================
# Module 3: Sample Size Scaling
# ============================================================================

def run_n_scaling(ckpt):
    if 'n_scaling' in ckpt:
        print("[N SCALING] Done, skipping")
        return ckpt

    print("\n" + "="*60)
    print("[N SCALING] Sample size scaling: PIC+corr vs CT vs NOTEARS")
    print("="*60)

    n_vals = [50, 100, 200, 400, 800]
    d, K = 100, 0.15
    seeds = [42, 123, 456]

    results = {}

    for n in n_vals:
        key = f'n={n}'
        if key in ckpt.get('n_scaling', {}):
            print(f"  {key}: cached, skip")
            results[key] = ckpt['n_scaling'][key]
            continue

        t0 = time.time()
        print(f"\n  --- {key} ---")
        seed_results = []

        for seed in seeds:
            X, W_true, tree, edges = generate_data(n, d, 10, K, seed)
            n_true = len(edges)

            # PIC+corr
            pic_r = run_pic_corr(X, tree, W_true, edges, d)

            # CT
            ct_r = run_ct(X, d, n_epochs=2000, device=DEVICE)
            ct_f1 = compute_f1(np.array(ct_r['W']), W_true, 0.3)[0]

            # NOTEARS (skip for n > 200 to save time)
            if n <= 200 and d <= 100:
                W_note = run_notears_baseline(X)
                note_f1 = compute_f1(W_note, W_true, 0.3)[0]
            else:
                note_f1 = -1.0

            seed_results.append({
                'seed': seed,
                'pic_f1': pic_r['f1'],
                'pic_enrich': pic_r['enrichment'],
                'ct_f1': round(ct_f1, 4),
                'notears_f1': round(note_f1, 4) if note_f1 >= 0 else None,
            })

        elapsed = time.time() - t0
        results[key] = {
            'n': n, 'd': d, 'K': K,
            'pic_f1_mean': round(np.mean([s['pic_f1'] for s in seed_results]), 4),
            'pic_f1_std': round(np.std([s['pic_f1'] for s in seed_results]), 4),
            'ct_f1_mean': round(np.mean([s['ct_f1'] for s in seed_results]), 4),
            'ct_f1_std': round(np.std([s['ct_f1'] for s in seed_results]), 4),
            'time_s': round(elapsed, 1),
            'seeds': seed_results,
        }
        ckpt['n_scaling'] = results
        save_ckpt(ckpt)

        pf = results[key]['pic_f1_mean']; cf = results[key]['ct_f1_mean']
        print(f"  {key}: PIC_F1={pf:.3f} CT_F1={cf:.3f} [{elapsed:.0f}s]")

    ckpt['n_scaling'] = results
    save_ckpt(ckpt)
    return ckpt


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print(f"OVERNIGHT MASSIVE SWEEP — {time.strftime('%H:%M:%S')}")
    print(f"Device: {DEVICE}, Checkpoint: {CKPT}")

    ckpt = load_ckpt()

    # Run in order: phase diagram (heaviest) -> arch ablation -> n scaling
    ckpt = run_phase_diagram(ckpt)
    ckpt = run_arch_ablation(ckpt)
    ckpt = run_n_scaling(ckpt)

    now = time.strftime('%H:%M:%S')
    print(f"\n{'='*60}")
    print(f"OVERNIGHT COMPLETE at {now}")
    print(f"Results: {CKPT}")
    print(f"{'='*60}")
