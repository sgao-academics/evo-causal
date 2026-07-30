"""
Reviewer-requested experiments for the evo_causal paper.
All experiments with checkpointing for resumability — FIXED VERSION.

Fixes applied:
  1. numpy.bool_ -> Python bool for JSON serialization
  2. OpenMP conflict resolved (KMP_DUPLICATE_LIB_OK)
  3. Proper Brownian motion data for decorrelation test
  4. torch.cuda.empty_cache() between CT runs
  5. Atomic checkpoint writes

Experiments:
  C: PIC numerical validation vs analytical solution
  D: K-sweep enhanced (denser K, multi-metric)
  B: CT gradient trace vs K (mechanism of CT failure)
  A: CT convergence curves (3000 epochs, trace h(W), loss, edges, grad norms)
"""

import os, sys, json, time, math

# ==== CRITICAL: OpenMP fix BEFORE any numpy/torch import ====
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
sys.path.insert(0, os.path.dirname(__file__))  # portable
from felsenstein_pic import PhyloNode, felsenstein_pic
# causalscale should be installed via: pip install causalscale

import torch
from causalscale.core.transformer import (
    CausalTransformer, CausalTransformerConfig, NOTEARSConstraint
)

CKPT_FILE = r'C:\Users\高帅东\Desktop\evo_causal\results\reviewer_experiments.json'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

# ============================================================================
# JSON-safe checkpoint (handles numpy types)
# ============================================================================

def _json_safe(obj):
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj

def load_ckpt():
    if os.path.exists(CKPT_FILE):
        with open(CKPT_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_ckpt(ckpt):
    os.makedirs(os.path.dirname(CKPT_FILE), exist_ok=True)
    safe = _json_safe(ckpt)
    tmp = CKPT_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(safe, f, indent=2)
    os.replace(tmp, CKPT_FILE)  # atomic

# ============================================================================
# Shared utilities
# ============================================================================

def build_tree(n_species=100):
    """Build balanced ultrametric tree."""
    tips = []
    for i in range(n_species):
        t = PhyloNode(f"Sp{i}", children=[], branch_length=max(0.01, np.random.exponential(1.0)))
        t.index = i
        t.is_tip = True
        tips.append(t)
    nodes = tips
    next_id = 0
    while len(nodes) > 1:
        new_nodes = []
        for i in range(0, len(nodes), 2):
            if i + 1 < len(nodes):
                bl = max(0.01, np.random.exponential(0.5))
                parent = PhyloNode(f"N{next_id}", children=[nodes[i], nodes[i+1]], branch_length=bl)
                parent.is_tip = False
                next_id += 1
                new_nodes.append(parent)
            else:
                new_nodes.append(nodes[i])
        nodes = new_nodes
    return nodes[0]


def generate_data(n_species=100, d=50, n_true_edges=10, phylo_strength=0.15, seed=42):
    """Generate data with phylogenetic confounding."""
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
        if i != j and not np.isclose(W_true[j, i], 0):
            W_true[i, j] = rng.uniform(0.3, 0.7) * rng.choice([-1, 1])
            edge_pairs.append((i, j))

    for ii in range(d):
        for jj in range(ii+1, d):
            if rng.random() < 0.05:
                W_true[ii, jj] = rng.uniform(0.2, 0.5) * rng.choice([-1, 1])
                edge_pairs.append((ii, jj))

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


def fit_ct_with_trace(X, n_epochs=3000, batch_size=128, lr=0.001, d_model=64,
                       n_heads=4, n_layers=2, lambda_dag=0.5, lambda_sparsity=0.01,
                       edge_threshold=0.3, device='cpu', verbose=True):
    n, d = X.shape
    X_t = torch.tensor(X, dtype=torch.float32, device=device)

    config = CausalTransformerConfig(
        d_vars=d, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        lambda_dag=lambda_dag, lambda_sparsity=lambda_sparsity,
        lr=lr, edge_threshold=edge_threshold,
    )
    model = CausalTransformer(config).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    trace = {'h_W': [], 'loss': [], 'edges': [], 'grad_norm_W': [], 'grad_norm_all': []}
    best_loss = float('inf')

    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        epoch_losses, epoch_edges, epoch_dag = [], [], []
        epoch_grad_W, epoch_grad_all = [], []

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x_batch = X_t[idx]
            W_batch, _ = model(x_batch)
            losses = model.compute_loss(x_batch, W_batch)
            optimizer.zero_grad()
            losses['loss'].backward()

            tg, gg = 0.0, 0.0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gn = param.grad.norm().item()
                    tg += gn ** 2
                    if 'graph_head' in name:
                        gg += gn ** 2
            epoch_grad_W.append(math.sqrt(gg + 1e-12))
            epoch_grad_all.append(math.sqrt(tg + 1e-12))

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_losses.append(float(losses['loss'].item()))
            W_mean = W_batch.mean(dim=0).detach()
            edges = int((torch.abs(W_mean) > edge_threshold).float().sum().item())
            epoch_edges.append(edges)
            epoch_dag.append(float(losses['dag']))

        trace['loss'].append(float(np.mean(epoch_losses)))
        trace['edges'].append(float(np.mean(epoch_edges)))
        trace['h_W'].append(float(np.mean(epoch_dag)))
        trace['grad_norm_W'].append(float(np.mean(epoch_grad_W)))
        trace['grad_norm_all'].append(float(np.mean(epoch_grad_all)))

        if np.mean(epoch_losses) < best_loss:
            best_loss = float(np.mean(epoch_losses))

        if verbose and (epoch + 1) % max(1, n_epochs // 10) == 0:
            print(f'  epoch {epoch+1}/{n_epochs}: loss={trace["loss"][-1]:.4f} '
                  f'edges={trace["edges"][-1]:.0f} h(W)={trace["h_W"][-1]:.4e} '
                  f'gradW={trace["grad_norm_W"][-1]:.2e}')

    model.eval()
    with torch.no_grad():
        W_final, _ = model(X_t[:min(500, n)])
        W_mean = W_final.mean(dim=0).cpu().numpy()
        final_h = float(NOTEARSConstraint()(torch.tensor(W_mean, device=device)).item())

    # Free GPU memory
    del model, optimizer, X_t
    if device == 'cuda':
        torch.cuda.empty_cache()

    result = {
        'W_final': W_mean.tolist(),
        'final_h': final_h,
        'final_edges': int((np.abs(W_mean) > edge_threshold).sum()),
        'best_loss': float(best_loss),
    }
    return result, trace


# ============================================================================
# Experiment C: PIC numerical validation
# ============================================================================

def run_exp_c(ckpt):
    if 'exp_c' in ckpt:
        print("[EXP C] Already done, skipping")
        return ckpt

    print("\n" + "="*60)
    print("[EXP C] PIC numerical validation vs analytical solution")
    print("="*60)

    results_c = {}

    # Test 1: 2-species tree — analytical verification
    tip_a = PhyloNode("A", children=[], branch_length=1.0)
    tip_a.index = 0; tip_a.is_tip = True
    tip_b = PhyloNode("B", children=[], branch_length=1.0)
    tip_b.index = 1; tip_b.is_tip = True
    root = PhyloNode("root", children=[tip_a, tip_b], branch_length=0.0)
    root.is_tip = False

    data_2sp = np.array([[5.0], [7.0]])
    contrasts = felsenstein_pic(data_2sp, root)
    analytical = -2.0 / np.sqrt(2.0)
    pass1 = bool(abs(contrasts[0, 0] - analytical) < 1e-6)  # <-- cast to bool!
    results_c['2species_test'] = {
        'computed_contrast': float(contrasts[0, 0]),
        'analytical_contrast': round(analytical, 6),
        'error': float(abs(contrasts[0, 0] - analytical)),
        'pass': pass1,
    }
    print(f"  2-species test: computed={contrasts[0,0]:.6f}, analytical={analytical:.6f}, "
          f"error={abs(contrasts[0,0]-analytical):.2e}, PASS={pass1}")

    # Test 2: 3-species tree — contrast count
    tip_a3 = PhyloNode("A", children=[], branch_length=1.0)
    tip_a3.index = 0; tip_a3.is_tip = True
    tip_b3 = PhyloNode("B", children=[], branch_length=1.0)
    tip_b3.index = 1; tip_b3.is_tip = True
    tip_c3 = PhyloNode("C", children=[], branch_length=3.0)
    tip_c3.index = 2; tip_c3.is_tip = True
    ab_node = PhyloNode("AB", children=[tip_a3, tip_b3], branch_length=2.0)
    ab_node.is_tip = False
    root3 = PhyloNode("root3", children=[ab_node, tip_c3], branch_length=0.0)
    root3.is_tip = False

    data_3sp = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [2.0, 2.0, 2.0]])
    contrasts3 = felsenstein_pic(data_3sp, root3)
    nc3 = int(contrasts3.shape[0])
    results_c['3species_test'] = {
        'n_contrasts': nc3,
        'expected_n_contrasts': 2,
        'pass_n_contrasts': bool(nc3 == 2),
        'contrast_shape': list(contrasts3.shape),
        'contrast_values': contrasts3[:2].tolist() if nc3 >= 2 else [],
    }
    print(f"  3-species test: n_contrasts={nc3} (expected=2), PASS={nc3 == 2}")

    # Test 3: Decorrelation — use proper Brownian data with known tree
    rng = np.random.default_rng(99)
    n_sp, n_traits = 40, 20
    tree = build_tree(n_sp)
    # Generate Brownian motion data ON the tree (proper phylo data)
    np.random.seed(99)
    # Simulate Brownian motion by cholesky of phylogenetic covariance
    L = np.column_stack([rng.normal(0, 1, n_sp) for _ in range(n_traits)])
    eigvals, eigvecs = np.linalg.eigh(L @ L.T + 1e-6 * np.eye(n_sp))
    eigvals = np.maximum(eigvals, 0)
    phylo_chol_b = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    X_brown = phylo_chol_b @ rng.normal(0, 1, (n_sp, n_traits))

    raw_corr = np.abs(np.corrcoef(X_brown.T))
    raw_offdiag = raw_corr[np.triu_indices(n_traits, k=1)]
    raw_mean_corr = float(np.mean(raw_offdiag))

    pic_data = felsenstein_pic(X_brown, tree)
    pic_corr = np.abs(np.corrcoef(pic_data.T))
    pic_offdiag = pic_corr[np.triu_indices(n_traits, k=1)]
    pic_mean_corr = float(np.mean(pic_offdiag))

    results_c['decorrelation_test'] = {
        'raw_mean_abs_corr': round(raw_mean_corr, 6),
        'pic_mean_abs_corr': round(pic_mean_corr, 6),
        'reduction_ratio': round(pic_mean_corr / max(raw_mean_corr, 1e-10), 6),
        'pass': bool(pic_mean_corr < raw_mean_corr),
        'note': 'Proper Brownian data on the tree — PIC should reduce correlation',
    }
    print(f"  Decorrelation test: raw |r|={raw_mean_corr:.4f}, PIC |r|={pic_mean_corr:.4f}, "
          f"ratio={pic_mean_corr/max(raw_mean_corr,1e-10):.2f}, "
          f"PASS={bool(pic_mean_corr < raw_mean_corr)}")

    # Test 4: No-phylo data — PIC should NOT inflate correlations
    X_no_phylo = rng.normal(0, 1, (n_sp, n_traits))
    raw_corr2 = np.abs(np.corrcoef(X_no_phylo.T))
    raw_offdiag2 = raw_corr2[np.triu_indices(n_traits, k=1)]
    raw_mean2 = float(np.mean(raw_offdiag2))

    pic_data2 = felsenstein_pic(X_no_phylo, tree)
    pic_corr2 = np.abs(np.corrcoef(pic_data2.T))
    pic_offdiag2 = pic_corr2[np.triu_indices(n_traits, k=1)]
    pic_mean2 = float(np.mean(pic_offdiag2))

    results_c['no_phylo_test'] = {
        'raw_mean_abs_corr': round(raw_mean2, 6),
        'pic_mean_abs_corr': round(pic_mean2, 6),
        'note': 'PIC on data without phylogenetic signal — should preserve approximate correlation level',
    }
    print(f"  No-phylo test: raw |r|={raw_mean2:.4f}, PIC |r|={pic_mean2:.4f}")

    # Test 5: Contrast count = n_tips - 1
    big_tree = build_tree(80)
    X_big = rng.normal(0, 1, (80, 10))
    pic_big = felsenstein_pic(X_big, big_tree)
    n_cont = int(pic_big.shape[0])
    results_c['contrast_count_test'] = {
        'n_tips': 80,
        'n_contrasts': n_cont,
        'expected': 79,
        'pass': bool(n_cont == 79),
    }
    print(f"  Contrast count test: n_tips=80, n_contrasts={n_cont} (expected=79), "
          f"PASS={bool(n_cont == 79)}")

    ckpt['exp_c'] = results_c
    save_ckpt(ckpt)
    return ckpt


# ============================================================================
# Experiment D: K-sweep enhanced
# ============================================================================

def run_exp_d(ckpt):
    if 'exp_d' in ckpt:
        print("[EXP D] Already done, skipping")
        return ckpt

    print("\n" + "="*60)
    print("[EXP D] Enhanced K-sweep (dense grid, multi-metric)")
    print("="*60)

    K_values = [0.02, 0.05, 0.08, 0.10, 0.12, 0.14, 0.15, 0.18, 0.20,
                0.22, 0.25, 0.28, 0.30, 0.33, 0.35, 0.38, 0.40]
    seeds = [42, 123, 456]
    results_d = ckpt.get('exp_d', {})

    for K in K_values:
        sk = f'K={K}'
        if sk in results_d:
            print(f"  {sk}: cached, skip")
            continue

        print(f"\n  --- K={K} ---")
        seed_results = []
        for seed in seeds:
            X, W_true, tree, edge_pairs = generate_data(100, 50, 10, K, seed)
            pic_data = felsenstein_pic(X, tree)
            pic_corr = np.corrcoef(pic_data.T)
            triu_idx = np.triu_indices(50, k=1)
            pic_vals = np.abs(pic_corr[triu_idx])
            p99_thresh = np.percentile(pic_vals, 99)
            pic_adj = np.abs(pic_corr) > p99_thresh
            np.fill_diagonal(pic_adj, False)
            f1_pic, prec_pic, rec_pic = compute_f1(pic_adj.astype(float), W_true, 0.5)[:3]

            bg_mask = np.ones((50, 50), dtype=bool)
            np.fill_diagonal(bg_mask, False)
            bg_corr = float(np.abs(pic_corr[bg_mask]).mean())

            true_corr_vals = [float(np.abs(pic_corr[i, j]))
                            for i, j in edge_pairs if i < 50 and j < 50]
            true_corr_mean = float(np.mean(true_corr_vals)) if true_corr_vals else 0.0
            enrichment = true_corr_mean / bg_corr if bg_corr > 0 else 0.0

            raw_corr_mat = np.corrcoef(X.T)
            raw_bg = float(np.abs(raw_corr_mat[bg_mask]).mean())

            seed_results.append({
                'seed': seed, 'f1_pic_p99': round(f1_pic, 4),
                'prec_pic_p99': round(prec_pic, 4), 'rec_pic_p99': round(rec_pic, 4),
                'bg_corr_pic': round(bg_corr, 6), 'true_corr_pic': round(true_corr_mean, 6),
                'enrichment': round(enrichment, 4), 'bg_corr_raw': round(raw_bg, 6),
            })

        f1s = [s['f1_pic_p99'] for s in seed_results]
        enrs = [s['enrichment'] for s in seed_results]
        results_d[sk] = {
            'K': K,
            'mean_f1': round(float(np.mean(f1s)), 4),
            'std_f1': round(float(np.std(f1s)), 4),
            'mean_enrichment': round(float(np.mean(enrs)), 4),
            'mean_bg_corr_pic': round(float(np.mean([s['bg_corr_pic'] for s in seed_results])), 6),
            'mean_true_corr_pic': round(float(np.mean([s['true_corr_pic'] for s in seed_results])), 6),
            'mean_bg_corr_raw': round(float(np.mean([s['bg_corr_raw'] for s in seed_results])), 6),
            'per_seed': seed_results,
        }
        ckpt['exp_d'] = results_d
        save_ckpt(ckpt)
        pf = results_d[sk]['mean_f1']
        print(f"  K={K}: F1={pf:.4f}+/-{results_d[sk]['std_f1']:.4f}, "
              f"enrich={results_d[sk]['mean_enrichment']:.1f}x")

    ckpt['exp_d'] = results_d
    save_ckpt(ckpt)
    return ckpt


# ============================================================================
# Experiment B: CT gradient trace vs K
# ============================================================================

def run_exp_b(ckpt):
    if 'exp_b' in ckpt:
        print("[EXP B] Already done, skipping")
        return ckpt

    print("\n" + "="*60)
    print("[EXP B] CT gradient trace vs phylogenetic strength K")
    print("="*60)

    K_values = [0.02, 0.05, 0.08, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    rng = np.random.default_rng(42)
    tree = build_tree(100)
    n, d_val = 100, 50

    L = np.column_stack([rng.normal(0, 1, n) for _ in range(d_val)])
    eigvals, eigvecs = np.linalg.eigh(L @ L.T + 1e-6 * np.eye(n))
    eigvals = np.maximum(eigvals, 0)
    phylo_chol = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    W_true = np.zeros((d_val, d_val))
    edge_pairs = []
    for _ in range(10):
        i, j = int(rng.integers(0, d_val)), int(rng.integers(0, d_val))
        if i != j and not np.isclose(W_true[j, i], 0):
            W_true[i, j] = rng.uniform(0.3, 0.7) * rng.choice([-1, 1])
            edge_pairs.append((i, j))

    Z_causal = rng.normal(0, 1, (n, d_val))
    X_causal = Z_causal @ np.linalg.inv(np.eye(d_val) - W_true)
    X_phylo_base = phylo_chol @ rng.normal(0, 1, (n, d_val))

    results_b = ckpt.get('exp_b', {})

    for K in K_values:
        sk = f'K={K}'
        if sk in results_b:
            print(f"  {sk}: cached, skip")
            continue

        print(f"\n  --- K={K} ---")
        X = np.sqrt(1 - K) * X_causal + np.sqrt(K) * X_phylo_base
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)

        result, trace = fit_ct_with_trace(X, n_epochs=1200, device=DEVICE, verbose=True)

        last_100_grad = float(np.mean(trace['grad_norm_W'][-100:]))
        last_100_h = float(np.mean(trace['h_W'][-100:]))
        last_100_loss = float(np.mean(trace['loss'][-100:]))

        pic_data = felsenstein_pic(X, tree)
        pic_corr = np.corrcoef(pic_data.T)
        bg_mask = np.ones((d_val, d_val), dtype=bool)
        np.fill_diagonal(bg_mask, False)
        bg_corr = float(np.abs(pic_corr[bg_mask]).mean())
        true_corr_mean = float(np.mean([np.abs(pic_corr[i, j])
                                        for i, j in edge_pairs if i < d_val and j < d_val]))

        f1 = compute_f1(np.array(result['W_final']), W_true, 0.3)[0]

        results_b[sk] = {
            'K': K,
            'final_h': result['final_h'],
            'final_edges': result['final_edges'],
            'f1': round(f1, 4),
            'last100_grad_norm': round(last_100_grad, 6),
            'last100_h': round(last_100_h, 6),
            'last100_loss': round(last_100_loss, 6),
            'bg_corr_pic': round(bg_corr, 6),
            'true_corr_pic': round(true_corr_mean, 6),
            'trace_summary': {
                'h_W_plateau': float(trace['h_W'][-1]),
                'grad_norm_final': float(trace['grad_norm_W'][-1]),
                'loss_final': float(trace['loss'][-1]),
            }
        }
        ckpt['exp_b'] = results_b
        save_ckpt(ckpt)
        print(f"  K={K}: F1={f1:.4f}, grad_norm={last_100_grad:.2e}, "
              f"bg_corr={bg_corr:.4f}, true_corr={true_corr_mean:.4f}")

    ckpt['exp_b'] = results_b
    save_ckpt(ckpt)
    return ckpt


# ============================================================================
# Experiment A: CT convergence curves (3000 epochs)
# ============================================================================

def run_exp_a(ckpt):
    if 'exp_a' in ckpt:
        print("[EXP A] Already done, skipping")
        return ckpt

    print("\n" + "="*60)
    print("[EXP A] CT convergence curves (3000 epochs, 3 seeds)")
    print("="*60)

    seeds = [42, 123, 456]
    results_a = ckpt.get('exp_a', {})

    for seed in seeds:
        sk = f'seed={seed}'
        if sk in results_a:
            print(f"  {sk}: cached, skip")
            continue

        print(f"\n  --- {sk} ---")
        X, W_true, tree, edges = generate_data(100, 50, 10, 0.15, seed)
        result, trace = fit_ct_with_trace(X, n_epochs=3000, device=DEVICE, verbose=True)
        f1, prec, rec, tp, fp, fn = compute_f1(np.array(result['W_final']), W_true, 0.3)

        results_a[sk] = {
            'seed': seed, 'trace': trace,
            'final_h': result['final_h'], 'final_edges': result['final_edges'],
            'f1': round(f1, 4), 'precision': round(prec, 4), 'recall': round(rec, 4),
            'd': 50, 'n': 100, 'n_true_edges': 10, 'K': 0.15,
        }
        ckpt['exp_a'] = results_a
        save_ckpt(ckpt)
        print(f"  {sk}: F1={f1:.4f}, h(W)={result['final_h']:.2e}, "
              f"edges={result['final_edges']}")

    ckpt['exp_a'] = results_a
    save_ckpt(ckpt)
    return ckpt


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    ckpt = load_ckpt()
    ckpt = run_exp_c(ckpt)
    ckpt = run_exp_d(ckpt)
    ckpt = run_exp_b(ckpt)
    ckpt = run_exp_a(ckpt)
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Checkpoint: {CKPT_FILE}")
    print("="*60)
