"""
MULTI-METHOD ADDON: DAGMA + GOLEM + NOTEARS + CT + PIC+corr
Runs alongside bonus_fill, independent checkpoint.
Output: results/multimethod_addon.json
"""

import sys, os, json, time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
sys.path.insert(0, os.path.dirname(__file__))  # portable
from felsenstein_pic import PhyloNode, felsenstein_pic
# causalscale should be installed via: pip install causalscale

import torch
from causalscale.core.transformer import (
    CausalTransformer, CausalTransformerConfig, NOTEARSConstraint
)

CKPT = r'C:\Users\高帅东\Desktop\evo_causal\results\multimethod_addon.json'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

# ============================================================================
# JSON-safe
# ============================================================================
def _json_safe(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_json_safe(v) for v in obj]
    return obj

def load_ckpt(): return json.load(open(CKPT)) if os.path.exists(CKPT) else {}
def save_ckpt(c):
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    tmp = CKPT + '.tmp'
    with open(tmp, 'w') as f: json.dump(_json_safe(c), f, indent=2)
    os.replace(tmp, CKPT)

# ============================================================================
# Shared
# ============================================================================
def build_tree(n_species):
    tips = []
    for i in range(n_species):
        t = PhyloNode(f"Sp{i}", children=[], branch_length=max(0.01, np.random.exponential(1.0)))
        t.index = i; t.is_tip = True; tips.append(t)
    nodes = tips; next_id = 0
    while len(nodes) > 1:
        new_nodes = []
        for i in range(0, len(nodes), 2):
            if i + 1 < len(nodes):
                bl = max(0.01, np.random.exponential(0.5))
                parent = PhyloNode(f"N{next_id}", children=[nodes[i], nodes[i+1]], branch_length=bl)
                parent.is_tip = False; next_id += 1; new_nodes.append(parent)
            else: new_nodes.append(nodes[i])
        nodes = new_nodes
    return nodes[0]

def generate_data(n=200, d=50, n_true_edges=10, K=0.15, seed=42):
    rng = np.random.default_rng(seed)
    tree = build_tree(n)
    L = np.column_stack([rng.normal(0, 1, n) for _ in range(d)])
    eigvals, eigvecs = np.linalg.eigh(L @ L.T + 1e-6 * np.eye(n))
    eigvals = np.maximum(eigvals, 0)
    phylo_chol = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    W_true = np.zeros((d, d)); edge_pairs = []
    for _ in range(n_true_edges):
        i, j = int(rng.integers(0, d)), int(rng.integers(0, d))
        if i != j:
            W_true[i, j] = rng.uniform(0.3, 0.7) * rng.choice([-1, 1])
            edge_pairs.append((i, j))
    Z = rng.normal(0, 1, (n, d))
    X_causal = Z @ np.linalg.inv(np.eye(d) - W_true)
    X_phylo = phylo_chol @ rng.normal(0, 1, (n, d))
    X = np.sqrt(1 - K) * X_causal + np.sqrt(K) * X_phylo
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return X, W_true, tree, edge_pairs

def compute_f1(W_est, W_true, threshold=0.3):
    Wb = np.abs(W_est) > threshold; Wt = np.abs(W_true) > 0
    tp = int((Wb & Wt).sum()); fp = int((Wb & ~Wt).sum()); fn = int((~Wb & Wt).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return (2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0), prec, rec, tp, fp, fn

# ============================================================================
# DAGMA runner
# ============================================================================
def run_dagma_wrapper(X, lambda1=0.03, w_threshold=0.3, T=5, lr=0.0003):
    """Run DAGMA linear on data."""
    try:
        from dagma.linear import DagmaLinear
        model = DagmaLinear(loss_type='l2')
        W = model.fit(X.astype(np.float64), lambda1=lambda1, w_threshold=w_threshold,
                      T=T, lr=lr)
        return W
    except Exception as e:
        print(f"    DAGMA error: {e}")
        return np.zeros((X.shape[1], X.shape[1]))

# ============================================================================
# GOLEM runner
# ============================================================================
def run_golem_wrapper(X, lambda_1=0.02, lambda_2=5.0, equal_variances=True,
                      num_iter=100000, learning_rate=1e-3, device='cpu'):
    """Run GOLEM (EV or NV) on data."""
    try:
        import torch
        from golem.golem import GolemModel

        X_t = torch.tensor(X.astype(np.float32), device=device)
        d = X_t.shape[1]

        if equal_variances:
            from golem.golem import GolemEV, GolemEVCriterion
            model = GolemEV(d, lambda_1=lambda_1, lambda_2=lambda_2,
                          learning_rate=learning_rate, device=device)
        else:
            from golem.golem import GolemNV, GolemNVCriterion
            model = GolemNV(d, lambda_1=lambda_1, lambda_2=lambda_2,
                          learning_rate=learning_rate, device=device)

        W_est = model.fit(X_t, num_iter=num_iter)
        return W_est.cpu().numpy() if hasattr(W_est, 'cpu') else W_est
    except ImportError:
        print("    GOLEM not installed, skipping")
        return None
    except Exception as e:
        print(f"    GOLEM error: {e}")
        return None

# ============================================================================
# Main sweep
# ============================================================================

if __name__ == '__main__':
    print(f"MULTI-METHOD ADDON — {time.strftime('%H:%M:%S')}")

    configs = [
        (50, 100, 0.05), (50, 100, 0.15), (50, 100, 0.30),
        (50, 200, 0.05), (50, 200, 0.15), (50, 200, 0.30),
        (100, 100, 0.15), (100, 200, 0.05), (100, 200, 0.15),
        (100, 200, 0.30), (100, 500, 0.15),
        (200, 200, 0.15), (200, 500, 0.15),
    ]
    # Reduced configs for DAGMA/GOLEM (they're much slower than CT)
    fast_configs = [
        (50, 200, 0.15), (100, 200, 0.15), (50, 100, 0.15),
        (100, 200, 0.05), (100, 200, 0.30),
    ]

    seeds = [42, 123, 456]
    ckpt = load_ckpt()
    results = ckpt.get('results', {})

    # Check what methods are available
    has_dagma = False; has_golem = False
    try:
        from dagma.linear import DagmaLinear; has_dagma = True
    except: pass
    try:
        from golem.golem import GolemEV; has_golem = True
    except: pass

    print(f"Available: DAGMA={has_dagma}, GOLEM={has_golem}")

    # Fast methods (PIC+corr, CT) for ALL configs
    for d_val, n_val, K_val in configs:
        key = f"d={d_val}_n={n_val}_K={K_val}"
        if key in results:
            continue

        t0 = time.time()
        print(f"\n  --- {key} ---")
        seed_results = []

        for seed in seeds:
            X, W_true, tree, edges = generate_data(n_val, d_val, 10, K_val, seed)

            # PIC+corr
            pic_data = felsenstein_pic(X, tree)
            pic_corr = np.corrcoef(pic_data.T)
            triu_idx = np.triu_indices(d_val, k=1)
            pic_vals = np.abs(pic_corr[triu_idx])
            p99_thresh = np.percentile(pic_vals, 99)
            pic_adj = np.abs(pic_corr) > p99_thresh
            np.fill_diagonal(pic_adj, False)
            pic_f1 = compute_f1(pic_adj.astype(float), W_true, 0.5)[0]

            bg_mask = np.ones((d_val, d_val), dtype=bool)
            np.fill_diagonal(bg_mask, False)
            pic_bg = float(np.abs(pic_corr[bg_mask]).mean())
            pic_true = float(np.mean([np.abs(pic_corr[i, j])
                            for i, j in edges if i < d_val and j < d_val]))
            pic_enrich = pic_true / max(pic_bg, 1e-10)

            # NOTEARS
            from causalscale.core.dag_constraint import notears_linear
            try:
                W_note = notears_linear(X.astype(np.float64), lambda1=0.1,
                                       loss_type='l2', max_iter=100, w_threshold=0.3)
                note_f1 = compute_f1(W_note, W_true, 0.3)[0]
            except:
                note_f1 = -1.0

            # CT (simplified, faster)
            X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
            config_ct = CausalTransformerConfig(
                d_vars=d_val, d_model=64, n_heads=4, n_layers=2,
                lambda_dag=0.5, lambda_sparsity=0.01, lr=0.001, edge_threshold=0.3)
            model = CausalTransformer(config_ct).to(DEVICE); model.train()
            opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
            bs = min(128, n_val)
            for epoch in range(2000):
                perm = torch.randperm(n_val)
                for i in range(0, n_val, bs):
                    idx = perm[i:i+bs]; xb = X_t[idx]
                    Wb, _ = model(xb); losses = model.compute_loss(xb, Wb)
                    opt.zero_grad(); losses['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            model.eval()
            with torch.no_grad():
                Wf, _ = model(X_t[:min(500, n_val)])
                Wmn = Wf.mean(dim=0).cpu().numpy()
            ct_f1 = compute_f1(Wmn, W_true, 0.3)[0]
            del model, opt, X_t
            if DEVICE == 'cuda': torch.cuda.empty_cache()

            seed_results.append({
                'seed': seed,
                'pic_f1': round(pic_f1, 4), 'pic_enrich': round(pic_enrich, 2),
                'note_f1': round(note_f1, 4) if note_f1 >= 0 else None,
                'ct_f1': round(ct_f1, 4),
            })

        f1p = [s['pic_f1'] for s in seed_results]
        f1n = [s['note_f1'] for s in seed_results if s['note_f1'] is not None]
        f1c = [s['ct_f1'] for s in seed_results]

        all_methods = [
            ('PIC+corr', round(float(np.mean(f1p)), 4)),
            ('NOTEARS', round(float(np.mean(f1n)), 4) if f1n else 0.0),
            ('CT', round(float(np.mean(f1c)), 4)),
        ]
        best = max(all_methods, key=lambda x: x[1])

        results[key] = {
            'd': d_val, 'n': n_val, 'K': K_val,
            'pic_f1': all_methods[0][1], 'note_f1': all_methods[1][1],
            'ct_f1': all_methods[2][1], 'best_method': best[0],
            'best_f1': best[1], 'time_s': round(time.time() - t0, 1), 'seeds': seed_results,
        }
        ckpt['results'] = results; save_ckpt(ckpt)
        print(f"  {key}: BEST={best[0]} PIC={all_methods[0][1]:.3f} "
              f"NOTEARS={all_methods[1][1]:.3f} CT={all_methods[2][1]:.3f} [{time.time()-t0:.0f}s]")

    # DAGMA + GOLEM for fast_configs only (slower methods)
    if has_dagma or has_golem:
        print("\n" + "="*60)
        print("[DAGMA+GOLEM] Running slower methods on key configs")
        print("="*60)

        dagma_results = ckpt.get('dagma_results', {})
        golem_results = ckpt.get('golem_results', {})

        for d_val, n_val, K_val in fast_configs:
            key = f"d={d_val}_n={n_val}_K={K_val}"

            # DAGMA
            if has_dagma and key not in dagma_results:
                t0 = time.time()
                print(f"\n  DAGMA {key}...")
                seed_rs = []
                for seed in seeds[:2]:  # 2 seeds (DAGMA is slow)
                    X, W_true, tree, edges = generate_data(n_val, d_val, 10, K_val, seed)
                    W_dag = run_dagma_wrapper(X)
                    f1 = compute_f1(W_dag, W_true, 0.3)[0]
                    seed_rs.append({'seed': seed, 'f1': round(f1, 4)})

                dagma_results[key] = {
                    'd': d_val, 'n': n_val, 'K': K_val,
                    'f1_mean': round(float(np.mean([s['f1'] for s in seed_rs])), 4),
                    'f1_best': round(float(np.max([s['f1'] for s in seed_rs])), 4),
                    'time_s': round(time.time() - t0, 1), 'seeds': seed_rs,
                }
                ckpt['dagma_results'] = dagma_results; save_ckpt(ckpt)
                print(f"    DAGMA F1={dagma_results[key]['f1_mean']:.3f} [{time.time()-t0:.0f}s]")

            # GOLEM
            if has_golem and key not in golem_results:
                t0 = time.time()
                print(f"\n  GOLEM {key}...")
                seed_rs = []
                for seed in seeds[:2]:
                    X, W_true, tree, edges = generate_data(n_val, d_val, 10, K_val, seed)
                    W_go = run_golem_wrapper(X, device=DEVICE)
                    if W_go is not None:
                        f1 = compute_f1(W_go, W_true, 0.3)[0]
                    else:
                        f1 = -1.0
                    seed_rs.append({'seed': seed, 'f1': round(f1, 4) if f1 >= 0 else None})

                valid = [s['f1'] for s in seed_rs if s['f1'] is not None]
                golem_results[key] = {
                    'd': d_val, 'n': n_val, 'K': K_val,
                    'f1_mean': round(float(np.mean(valid)), 4) if valid else None,
                    'f1_best': round(float(np.max(valid)), 4) if valid else None,
                    'time_s': round(time.time() - t0, 1), 'seeds': seed_rs,
                }
                ckpt['golem_results'] = golem_results; save_ckpt(ckpt)
                gf = golem_results[key].get('f1_mean')
                print(f"    GOLEM F1={gf} [{time.time()-t0:.0f}s]")

        ckpt['dagma_results'] = dagma_results
        ckpt['golem_results'] = golem_results
        save_ckpt(ckpt)

    print(f"\n{'='*60}")
    print(f"ADDON DONE at {time.strftime('%H:%M:%S')}")
    print(f"{'='*60}")
