"""
BONUS FILL: Runs after overnight_massive.py finishes.
Fills remaining GPU hours with:
  1. CT multi-seed (3 seeds) for ALL phase diagram configs
  2. NOTEARS baseline for ALL d,K (including d>100)
  3. Finer K-grid around critical transition zone
  4. Larger model capacity test (d_model=128, n_layers=4)
  5. Full CT vs PIC+corr vs NOTEARS head-to-head at selected configs

Checkpoint: results/bonus_fill.json (independent, reads overnight for completion status)
"""

import sys, os, json, time, math

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
sys.path.insert(0, os.path.dirname(__file__))  # portable
from felsenstein_pic import PhyloNode, felsenstein_pic
# causalscale should be installed via: pip install causalscale

import torch
from causalscale.core.transformer import (
    CausalTransformer, CausalTransformerConfig, NOTEARSConstraint
)
from causalscale.core.dag_constraint import notears_linear

CKPT = r'C:\Users\高帅东\Desktop\evo_causal\results\bonus_fill.json'
OVERNIGHT_CKPT = r'C:\Users\高帅东\Desktop\evo_causal\results\overnight_massive.json'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

# ============================================================================
# JSON-safe checkpoint
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
# Shared (copied from overnight — keep in sync)
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
    Wb = np.abs(W_est) > threshold; Wt = np.abs(W_true) > 0
    tp = int((Wb & Wt).sum()); fp = int((Wb & ~Wt).sum()); fn = int((~Wb & Wt).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return (2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0), prec, rec, tp, fp, fn

def run_ct(X, d, n_epochs=2000, d_model=64, n_heads=4, n_layers=2,
           edge_threshold=0.3, lr=0.001):
    n, _d = X.shape
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    config = CausalTransformerConfig(
        d_vars=d, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        lambda_dag=0.5, lambda_sparsity=0.01, lr=lr, edge_threshold=edge_threshold)
    model = CausalTransformer(config).to(DEVICE); model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    bs = min(128, n)
    edge_h, dag_h = [], []
    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]; xb = X_t[idx]
            Wb, _ = model(xb); losses = model.compute_loss(xb, Wb)
            opt.zero_grad(); losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        Wm = Wb.mean(dim=0).detach()
        edge_h.append(int((torch.abs(Wm) > edge_threshold).float().sum().item()))
        dag_h.append(float(losses['dag']))
    model.eval()
    with torch.no_grad():
        Wf, _ = model(X_t[:min(500, n)])
        Wmn = Wf.mean(dim=0).cpu().numpy()
        fh = float(NOTEARSConstraint()(torch.tensor(Wmn, device=DEVICE)).item())
    del model, opt, X_t
    if DEVICE == 'cuda': torch.cuda.empty_cache()
    return {
        'W': Wmn.tolist(), 'final_h': round(fh, 6),
        'final_edges': int((np.abs(Wmn) > edge_threshold).sum()),
        'edge_plateau': float(np.mean(edge_h[-200:])),
        'h_plateau': float(np.mean(dag_h[-200:])),
    }

def run_pic_corr(X, tree, W_true, edge_pairs, d, pct=99):
    pic_data = felsenstein_pic(X, tree)
    pic_corr = np.corrcoef(pic_data.T)
    triu_idx = np.triu_indices(d, k=1)
    pic_vals = np.abs(pic_corr[triu_idx])
    threshold = np.percentile(pic_vals, pct)
    adj = np.abs(pic_corr) > threshold; np.fill_diagonal(adj, False)
    f1 = compute_f1(adj.astype(float), W_true, 0.5)[0]
    bg_mask = np.ones((d, d), dtype=bool); np.fill_diagonal(bg_mask, False)
    bg_corr = float(np.abs(pic_corr[bg_mask]).mean())
    true_corrs = [float(np.abs(pic_corr[i, j])) for i, j in edge_pairs if i < d and j < d]
    true_corr = float(np.mean(true_corrs)) if true_corrs else 0.0
    return {'f1': round(f1, 4), 'bg_corr': round(bg_corr, 6),
            'true_corr': round(true_corr, 6),
            'enrichment': round(true_corr / max(bg_corr, 1e-10), 4)}


# ============================================================================
# MODULE 1: CT multi-seed for ALL phase diagram configs (3 seeds each)
# ============================================================================

def mod_ct_multiseed(ckpt):
    if 'ct_multiseed' in ckpt:
        print("[CT MULTISEED] Done"); return ckpt
    if not os.path.exists(OVERNIGHT_CKPT):
        print("[CT MULTISEED] Overnight not ready yet"); return ckpt

    print("\n" + "="*60)
    print("[CT MULTISEED] CT with 3 seeds for all d x K configs")
    print("="*60)

    overnight = json.load(open(OVERNIGHT_CKPT))
    pd = overnight.get('phase_diagram', {})
    if not pd:
        print("[CT MULTISEED] Phase diagram not in overnight yet"); return ckpt

    d_vals = sorted(set(int(k.split('_')[0].split('=')[1]) for k in pd.keys()))
    K_vals = sorted(set(float(k.split('_')[1].split('=')[1]) for k in pd.keys()))
    print(f"Configs: {len(d_vals)} d x {len(K_vals)} K = {len(d_vals)*len(K_vals)} total")

    seeds_ct = [42, 123, 456, 789, 999]  # 5 seeds for CT
    n = 200
    results = ckpt.get('ct_multiseed', {})

    for d_val in d_vals:
        for K_val in K_vals:
            key = f"d={d_val}_K={K_val}"
            if key in results:
                continue

            t0 = time.time()
            print(f"\n  --- {key} (CT x 5 seeds) ---")
            seed_results = []
            for seed in seeds_ct:
                X, W_true, tree, edges = generate_data(n, d_val, 10, K_val, seed)
                ct_r = run_ct(X, d_val, n_epochs=2000)
                f1 = compute_f1(np.array(ct_r['W']), W_true, 0.3)[0]
                seed_results.append({'seed': seed, 'f1': round(f1, 4),
                                     'edges': ct_r['final_edges'], 'h': ct_r['final_h']})

            f1s = [s['f1'] for s in seed_results]
            results[key] = {
                'd': d_val, 'K': K_val, 'n': n,
                'ct_f1_mean': round(float(np.mean(f1s)), 4),
                'ct_f1_std': round(float(np.std(f1s)), 4),
                'ct_f1_best': round(float(np.max(f1s)), 4),
                'ct_edges_mean': round(float(np.mean([s['edges'] for s in seed_results])), 1),
                'time_s': round(time.time() - t0, 1),
                'seeds': seed_results,
            }
            ckpt['ct_multiseed'] = results; save_ckpt(ckpt)
            print(f"  {key}: CT_F1={results[key]['ct_f1_mean']:.3f}+/-{results[key]['ct_f1_std']:.3f} "
                  f"[{time.time()-t0:.0f}s]")

    ckpt['ct_multiseed'] = results; save_ckpt(ckpt)
    return ckpt


# ============================================================================
# MODULE 2: NOTEARS baseline for ALL d,K (including d>100)
# ============================================================================

def mod_notears_all(ckpt):
    if 'notears_all' in ckpt:
        print("[NOTEARS ALL] Done"); return ckpt
    if not os.path.exists(OVERNIGHT_CKPT):
        print("[NOTEARS ALL] Overnight not ready"); return ckpt

    print("\n" + "="*60)
    print("[NOTEARS ALL] NOTEARS baseline for ALL d x K configs")
    print("="*60)

    overnight = json.load(open(OVERNIGHT_CKPT))
    pd = overnight.get('phase_diagram', {})
    if not pd: return ckpt

    d_vals = sorted(set(int(k.split('_')[0].split('=')[1]) for k in pd.keys()))
    K_vals = sorted(set(float(k.split('_')[1].split('=')[1]) for k in pd.keys()))
    seeds = [42, 123, 456]
    n = 200
    results = ckpt.get('notears_all', {})

    for d_val in d_vals:
        for K_val in K_vals:
            key = f"d={d_val}_K={K_val}"
            if key in results: continue

            t0 = time.time()
            print(f"\n  --- {key} (NOTEARS x 3 seeds) ---")
            seed_results = []
            for seed in seeds:
                X, W_true, tree, edges = generate_data(n, d_val, 10, K_val, seed)
                try:
                    W_note = notears_linear(X.astype(np.float64), lambda1=0.1,
                                           loss_type='l2', max_iter=100, w_threshold=0.3)
                    f1 = compute_f1(W_note, W_true, 0.3)[0]
                except Exception:
                    W_note = np.zeros((d_val, d_val)); f1 = -1.0
                seed_results.append({'seed': seed, 'f1': round(f1, 4) if f1 >= 0 else None})

            valid_f1s = [s['f1'] for s in seed_results if s['f1'] is not None]
            results[key] = {
                'd': d_val, 'K': K_val, 'n': n,
                'note_f1_mean': round(float(np.mean(valid_f1s)), 4) if valid_f1s else None,
                'note_f1_std': round(float(np.std(valid_f1s)), 4) if valid_f1s else None,
                'note_f1_best': round(float(np.max(valid_f1s)), 4) if valid_f1s else None,
                'note_valid_count': len(valid_f1s),
                'time_s': round(time.time() - t0, 1),
                'seeds': seed_results,
            }
            ckpt['notears_all'] = results; save_ckpt(ckpt)
            nf = results[key].get('note_f1_mean')
            print(f"  {key}: NOTEARS_F1={nf} [{time.time()-t0:.0f}s]")

    ckpt['notears_all'] = results; save_ckpt(ckpt)
    return ckpt


# ============================================================================
# MODULE 3: Finer K-grid in critical zone (K=0.08 to 0.30, step 0.02)
# ============================================================================

def mod_fine_K(ckpt):
    if 'fine_K' in ckpt:
        print("[FINE K] Done"); return ckpt
    if not os.path.exists(OVERNIGHT_CKPT):
        print("[FINE K] Overnight not ready"); return ckpt

    print("\n" + "="*60)
    print("[FINE K] Dense K-grid in critical transition zone")
    print("="*60)

    # Finer grid: K from 0.06 to 0.34, step 0.02 (fills gaps between overnight's grid)
    K_fine = [0.04, 0.06, 0.09, 0.11, 0.13, 0.16, 0.17, 0.19,
              0.21, 0.23, 0.24, 0.26, 0.27, 0.29, 0.31, 0.32, 0.34, 0.36, 0.39]
    d_vals = [50, 100, 200]  # focus on key dimensions
    seeds = [42, 123, 456]
    n = 200
    results = ckpt.get('fine_K', {})

    for d_val in d_vals:
        for K_val in K_fine:
            key = f"d={d_val}_K={K_val}"
            if key in results: continue

            t0 = time.time()
            print(f"\n  --- {key} ---")
            seed_results = []
            for seed in seeds:
                X, W_true, tree, edges = generate_data(n, d_val, 10, K_val, seed)
                pic_r = run_pic_corr(X, tree, W_true, edges, d_val)
                ct_r = run_ct(X, d_val, n_epochs=2000)
                ct_f1 = compute_f1(np.array(ct_r['W']), W_true, 0.3)[0]
                seed_results.append({'seed': seed, 'pic_f1': pic_r['f1'],
                                     'pic_enrich': pic_r['enrichment'],
                                     'ct_f1': round(ct_f1, 4), 'ct_edges': ct_r['final_edges']})

            f1s_pic = [s['pic_f1'] for s in seed_results]
            f1s_ct = [s['ct_f1'] for s in seed_results]
            results[key] = {
                'd': d_val, 'K': K_val, 'n': n,
                'pic_f1_mean': round(float(np.mean(f1s_pic)), 4),
                'pic_f1_std': round(float(np.std(f1s_pic)), 4),
                'ct_f1_mean': round(float(np.mean(f1s_ct)), 4),
                'ct_f1_std': round(float(np.std(f1s_ct)), 4),
                'ct_f1_best': round(float(np.max(f1s_ct)), 4),
                'time_s': round(time.time() - t0, 1),
                'seeds': seed_results,
            }
            ckpt['fine_K'] = results; save_ckpt(ckpt)
            print(f"  {key}: PIC_F1={results[key]['pic_f1_mean']:.3f} "
                  f"CT_F1={results[key]['ct_f1_mean']:.3f} [{time.time()-t0:.0f}s]")

    ckpt['fine_K'] = results; save_ckpt(ckpt)
    return ckpt


# ============================================================================
# MODULE 4: Large model capacity test (d_model=128,256; n_layers=4)
# ============================================================================

def mod_large_model(ckpt):
    if 'large_model' in ckpt:
        print("[LARGE MODEL] Done"); return ckpt

    print("\n" + "="*60)
    print("[LARGE MODEL] Testing larger CT architectures")
    print("="*60)

    architectures = [
        (128, 4), (256, 4), (64, 6), (128, 6),
    ]
    d_val, n, K = 100, 200, 0.15
    seeds = [42, 123, 456]
    results = ckpt.get('large_model', {})

    for dm, nl in architectures:
        key = f"dm={dm}_nl={nl}"
        if key in results: continue

        t0 = time.time()
        print(f"\n  --- {key} ---")
        seed_results = []
        for seed in seeds:
            X, W_true, tree, edges = generate_data(n, d_val, 10, K, seed)
            ct_r = run_ct(X, d_val, n_epochs=2000, d_model=dm, n_layers=nl)
            f1 = compute_f1(np.array(ct_r['W']), W_true, 0.3)[0]
            seed_results.append({'seed': seed, 'f1': round(f1, 4),
                                 'edges': ct_r['final_edges'], 'h': ct_r['final_h']})

        f1s = [s['f1'] for s in seed_results]
        results[key] = {
            'd_model': dm, 'n_layers': nl, 'd': d_val, 'K': K,
            'f1_mean': round(float(np.mean(f1s)), 4),
            'f1_std': round(float(np.std(f1s)), 4),
            'f1_best': round(float(np.max(f1s)), 4),
            'time_s': round(time.time() - t0, 1),
            'seeds': seed_results,
        }
        ckpt['large_model'] = results; save_ckpt(ckpt)
        print(f"  {key}: F1={results[key]['f1_mean']:.3f}+/-{results[key]['f1_std']:.3f} "
              f"[{time.time()-t0:.0f}s]")

    ckpt['large_model'] = results; save_ckpt(ckpt)
    return ckpt


# ============================================================================
# MODULE 5: Head-to-head comparison at key configs (PIC+corr vs CT vs NOTEARS)
# ============================================================================

def mod_head2head(ckpt):
    if 'head2head' in ckpt:
        print("[HEAD2HEAD] Done"); return ckpt

    print("\n" + "="*60)
    print("[HEAD2HEAD] PIC+corr vs CT vs NOTEARS at key (d,n,K) configs")
    print("="*60)

    configs = [
        (50, 100, 0.15), (50, 200, 0.15), (50, 500, 0.15),
        (100, 100, 0.15), (100, 200, 0.15), (100, 500, 0.15),
        (200, 200, 0.15), (200, 500, 0.15),
        (100, 200, 0.05), (100, 200, 0.25), (100, 200, 0.35),
    ]
    seeds = [42, 123, 456, 789, 999]  # 5 seeds for robustness
    results = ckpt.get('head2head', {})

    for d_val, n_val, K_val in configs:
        key = f"d={d_val}_n={n_val}_K={K_val}"
        if key in results: continue

        t0 = time.time()
        print(f"\n  --- {key} ---")
        seed_results = []
        for seed in seeds:
            X, W_true, tree, edges = generate_data(n_val, d_val, 10, K_val, seed)
            pic_r = run_pic_corr(X, tree, W_true, edges, d_val)
            ct_r = run_ct(X, d_val, n_epochs=2000)
            ct_f1 = compute_f1(np.array(ct_r['W']), W_true, 0.3)[0]

            try:
                W_note = notears_linear(X.astype(np.float64), lambda1=0.1,
                                       loss_type='l2', max_iter=100, w_threshold=0.3)
                note_f1 = compute_f1(W_note, W_true, 0.3)[0]
            except Exception:
                note_f1 = -1.0

            seed_results.append({
                'seed': seed,
                'pic_f1': pic_r['f1'], 'pic_enrich': pic_r['enrichment'],
                'ct_f1': round(ct_f1, 4), 'note_f1': round(note_f1, 4) if note_f1 >= 0 else None,
            })

        f1p = [s['pic_f1'] for s in seed_results]
        f1c = [s['ct_f1'] for s in seed_results]
        f1n = [s['note_f1'] for s in seed_results if s['note_f1'] is not None]
        results[key] = {
            'd': d_val, 'n': n_val, 'K': K_val,
            'pic_f1_mean': round(float(np.mean(f1p)), 4),
            'ct_f1_mean': round(float(np.mean(f1c)), 4),
            'note_f1_mean': round(float(np.mean(f1n)), 4) if f1n else None,
            'best_method': max(
                ('PIC', round(float(np.mean(f1p)), 4)),
                ('CT', round(float(np.mean(f1c)), 4)),
                ('NOTEARS', round(float(np.mean(f1n)), 4) if f1n else (-1,)),
                key=lambda x: x[1]
            )[0],
            'time_s': round(time.time() - t0, 1),
            'seeds': seed_results,
        }
        ckpt['head2head'] = results; save_ckpt(ckpt)
        best = results[key]['best_method']
        print(f"  {key}: BEST={best} PIC={results[key]['pic_f1_mean']:.3f} "
              f"CT={results[key]['ct_f1_mean']:.3f} NOTEARS={results[key].get('note_f1_mean','N/A')} "
              f"[{time.time()-t0:.0f}s]")

    ckpt['head2head'] = results; save_ckpt(ckpt)
    return ckpt


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print(f"BONUS FILL — {time.strftime('%H:%M:%S')}")
    print("Waiting for overnight to provide phase_diagram baseline...")

    # Wait up to 5 minutes for overnight to finish enough configs
    waited = 0
    while waited < 300:
        if os.path.exists(OVERNIGHT_CKPT):
            overnight = json.load(open(OVERNIGHT_CKPT))
            pd = overnight.get('phase_diagram', {})
            if len(pd) >= 5:  # enough data to start
                print(f"Overnight has {len(pd)} phase configs — starting bonus")
                break
        time.sleep(15)
        waited += 15

    ckpt = load_ckpt()
    ckpt = mod_ct_multiseed(ckpt)
    ckpt = mod_notears_all(ckpt)
    ckpt = mod_fine_K(ckpt)
    ckpt = mod_large_model(ckpt)
    ckpt = mod_head2head(ckpt)

    print(f"\n{'='*60}")
    print(f"BONUS FILL COMPLETE at {time.strftime('%H:%M:%S')}")
    print(f"{'='*60}")
