"""
EXPERIMENT A+B: Nonlinear SEM Validation + Wall Penetration (n->inf, alpha->0)
Purpose: 
  A) Verify that the wall persists under nonlinear data generation (not just linear SEM)
  B) Find the conditions under which the wall CAN be penetrated (large n, weak alpha)
All with checkpointing + incremental save.
"""
import sys, os, json, time, math
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
torch.set_num_threads(8)

sys.path.insert(0, os.path.dirname(__file__))  # portable
from run_master_all import (generate_data, run_ct, build_tree, felsenstein_pic,
                             compute_f1, _json_safe, NOTEARSConstraint)

CKPT = r'C:\Users\高帅东\Desktop\evo_causal\results\nonlinear_wall.json'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEEDS = [42, 123, 456]
BASE_SEED = 9999

def load_ckpt():
    if os.path.exists(CKPT):
        with open(CKPT, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_ckpt(ckpt):
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    tmp = CKPT + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(_json_safe(ckpt), f, indent=2)
    os.replace(tmp, CKPT)

# ============================================================
# Nonlinear data generators
# ============================================================

def generate_data_nonlinear_mlp(n=200, d=50, n_true=10, alpha=0.15, seed=42):
    """MLP-based causal mechanism: X = f(Z) where f is a 2-layer MLP with tanh."""
    rng = np.random.default_rng(seed)
    tree = build_tree(n, seed=seed + 1000)
    
    # Phylogeny
    L = rng.normal(0, 1, (n, d))
    C_chol = np.linalg.cholesky(L @ L.T + 1e-6 * np.eye(n))
    X_phylo = C_chol @ rng.normal(0, 1, (n, d))
    
    # Causal DAG (same style)
    W_true = np.zeros((d, d))
    placed = 0
    while placed < n_true:
        i, j = rng.integers(0, d, 2)
        if i < j and W_true[i, j] == 0:
            W_true[i, j] = rng.uniform(0.3, 0.7) * rng.choice([-1, 1])
            placed += 1
    
    # Generate independent noise
    Z = rng.normal(0, 1, (n, d))
    
    # Nonlinear causal mechanism: each variable = MLP(parents)
    W1 = rng.normal(0, 1/np.sqrt(d), (d, 8))  # hidden layer
    W2 = rng.normal(0, 1/np.sqrt(8), (8, d))
    b1 = rng.normal(0, 0.1, 8)
    b2 = rng.normal(0, 0.1, d)
    
    X_causal = np.zeros((n, d))
    for j in range(d):
        parents = np.where(W_true[:, j] != 0)[0]
        if len(parents) > 0:
            parent_contrib = X_causal[:, parents] @ W_true[parents, j].reshape(-1, 1)
        else:
            parent_contrib = np.zeros((n, 1))
        # MLP nonlinearity
        h = np.tanh(parent_contrib[:, 0].reshape(-1, 1) @ np.ones((1, 8)) + b1)
        nonlinear_effect = np.tanh(h @ W2[:, j:j+1] + b2[j])[:, 0]
        X_causal[:, j] = nonlinear_effect + Z[:, j]
    
    # Mix
    X = np.sqrt(1 - alpha) * X_causal + np.sqrt(alpha) * X_phylo
    
    # Standardize
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    
    # Compute PIC tree edges
    n_tips = n
    tree_depths = {}
    def assign_depth(node, d=0):
        node.depth = d
        if hasattr(node, 'children') and node.children:
            for ch in node.children:
                assign_depth(ch, d + 1)
    assign_depth(tree)
    
    # PIC data
    pic_data = felsenstein_pic(X, tree)
    n_contrasts = pic_data.shape[0]
    pic_corr = np.corrcoef(pic_data.T)
    
    # Edge percentiles
    edge_ep = {}
    for K_true in range(n_true):
        idx = np.where(W_true != 0)
        if K_true < len(idx[0]):
            i, j = idx[0][K_true], idx[1][K_true]
            edge_ep[f'true_edge_{K_true}'] = float(pic_corr[i, j])
    
    # p99 threshold
    triu_idx = np.triu_indices(d, k=1)
    triu_vals = np.abs(pic_corr[triu_idx])
    p99 = float(np.percentile(triu_vals, 99))
    
    return X, W_true, tree, edge_ep, p99, pic_data

def generate_data_nonlinear_sigmoid(n=200, d=50, n_true=10, alpha=0.15, seed=42):
    """Sigmoid-based: each variable = sigmoid(linear combination of parents + noise)."""
    rng = np.random.default_rng(seed)
    tree = build_tree(n, seed=seed + 1000)
    
    L = rng.normal(0, 1, (n, d))
    C_chol = np.linalg.cholesky(L @ L.T + 1e-6 * np.eye(n))
    X_phylo = C_chol @ rng.normal(0, 1, (n, d))
    
    W_true = np.zeros((d, d))
    placed = 0
    while placed < n_true:
        i, j = rng.integers(0, d, 2)
        if i < j and W_true[i, j] == 0:
            W_true[i, j] = rng.uniform(0.5, 1.5) * rng.choice([-1, 1])
            placed += 1
    
    Z = rng.normal(0, 0.5, (n, d))
    X_causal = np.zeros((n, d))
    order = list(range(d))
    
    for j in order:
        parents = np.where(W_true[:, j] != 0)[0]
        if len(parents) > 0:
            parent_term = X_causal[:, parents] @ W_true[parents, j]
        else:
            parent_term = np.zeros(n)
        X_causal[:, j] = 2.0 / (1.0 + np.exp(-parent_term)) - 1.0 + Z[:, j]
    
    X = np.sqrt(1 - alpha) * X_causal + np.sqrt(alpha) * X_phylo
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    
    pic_data = felsenstein_pic(X, tree)
    pic_corr = np.corrcoef(pic_data.T)
    triu_idx = np.triu_indices(d, k=1)
    triu_vals = np.abs(pic_corr[triu_idx])
    p99 = float(np.percentile(triu_vals, 99))
    
    edge_ep = {}
    for K_true in range(n_true):
        idx = np.where(W_true != 0)
        if K_true < len(idx[0]):
            i, j = idx[0][K_true], idx[1][K_true]
            edge_ep[f'true_edge_{K_true}'] = float(pic_corr[i, j])
    
    return X, W_true, tree, edge_ep, p99, pic_data

# ============================================================
# EXPERIMENT A: Nonlinear SEM validation (5 key configs)
# ============================================================

def experiment_a_nonlinear(ckpt):
    print("\n" + "="*60)
    print("EXPERIMENT A: Nonlinear SEM Validation")
    print("="*60)
    
    a_data = ckpt.get('exp_a_nonlinear', {})
    
    configs = [
        {'d': 30, 'alpha': 0.02, 'n': 200},
        {'d': 50, 'alpha': 0.02, 'n': 200},
        {'d': 100, 'alpha': 0.02, 'n': 200},
        {'d': 30, 'alpha': 0.10, 'n': 200},
        {'d': 50, 'alpha': 0.10, 'n': 200},
    ]
    
    generators = {
        'linear': generate_data,
        'mlp': generate_data_nonlinear_mlp,
        'sigmoid': generate_data_nonlinear_sigmoid,
    }
    
    for cfg in configs:
        for gen_name, gen_fn in generators.items():
            key = f"d={cfg['d']}_a={cfg['alpha']}_n={cfg['n']}_{gen_name}"
            if key in a_data:
                print(f"  [{gen_name}] {key}: CACHED")
                continue
            
            t0 = time.time()
            sr_pic, sr_ct = [], []
            
            for s in SEEDS:
                if gen_name == 'linear':
                    X, Wt, tr, ep = gen_fn(cfg['n'], cfg['d'], 10, cfg['alpha'], s)
                    pic_data = felsenstein_pic(X, tr)
                    pic_corr = np.corrcoef(pic_data.T)
                    triu = np.triu_indices(cfg['d'], k=1)
                    p99 = float(np.percentile(np.abs(pic_corr[triu]), 99))
                    pic_W = (np.abs(pic_corr) > p99).astype(float)
                    pic_f1 = compute_f1(pic_W, Wt, 0.3)[0]
                elif gen_name == 'mlp':
                    X, Wt, tr, ep, p99, pic_data = gen_fn(cfg['n'], cfg['d'], 10, cfg['alpha'], s)
                    pic_corr = np.corrcoef(pic_data.T)
                    pic_W = (np.abs(pic_corr) > p99).astype(float)
                    pic_f1 = compute_f1(pic_W, Wt, 0.3)[0]
                else:  # sigmoid
                    X, Wt, tr, ep, p99, pic_data = gen_fn(cfg['n'], cfg['d'], 10, cfg['alpha'], s)
                    pic_corr = np.corrcoef(pic_data.T)
                    pic_W = (np.abs(pic_corr) > p99).astype(float)
                    pic_f1 = compute_f1(pic_W, Wt, 0.3)[0]
                
                ct_r = run_ct(X, cfg['d'], n_epochs=1200, seed=s)
                ct_f1 = compute_f1(np.array(ct_r['W']), Wt, 0.3)[0]
                sr_pic.append(round(pic_f1, 4))
                sr_ct.append(round(ct_f1, 4))
            
            elapsed = time.time() - t0
            a_data[key] = {
                'pic_f1': round(float(np.mean(sr_pic)), 4),
                'pic_best': round(float(np.max(sr_pic)), 4),
                'ct_f1': round(float(np.mean(sr_ct)), 4),
                'ct_best': round(float(np.max(sr_ct)), 4),
                'time_s': round(elapsed, 1),
            }
            
            consistent = (a_data[key]['pic_f1'] <= 0.35) == (a_data.get(f"d={cfg['d']}_a={cfg['alpha']}_n={cfg['n']}_linear", {}).get('pic_f1', -1) <= 0.35)
            print(f"  [{gen_name}] {key}: PIC={a_data[key]['pic_f1']:.3f} CT={a_data[key]['ct_f1']:.3f} [{elapsed:.0f}s] {'CONSISTENT' if consistent else '***DIVERGENT***'}")
            
            ckpt['exp_a_nonlinear'] = a_data
            save_ckpt(ckpt)
    
    ckpt['exp_a_nonlinear'] = a_data
    save_ckpt(ckpt)
    return ckpt

# ============================================================
# EXPERIMENT B: Wall Penetration (n->inf, alpha->0)
# ============================================================

def experiment_b_penetrate(ckpt):
    print("\n" + "="*60)
    print("EXPERIMENT B: Wall Penetration (n->inf, alpha->0)")
    print("="*60)
    
    b_data = ckpt.get('exp_b_penetration', {})
    
    # Scan n from 100 to 5000
    n_vals = [100, 200, 500, 1000, 2000, 5000]
    alpha_vals = [0.01, 0.005, 0.001, 0.0001, 0.0]
    d = 50  # Fixed dimension where wall was sharpest
    n_true = 10
    
    for n_val in n_vals:
        for alpha_val in alpha_vals:
            key = f"n={n_val}_a={alpha_val}"
            if key in b_data:
                print(f"  {key}: CACHED")
                continue
            
            t0 = time.time()
            sr_pic, sr_ct = [], []
            
            for s in SEEDS[:2]:  # 2 seeds for speed (large n)
                X, Wt, tr, ep = generate_data(n_val, d, n_true, alpha_val, s)
                
                # PIC+corr
                pic_data = felsenstein_pic(X, tr)
                pic_corr = np.corrcoef(pic_data.T)
                triu = np.triu_indices(d, k=1)
                triu_vals = np.abs(pic_corr[triu])
                p99 = float(np.percentile(triu_vals, 99))
                pic_W = (np.abs(pic_corr) > p99).astype(float)
                pic_f1 = compute_f1(pic_W, Wt, 0.3)[0]
                sr_pic.append(round(pic_f1, 4))
                
                # CT (only for n<=1000 to save time)
                if n_val <= 1000:
                    ct_r = run_ct(X, d, n_epochs=1200, seed=s)
                    ct_f1 = compute_f1(np.array(ct_r['W']), Wt, 0.3)[0]
                    sr_ct.append(round(ct_f1, 4))
                else:
                    sr_ct.append(0.0)  # placeholder for large n
            
            elapsed = time.time() - t0
            b_data[key] = {
                'pic_mean': round(float(np.mean(sr_pic)), 4),
                'pic_max': round(float(np.max(sr_pic)), 4),
                'ct_mean': round(float(np.mean(sr_ct)), 4) if len(sr_ct) > 1 else None,
                'penetrated': sr_pic[0] > 0.1,  # F1 > 0.1 = wall penetrated
                'time_s': round(elapsed, 1),
            }
            
            p_str = "PENETRATED" if b_data[key]['penetrated'] else "WALL HOLDS"
            print(f"  {key}: PIC={b_data[key]['pic_mean']:.4f} [{elapsed:.0f}s] {p_str}")
            
            ckpt['exp_b_penetration'] = b_data
            save_ckpt(ckpt)
    
    ckpt['exp_b_penetration'] = b_data
    save_ckpt(ckpt)
    return ckpt

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CKPT}")
    
    ckpt = load_ckpt()
    print(f"Loaded checkpoint with keys: {list(ckpt.keys())}")
    
    ckpt = experiment_a_nonlinear(ckpt)
    ckpt = experiment_b_penetrate(ckpt)
    
    print("\n" + "="*60)
    print("ALL DONE")
    print("="*60)
    
    # Summary
    a_data = ckpt.get('exp_a_nonlinear', {})
    b_data = ckpt.get('exp_b_penetration', {})
    
    print(f"\nExp A: {len(a_data)} nonlinear configs completed")
    print(f"Exp B: {len(b_data)} penetration configs completed")
    
    # Find penetration threshold
    penetrated = [(k, v['pic_mean']) for k, v in b_data.items() if v.get('penetrated')]
    if penetrated:
        print(f"\nWALL PENETRATED at {len(penetrated)} configs:")
        for k, f1 in sorted(penetrated, key=lambda x: -x[1])[:5]:
            print(f"  {k}: F1={f1:.4f}")
    else:
        print("\nWALL NOT PENETRATED in any config tested")
    
    # Consistency check
    linear_keys = [k for k in a_data if 'linear' in k]
    nonlinear_keys = [k for k in a_data if 'linear' not in k]
    divergent = []
    for lk in linear_keys:
        base_cfg = lk.replace('_linear', '')
        base_d_a = '_'.join(base_cfg.split('_')[:2])
        linear_f1 = a_data[lk]['pic_f1']
        for nlk in nonlinear_keys:
            if nlk.startswith(base_d_a):
                nl_f1 = a_data[nlk]['pic_f1']
                # Consistent if both are <= 0.35 or both > 0.35
                if (linear_f1 > 0.35) != (nl_f1 > 0.35):
                    divergent.append(f"{lk} vs {nlk}: linear={linear_f1:.3f} vs nonlinear={nl_f1:.3f}")
    
    if divergent:
        print(f"\n*** DIVERGENT CONFIGS ({len(divergent)}) ***")
        for d in divergent:
            print(f"  {d}")
    else:
        print(f"\nAll {len(linear_keys) + len(nonlinear_keys)} configs CONSISTENT: wall conclusion robust to nonlinearity")
