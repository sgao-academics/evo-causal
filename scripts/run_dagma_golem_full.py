"""
FULL DAGMA + GOLEM sweep: same configs as overnight phase diagram.
Runs ALL (d,K,seed) combinations for fair comparison.
Independent checkpoint: results/dagma_golem_full.json
"""

import sys, os, json, time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
sys.path.insert(0, os.path.dirname(__file__))  # portable
from felsenstein_pic import PhyloNode, felsenstein_pic
# causalscale should be installed via: pip install causalscale

import torch

CKPT = r'C:\Users\高帅东\Desktop\evo_causal\results\dagma_golem_full.json'
OVERNIGHT = r'C:\Users\高帅东\Desktop\evo_causal\results\overnight_massive.json'
DEVICE = 'cpu'  # CPU-only: don't compete with CT for GPU VRAM

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

def build_tree(n):
    tips = []
    for i in range(n):
        t = PhyloNode(f"Sp{i}", children=[], branch_length=max(0.01, np.random.exponential(1.0)))
        t.index = i; t.is_tip = True; tips.append(t)
    nodes = tips; nid = 0
    while len(nodes) > 1:
        new_nodes = []
        for i in range(0, len(nodes), 2):
            if i + 1 < len(nodes):
                p = PhyloNode(f"N{nid}", children=[nodes[i], nodes[i+1]],
                             branch_length=max(0.01, np.random.exponential(0.5)))
                p.is_tip = False; nid += 1; new_nodes.append(p)
            else: new_nodes.append(nodes[i])
        nodes = new_nodes
    return nodes[0]

def generate_data(n=200, d=50, n_true=10, K=0.15, seed=42):
    rng = np.random.default_rng(seed); tree = build_tree(n)
    L = np.column_stack([rng.normal(0, 1, n) for _ in range(d)])
    ev, evec = np.linalg.eigh(L @ L.T + 1e-6 * np.eye(n)); ev = np.maximum(ev, 0)
    chol = evec @ np.diag(np.sqrt(ev)) @ evec.T
    Wt = np.zeros((d, d)); ep = []
    for _ in range(n_true):
        i, j = int(rng.integers(0, d)), int(rng.integers(0, d))
        if i != j: Wt[i, j] = rng.uniform(0.3, 0.7) * rng.choice([-1, 1]); ep.append((i, j))
    Z = rng.normal(0, 1, (n, d))
    Xc = Z @ np.linalg.inv(np.eye(d) - Wt)
    Xp = chol @ rng.normal(0, 1, (n, d))
    X = np.sqrt(1 - K) * Xc + np.sqrt(K) * Xp
    return (X - X.mean(0)) / (X.std(0) + 1e-8), Wt, tree, ep

def compute_f1(We, Wt, th=0.3):
    Wb = np.abs(We) > th; Wtt = np.abs(Wt) > 0
    tp = int((Wb & Wtt).sum()); fp = int((Wb & ~Wtt).sum()); fn = int((~Wb & Wtt).sum())
    p = tp/(tp+fp) if tp+fp>0 else 0; r = tp/(tp+fn) if tp+fn>0 else 0
    return (2*p*r/(p+r) if p+r>0 else 0), p, r, tp, fp, fn

def run_dagma(X):
    try:
        from dagma.linear import DagmaLinear
        return DagmaLinear(loss_type='l2').fit(
            X.astype(np.float64), lambda1=0.03, w_threshold=0.3, T=5, lr=0.0003)
    except Exception as e: print(f"    DAGMA err: {e}"); return np.zeros((X.shape[1],)*2)

def run_golem(X):
    try:
        import torch as t
        from golem.golem import GolemEV
        Xt = t.tensor(X.astype(np.float32), device=DEVICE)
        m = GolemEV(Xt.shape[1], lambda_1=0.02, lambda_2=5.0,
                    learning_rate=1e-3, device=DEVICE)
        W = m.fit(Xt, num_iter=100000)
        return W.cpu().numpy() if hasattr(W, 'cpu') else W
    except ImportError: return None
    except Exception as e: print(f"    GOLEM err: {e}"); return None


if __name__ == '__main__':
    print(f"DAGMA+GOLEM FULL SWEEP — {time.strftime('%H:%M:%S')}")
    print(f"Device: {DEVICE}")

    # Check available methods
    has_dagma = has_golem = False
    try: from dagma.linear import DagmaLinear; has_dagma = True
    except: pass
    try: from golem.golem import GolemEV; has_golem = True
    except: pass
    print(f"Available: DAGMA={has_dagma}, GOLEM={has_golem}")

    if not has_dagma and not has_golem:
        print("Neither DAGMA nor GOLEM installed. Exiting.")
        sys.exit(0)

    # Wait for overnight to define configs
    print("Waiting for overnight phase diagram config list...")
    d_vals = K_vals = None
    for _ in range(40):  # max 10 min wait
        if os.path.exists(OVERNIGHT):
            ov = json.load(open(OVERNIGHT))
            pd = ov.get('phase_diagram', {})
            if len(pd) >= 5:
                d_vals = sorted(set(int(k.split('_')[0].split('=')[1]) for k in pd))
                K_vals = sorted(set(float(k.split('_')[1].split('=')[1]) for k in pd))
                break
        time.sleep(15)
    if d_vals is None: d_vals, K_vals = [30, 50, 100, 150, 200], [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]

    print(f"Configs: {len(d_vals)} d x {len(K_vals)} K = {len(d_vals)*len(K_vals)} total")
    seeds = [42, 123, 456]
    n = 200
    ckpt = load_ckpt()

    # === DAGMA ===
    if has_dagma:
        dagma_res = ckpt.get('dagma', {})
        for d_val in d_vals:
            for K_val in K_vals:
                key = f"d={d_val}_K={K_val}"
                if key in dagma_res: continue

                t0 = time.time(); print(f"\n  DAGMA {key}...")
                sr = []
                for seed in seeds:
                    X, Wt, tree, ep = generate_data(n, d_val, 10, K_val, seed)
                    Wd = run_dagma(X)
                    f1 = compute_f1(Wd, Wt, 0.3)[0]
                    edges = int((np.abs(Wd) > 0.3).sum())
                    sr.append({'seed': seed, 'f1': round(f1, 4), 'edges': edges})

                f1s = [s['f1'] for s in sr]
                dagma_res[key] = {
                    'd': d_val, 'K': K_val,
                    'f1_mean': round(float(np.mean(f1s)), 4),
                    'f1_std': round(float(np.std(f1s)), 4),
                    'f1_best': round(float(np.max(f1s)), 4),
                    'edges_mean': round(float(np.mean([s['edges'] for s in sr])), 1),
                    'time_s': round(time.time() - t0, 1),
                    'seeds': sr,
                }
                ckpt['dagma'] = dagma_res; save_ckpt(ckpt)
                print(f"    DAGMA F1={dagma_res[key]['f1_mean']:.3f} [{time.time()-t0:.0f}s]")

        ckpt['dagma'] = dagma_res; save_ckpt(ckpt)

    # === GOLEM ===
    if has_golem:
        golem_res = ckpt.get('golem', {})
        for d_val in d_vals:
            for K_val in K_vals:
                key = f"d={d_val}_K={K_val}"
                if key in golem_res: continue

                t0 = time.time(); print(f"\n  GOLEM {key}...")
                sr = []
                for seed in seeds:
                    X, Wt, tree, ep = generate_data(n, d_val, 10, K_val, seed)
                    Wg = run_golem(X)
                    if Wg is not None:
                        f1 = compute_f1(Wg, Wt, 0.3)[0]
                        edges = int((np.abs(Wg) > 0.3).sum())
                        sr.append({'seed': seed, 'f1': round(f1, 4), 'edges': edges})
                    else:
                        sr.append({'seed': seed, 'f1': None, 'edges': 0})

                valid = [s['f1'] for s in sr if s['f1'] is not None]
                golem_res[key] = {
                    'd': d_val, 'K': K_val,
                    'f1_mean': round(float(np.mean(valid)), 4) if valid else None,
                    'f1_best': round(float(np.max(valid)), 4) if valid else None,
                    'time_s': round(time.time() - t0, 1),
                    'seeds': sr,
                }
                ckpt['golem'] = golem_res; save_ckpt(ckpt)
                gf = golem_res[key].get('f1_mean')
                print(f"    GOLEM F1={gf} [{time.time()-t0:.0f}s]")

        ckpt['golem'] = golem_res; save_ckpt(ckpt)

    print(f"\nDAGMA+GOLEM DONE at {time.strftime('%H:%M:%S')}")
