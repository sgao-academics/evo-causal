"""
MASTER ALL-IN-ONE: Sequential GPU pipeline for 10-hour overnight run.
Every task uses GPU where possible, runs one at a time (no OOM risk).

Pipeline:
  Phase 1: PIC validation + K-sweep + CT gradient trace + CT convergence
  Phase 2: Full phase diagram (d x K) — PIC+corr + CT + NOTEARS
  Phase 3: CT architecture ablation + n-scaling
  Phase 4: DAGMA + GOLEM full sweep
  Phase 5: Multi-method head-to-head at key configs

All results in: results/master_all.json (checkpoint, resumable)
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
from scipy.optimize import minimize

CKPT = r'C:\Users\高帅东\Desktop\evo_causal\results\master_all.json'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}\n{'='*60}")

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
# Shared utilities
# ============================================================================
def build_tree(n, seed=42):
    rng = np.random.default_rng(seed)
    tips = []
    for i in range(n):
        t = PhyloNode(f"Sp{i}", children=[], branch_length=max(0.01, rng.exponential(1.0)))
        t.index = i; t.is_tip = True; tips.append(t)
    nodes = tips; nid = 0
    while len(nodes) > 1:
        new = []
        for i in range(0, len(nodes), 2):
            if i+1 < len(nodes):
                p = PhyloNode(f"N{nid}", children=[nodes[i], nodes[i+1]],
                             branch_length=max(0.01, rng.exponential(0.5)))
                p.is_tip = False; nid += 1; new.append(p)
            else: new.append(nodes[i])
        nodes = new
    return nodes[0]

def generate_data(n=200, d=50, n_true=10, K=0.15, seed=42):
    rng = np.random.default_rng(seed); tree = build_tree(n, seed=seed+1000)
    L = np.column_stack([rng.normal(0, 1, n) for _ in range(d)])
    ev, evec = np.linalg.eigh(L @ L.T + 1e-6 * np.eye(n)); ev = np.maximum(ev, 0)
    chol = evec @ np.diag(np.sqrt(ev)) @ evec.T
    Wt = np.zeros((d, d)); ep = []
    for _ in range(n_true):
        i, j = int(rng.integers(0, d)), int(rng.integers(0, d))
        if i != j: Wt[i, j] = rng.uniform(0.3, 0.7) * rng.choice([-1, 1]); ep.append((i, j))
    Z = rng.normal(0, 1, (n, d))
    Xc = Z @ np.linalg.inv(np.eye(d) - Wt); Xp = chol @ rng.normal(0, 1, (n, d))
    X = np.sqrt(1 - K) * Xc + np.sqrt(K) * Xp
    return (X - X.mean(0)) / (X.std(0) + 1e-8), Wt, tree, ep

def compute_f1(We, Wt, th=0.3):
    Wb = np.abs(We) > th; Wtt = np.abs(Wt) > 0
    tp = int((Wb & Wtt).sum()); fp = int((Wb & ~Wtt).sum()); fn = int((~Wb & Wtt).sum())
    p = tp/(tp+fp) if tp+fp>0 else 0; r = tp/(tp+fn) if tp+fn>0 else 0
    return (2*p*r/(p+r) if p+r>0 else 0), p, r, tp, fp, fn

def run_ct(X, d, n_epochs=2000, d_model=64, n_heads=4, n_layers=2,
           edge_threshold=0.3, lr=0.001, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    n, _d = X.shape; Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    cfg = CausalTransformerConfig(d_vars=d, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, lambda_dag=0.5, lambda_sparsity=0.01, lr=lr, edge_threshold=edge_threshold)
    model = CausalTransformer(cfg).to(DEVICE); model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    bs = min(128, n); eh, dh = [], []
    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]; xb = Xt[idx]
            Wb, _ = model(xb); losses = model.compute_loss(xb, Wb)
            opt.zero_grad(); losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        Wm = Wb.mean(dim=0).detach()
        eh.append(int((torch.abs(Wm) > edge_threshold).float().sum().item()))
        dh.append(float(losses['dag']))
    model.eval()
    with torch.no_grad():
        Wf, _ = model(Xt[:min(500, n)]); Wmn = Wf.mean(dim=0).cpu().numpy()
        fh = float(NOTEARSConstraint()(torch.tensor(Wmn, device=DEVICE)).item())
    del model, opt, Xt; torch.cuda.empty_cache()
    return {'W': Wmn.tolist(), 'final_h': round(fh, 6),
            'final_edges': int((np.abs(Wmn) > edge_threshold).sum()),
            'edge_plateau': float(np.mean(eh[-200:])), 'h_plateau': float(np.mean(dh[-200:]))}

def run_pic_corr(X, tree, Wt, ep, d, pct=99):
    pic = felsenstein_pic(X, tree); cc = np.corrcoef(pic.T)
    ti = np.triu_indices(d, k=1); th = np.percentile(np.abs(cc[ti]), pct)
    adj = np.abs(cc) > th; np.fill_diagonal(adj, False)
    f1 = compute_f1(adj.astype(float), Wt, 0.5)[0]
    bm = np.ones((d, d), dtype=bool); np.fill_diagonal(bm, False)
    bg = float(np.abs(cc[bm]).mean())
    tc = float(np.mean([np.abs(cc[i, j]) for i, j in ep if i < d and j < d]))
    return {'f1': round(f1, 4), 'bg_corr': round(bg, 6), 'true_corr': round(tc, 6),
            'enrichment': round(tc / max(bg, 1e-10), 4)}

def run_notears(X):
    try:
        W = notears_linear(X.astype(np.float64), lambda1=0.1, loss_type='l2', max_iter=100, w_threshold=0.3)
        return W
    except: return np.zeros((X.shape[1], X.shape[1]))

def run_dagma(X):
    try:
        from dagma.linear import DagmaLinear
        return DagmaLinear(loss_type='l2').fit(X.astype(np.float64), lambda1=0.03, w_threshold=0.3, T=5, lr=0.0003)
    except: return np.zeros((X.shape[1],)*2)

def run_golem(X):
    """GOLEM-EV: likelihood-based causal discovery with L-BFGS-B."""
    n, d = X.shape
    w_vec = np.random.randn(d*d)*0.1
    rho, alpha = 0.1, 0.0
    best_w, best_h = w_vec.copy(), 1e10
    X_t = torch.tensor(X, device=DEVICE, dtype=torch.float64)

    for it in range(35):
        def loss_golem(w):
            W = w.reshape(d,d)
            W_t = torch.tensor(W, device=DEVICE, dtype=torch.float64)
            diff = X_t - X_t @ W_t
            rss = max(torch.sum(diff*diff).item(), 1e-20)
            ll = 0.5 * d * np.log(rss / n)
            M = torch.eye(d, device=DEVICE, dtype=torch.float64) - W_t
            logdet = torch.log(torch.clamp(torch.abs(torch.det(M)), min=1e-20)).item()
            l1 = 0.01 * np.sum(np.abs(W))
            W2 = W_t * W_t
            hv = torch.trace(torch.linalg.matrix_exp(W2)).item() - d
            return ll - logdet + l1 + 0.5*rho*hv**2 + alpha*hv

        def loss_golem_grad(w):
            W = w.reshape(d,d)
            W_t = torch.tensor(W, device=DEVICE, dtype=torch.float64)
            diff = X_t - X_t @ W_t
            rss = max(torch.sum(diff*diff).item(), 1e-20)
            grad_ll = -d * (X_t.T @ diff).cpu().numpy() / rss
            M = torch.eye(d, device=DEVICE, dtype=torch.float64) - W_t
            try: grad_logdet = torch.linalg.inv(M).T.cpu().numpy()
            except: grad_logdet = np.zeros((d, d))
            grad_l1 = 0.01 * np.sign(W)
            W2 = W_t * W_t
            expW2 = torch.linalg.matrix_exp(W2)
            hv = torch.trace(expW2).item() - d
            dh = (2 * W_t * expW2.T).cpu().numpy()
            return (grad_ll + grad_logdet + grad_l1 + rho*hv*dh + alpha*dh).flatten()

        res = minimize(loss_golem, w_vec, method='L-BFGS-B', jac=loss_golem_grad,
                       options={'maxiter':120,'ftol':1e-10,'gtol':1e-8})
        w_vec = res.x; W = w_vec.reshape(d,d)
        W_t = torch.tensor(W, device=DEVICE, dtype=torch.float64)
        hv = torch.trace(torch.linalg.matrix_exp(W_t*W_t)).item() - d
        alpha += rho*hv
        if abs(hv)<best_h: best_h,best_w = abs(hv),w_vec.copy()
        if abs(hv)<1e-7: break
        if hv>0: rho = min(rho*2,1e10)
    return best_w.reshape(d,d)

# ============================================================================
# Phase 1: Reviewer experiments (PIC validation + K-sweep + CT trace + CT convergence)
# ============================================================================
def phase1(ckpt):
    if 'p1' in ckpt:
        print("[P1] Done, skip"); return ckpt
    print("\n" + "="*60 + "\nPHASE 1: Reviewer experiments\n" + "="*60)

    p1_data = ckpt.get('p1', {'pic_val': None, 'k_sweep': {}, 'ct_trace': {}, 'ct_convergence': {}})

    # 1a: PIC validation
    if p1_data.get('pic_val') is not None:
        print("\n--- PIC Validation [CACHED] ---")
        pic_val = p1_data['pic_val']
    else:
        print("\n--- PIC Validation ---")
    tip_a = PhyloNode("A", children=[], branch_length=1.0); tip_a.index = 0; tip_a.is_tip = True
    tip_b = PhyloNode("B", children=[], branch_length=1.0); tip_b.index = 1; tip_b.is_tip = True
    root = PhyloNode("r", children=[tip_a, tip_b], branch_length=0.0); root.is_tip = False
    c = felsenstein_pic(np.array([[5.0],[7.0]]), root)
    pic_val = {'2sp_pass': bool(abs(c[0,0] + 2/np.sqrt(2)) < 1e-6),
               '2sp_error': float(abs(c[0,0] + 2/np.sqrt(2)))}

    bt = build_tree(80); Xb = np.random.default_rng(99).normal(0, 1, (80, 10))
    pb = felsenstein_pic(Xb, bt)
    pic_val['contrast_count'] = int(pb.shape[0])
    pic_val['contrast_pass'] = bool(pb.shape[0] == 79)
    print(f"  PIC: 2sp={pic_val['2sp_pass']}, contrasts={pic_val['contrast_count']}/79 "
          f"PASS={pic_val['contrast_pass']}")

    # 1b: K-sweep
    if p1_data.get("k_sweep"):
        print("\n--- K-sweep [CACHED] ---")
        ks_results = p1_data["k_sweep"]
    else:
        print("\n--- K-sweep (17 K values x 3 seeds) ---")
    Kv = [0.02, 0.05, 0.08, 0.10, 0.12, 0.14, 0.15, 0.18, 0.20,
          0.22, 0.25, 0.28, 0.30, 0.33, 0.35, 0.38, 0.40]
    seeds = [42, 123, 456]
    ks_results = {}
    for K in Kv:
        sk = f'K={K}'; sr = []
        for s in seeds:
            X, Wt, tr, ep = generate_data(100, 50, 10, K, s)
            r = run_pic_corr(X, tr, Wt, ep, 50); sr.append(r)
        ks_results[sk] = {'K': K, 'f1_mean': round(float(np.mean([r['f1'] for r in sr])), 4),
                          'enrich_mean': round(float(np.mean([r['enrichment'] for r in sr])), 4)}
        print(f"  K={K}: F1={ks_results[sk]['f1_mean']:.3f} enrich={ks_results[sk]['enrich_mean']:.1f}x")

        ckpt["p1"] = p1_data; save_ckpt(ckpt)

    # 1c: CT gradient trace vs K
    if p1_data.get("ct_trace"):
        print("\n--- CT Gradient Trace [CACHED] ---")
        ct_trace = p1_data["ct_trace"]
    else:
        print("\n--- CT Gradient Trace vs K ---")
    rng = np.random.default_rng(42); tree = build_tree(100)
    L = np.column_stack([rng.normal(0, 1, 100) for _ in range(50)])
    ev, evec = np.linalg.eigh(L @ L.T + 1e-6 * np.eye(100)); ev = np.maximum(ev, 0)
    chol = evec @ np.diag(np.sqrt(ev)) @ evec.T
    Wt = np.zeros((50, 50)); ep = []
    for _ in range(10):
        i, j = int(rng.integers(0, 50)), int(rng.integers(0, 50))
        if i != j: Wt[i, j] = rng.uniform(0.3, 0.7) * rng.choice([-1, 1]); ep.append((i, j))
    Z = rng.normal(0, 1, (100, 50)); Xc = Z @ np.linalg.inv(np.eye(50) - Wt)
    Xp_base = chol @ rng.normal(0, 1, (100, 50))
    ct_trace = {}
    for K in [0.02, 0.05, 0.08, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        X = np.sqrt(1-K)*Xc + np.sqrt(K)*Xp_base; X = (X-X.mean(0))/(X.std(0)+1e-8)
        ct_r = run_ct(X, 50, n_epochs=1200)
        f1 = compute_f1(np.array(ct_r['W']), Wt, 0.3)[0]
        ct_trace[f'K={K}'] = {'f1': round(f1, 4), 'edges': ct_r['final_edges'],
                               'h_plateau': ct_r['h_plateau']}
        print(f"  K={K}: CT_F1={f1:.3f} edges={ct_r['final_edges']} h={ct_r['h_plateau']:.2e}")

        ckpt["p1"] = p1_data; save_ckpt(ckpt)

    # 1d: CT convergence (3 seeds x 3000 epochs)
    if p1_data.get("ct_convergence"):
        print("\n--- CT Convergence [CACHED] ---")
        conv_results = p1_data["ct_convergence"]
    else:
        print("\n--- CT Convergence (3 seeds x 3000 epochs) ---")
    conv_results = {}
    for s in seeds:
        X, Wt, tr, ep = generate_data(100, 50, 10, 0.15, s)
        ct_r = run_ct(X, 50, n_epochs=3000)
        f1 = compute_f1(np.array(ct_r['W']), Wt, 0.3)[0]
        conv_results[f'seed={s}'] = {'f1': round(f1, 4), 'edges': ct_r['final_edges'],
                                      'h': ct_r['h_plateau']}
        print(f"  seed={s}: F1={f1:.3f} edges={ct_r['final_edges']} h={ct_r['h_plateau']:.2e}")

    p1_data = ckpt.get('p1', {'pic_val': None, 'k_sweep': {}, 'ct_trace': {}, 'ct_convergence': {}})
    p1_data['pic_val'] = pic_val; p1_data['k_sweep'] = ks_results; ckpt['p1'] = p1_data; save_ckpt(ckpt)
    p1_data['ct_trace'] = ct_trace; ckpt['p1'] = p1_data; save_ckpt(ckpt)
    p1_data['ct_convergence'] = conv_results; ckpt['p1'] = p1_data; save_ckpt(ckpt)
    return ckpt


# ============================================================================
# Phase 2: Full phase diagram (d x K) — PIC+corr + CT + NOTEARS
# ============================================================================
def phase2(ckpt):
    if 'p2' in ckpt:
        print("[P2] Done, skip"); return ckpt
    print("\n" + "="*60 + "\nPHASE 2: Phase diagram (5d x 8K x 3 seeds)\n" + "="*60)

    d_vals = [30, 50, 100, 150, 200]; K_vals = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
    seeds = [42, 123, 456]; n = 200
    results = ckpt.get('p2', {})

    for d_val in d_vals:
        for K_val in K_vals:
            key = f'd={d_val}_K={K_val}'
            if key in results: continue
            t0 = time.time(); print(f"\n  {key}...")
            sr = []
            for s in seeds:
                X, Wt, tr, ep = generate_data(n, d_val, 10, K_val, s)
                pic = run_pic_corr(X, tr, Wt, ep, d_val)
                ct = run_ct(X, d_val, n_epochs=2000)
                ct_f1 = compute_f1(np.array(ct['W']), Wt, 0.3)[0]
                Wn = run_notears(X); n_f1 = compute_f1(Wn, Wt, 0.3)[0]
                sr.append({'seed': s, 'pic_f1': pic['f1'], 'pic_enrich': pic['enrichment'],
                           'ct_f1': round(ct_f1, 4), 'note_f1': round(n_f1, 4),
                           'ct_edges': ct['final_edges']})
            results[key] = {'d': d_val, 'K': K_val,
                'pic_f1': round(float(np.mean([s['pic_f1'] for s in sr])), 4),
                'ct_f1': round(float(np.mean([s['ct_f1'] for s in sr])), 4),
                'note_f1': round(float(np.mean([s['note_f1'] for s in sr])), 4),
                'ct_best': round(float(np.max([s['ct_f1'] for s in sr])), 4),
                'time_s': round(time.time()-t0, 1), 'seeds': sr}
            ckpt['p2'] = results; save_ckpt(ckpt)
            print(f"    PIC={results[key]['pic_f1']:.3f} CT={results[key]['ct_f1']:.3f} "
                  f"NOTEARS={results[key]['note_f1']:.3f} [{time.time()-t0:.0f}s]")
    ckpt['p2'] = results; save_ckpt(ckpt); return ckpt


# ============================================================================
# Phase 3: CT architecture ablation + N-scaling
# ============================================================================
def phase3(ckpt):
    if 'p3' in ckpt:
        print("[P3] Done, skip"); return ckpt
    print("\n" + "="*60 + "\nPHASE 3: Architecture ablation + N-scaling\n" + "="*60)

    seeds = [42, 123, 456]; n, d, K = 200, 100, 0.15
    arch = {}

    # Architecture ablation
    print("--- CT Architecture Ablation ---")
    p3_data = ckpt.get('p3', {'arch': {}, 'n_scaling': {}})
    arch = p3_data.get('arch', {})
    nsc = p3_data.get('n_scaling', {})
    X0, Wt, tr0, ep0 = generate_data(n, d, 10, K, 42)
    for dm in [32, 64, 128]:
        for nl in [2, 4]:
            key = f'dm={dm}_nl={nl}'
            if key in arch: continue
            sr = []
            for s in seeds:
                torch.manual_seed(s); np.random.seed(s)
                ct = run_ct(X0, d, n_epochs=2000, d_model=dm, n_layers=nl)
                f1 = compute_f1(np.array(ct['W']), Wt, 0.3)[0]
                sr.append({'f1': round(f1, 4), 'edges': ct['final_edges']})
            arch[key] = {'f1_mean': round(float(np.mean([s['f1'] for s in sr])), 4),
                         'f1_best': round(float(np.max([s['f1'] for s in sr])), 4)}
            ckpt['p3'] = {'arch': arch, 'n_scaling': nsc}; save_ckpt(ckpt)
            print(f"  {key}: F1={arch[key]['f1_mean']:.3f} best={arch[key]['f1_best']:.3f}")

    # N-scaling
    print("--- N-scaling ---")
    # nsc already initialized above
    for n_val in [50, 100, 200, 400, 800]:
        key = f'n={n_val}'
        if key in nsc: continue
        sr = []
        for s in seeds:
            X, Wt, tr, ep = generate_data(n_val, d, 10, K, s)
            pic = run_pic_corr(X, tr, Wt, ep, d)
            ct = run_ct(X, d, n_epochs=2000); ct_f1 = compute_f1(np.array(ct['W']), Wt, 0.3)[0]
            sr.append({'pic_f1': pic['f1'], 'ct_f1': round(ct_f1, 4)})
        nsc[key] = {'n': n_val, 'pic_f1': round(float(np.mean([s['pic_f1'] for s in sr])), 4),
                    'ct_f1': round(float(np.mean([s['ct_f1'] for s in sr])), 4)}
        ckpt['p3'] = {'arch': arch, 'n_scaling': nsc}; save_ckpt(ckpt)
        print(f"  {key}: PIC={nsc[key]['pic_f1']:.3f} CT={nsc[key]['ct_f1']:.3f}")

    p3_data = ckpt.get('p3', {'arch': {}, 'n_scaling': {}})
    p3_data['arch'] = arch; p3_data['n_scaling'] = nsc; ckpt['p3'] = p3_data; save_ckpt(ckpt)
    return ckpt


# ============================================================================
# Phase 4: DAGMA + GOLEM full sweep (same configs as phase 2)
# ============================================================================
def phase4(ckpt):
    print("\n" + "="*60 + "\nPHASE 4: DAGMA + GOLEM\n" + "="*60)
    p4_data = ckpt.get('p4', {})
    d_vals = [30, 50, 100, 150, 200]
    K_vals = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
    seeds = [42, 123, 456]; n = 200
    from dagma.linear import DagmaLinear

    for d_val in d_vals:
        for K_val in K_vals:
            key = f'd={d_val}_K={K_val}'
            entry = p4_data.get(key, {})
            rerun = False

            # DAGMA
            if 'dagma_f1' not in entry:
                t0 = time.time(); print(f"\n  DAGMA {key}...")
                sr = []
                for s in seeds:
                    X, Wt, tr, ep = generate_data(n, d_val, 10, K_val, s)
                    Wd = run_dagma(X); f1 = compute_f1(Wd, Wt, 0.3)[0]
                    sr.append({'f1': round(f1, 4)})
                entry['dagma_f1'] = round(float(np.mean([s['f1'] for s in sr])), 4)
                entry['dagma_best'] = round(float(np.max([s['f1'] for s in sr])), 4)
                entry['dagma_time'] = round(time.time() - t0, 1)
                rerun = True
                print(f"    DAGMA F1={entry['dagma_f1']:.3f}")

            # GOLEM (inline implementation)
            if 'golem_f1' not in entry:
                t0 = time.time(); print(f"\n  GOLEM {key}...")
                sr = []
                for s in seeds:
                    X, Wt, tr, ep = generate_data(n, d_val, 10, K_val, s)
                    Wg = run_golem(X)
                    f1 = compute_f1(Wg, Wt, 0.3)[0] if Wg is not None else -1
                    sr.append({'f1': round(f1, 4) if f1 >= 0 else None})
                valid = [s['f1'] for s in sr if s['f1'] is not None]
                entry['golem_f1'] = round(float(np.mean(valid)), 4) if valid else None
                entry['golem_best'] = round(float(np.max(valid)), 4) if valid else None
                entry['golem_time'] = round(time.time() - t0, 1)
                rerun = True
                gf = entry.get('golem_f1')
                print(f"    GOLEM F1={gf}")

            if rerun:
                p4_data[key] = entry; ckpt['p4'] = p4_data; save_ckpt(ckpt)

    ckpt['p4'] = p4_data; save_ckpt(ckpt); return ckpt

def phase5(ckpt):
    if 'p5' in ckpt:
        print("[P5] Done, skip"); return ckpt
    print("\n" + "="*60 + "\nPHASE 5: Multi-method head-to-head\n" + "="*60)

    has_dagma = has_golem = False
    try: from dagma.linear import DagmaLinear; has_dagma = True
    except: pass
    try: from golem.golem import GolemEV; has_golem = True
    except: pass

    configs = [
        (50, 100, 0.05), (50, 100, 0.15), (50, 100, 0.30),
        (50, 200, 0.15), (100, 100, 0.15),
        (100, 200, 0.05), (100, 200, 0.15), (100, 200, 0.30),
        (100, 500, 0.15), (200, 200, 0.15), (200, 500, 0.15),
        (50, 200, 0.05), (100, 200, 0.10), (100, 200, 0.25),
        (150, 200, 0.15),
    ]
    seeds = [42, 123, 456, 789, 999]
    results = ckpt.get('p5', {})

    for d_val, n_val, K_val in configs:
        key = f'd={d_val}_n={n_val}_K={K_val}'
        if key in results: continue
        t0 = time.time(); print(f"\n  {key}...")
        sr = []
        for s in seeds:
            X, Wt, tr, ep = generate_data(n_val, d_val, 10, K_val, s)
            pic = run_pic_corr(X, tr, Wt, ep, d_val)
            ct = run_ct(X, d_val, n_epochs=2000); ct_f1 = compute_f1(np.array(ct['W']), Wt, 0.3)[0]
            Wn = run_notears(X); nf1 = compute_f1(Wn, Wt, 0.3)[0]

            entry_s = {'seed': s, 'pic_f1': pic['f1'], 'ct_f1': round(ct_f1, 4), 'note_f1': round(nf1, 4)}
            if has_dagma:
                Wd = run_dagma(X); entry_s['dagma_f1'] = round(compute_f1(Wd, Wt, 0.3)[0], 4)
            if has_golem:
                Wg = run_golem(X)
                entry_s['golem_f1'] = round(compute_f1(Wg, Wt, 0.3)[0], 4) if Wg is not None else None
            sr.append(entry_s)

        methods = {}
        for m in ['pic_f1', 'ct_f1', 'note_f1']:
            vs = [s[m] for s in sr]; methods[m] = round(float(np.mean(vs)), 4)
        if has_dagma:
            vs = [s['dagma_f1'] for s in sr]; methods['dagma_f1'] = round(float(np.mean(vs)), 4)
        if has_golem:
            vs = [s['golem_f1'] for s in sr if s['golem_f1'] is not None]
            if vs: methods['golem_f1'] = round(float(np.mean(vs)), 4)

        best = max(methods.items(), key=lambda x: x[1])
        results[key] = {'d': d_val, 'n': n_val, 'K': K_val, 'best': best[0],
                         'best_f1': best[1], 'methods': methods,
                         'time_s': round(time.time()-t0, 1), 'seeds': sr}
        ckpt['p5'] = results; save_ckpt(ckpt)
        print(f"    BEST={best[0]} ({best[1]:.3f}) [{time.time()-t0:.0f}s]")

    ckpt['p5'] = results; save_ckpt(ckpt); return ckpt


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    t_start = time.time()
    ckpt = load_ckpt()

    ckpt = phase1(ckpt)   # ~15 min
    ckpt = phase2(ckpt)   # ~2h
    ckpt = phase3(ckpt)   # ~30min
    ckpt = phase4(ckpt)   # ~2h (DAGMA+GOLEM CPU-heavy but GPU for GOLEM)
    ckpt = phase5(ckpt)   # ~1h

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"MASTER ALL DONE at {time.strftime('%H:%M:%S')} ({elapsed/3600:.1f}h)")
    print(f"Checkpoint: {CKPT} ({os.path.getsize(CKPT)} bytes)")
    print(f"{'='*60}")
