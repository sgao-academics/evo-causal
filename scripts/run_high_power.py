"""
High-powered validation: 100 species x 50 genes, CT with 1200 epochs,
plus a direct correlation-based causal discovery as baseline.
"""

import numpy as np
import torch
import json, os, sys, time

sys.path.insert(0, os.path.dirname(__file__))
from felsenstein_pic import felsenstein_pic
from run_phylo_sweep import (
    build_balanced_tree, _make_ultrametric, _count_tips, _compute_tip_heights,
    generate_clean_data, run_causal_transformer, f1_score
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR = os.path.join(ROOT, 'results')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

# ================================================================
# Config: More species = more PIC contrasts = more power
# ================================================================
N_SPECIES = 100
N_GENES = 50
N_CAUSAL = 10
PHYLO = 0.15
SEEDS = [42, 123, 456]
EPOCHS = 1200

print(f"=== HIGH-POWER VALIDATION ===")
print(f"{N_SPECIES} species x {N_GENES} genes x {N_CAUSAL} causal, phylo={PHYLO}, epochs={EPOCHS}\n")

results = []

for seed in SEEDS:
    print(f"--- Seed {seed} ---")
    t0 = time.time()

    X, W_true, tree, species, dist = generate_clean_data(
        n_species=N_SPECIES, n_genes=N_GENES, n_causal=N_CAUSAL,
        phylo_signal=PHYLO, noise=0.1, seed=seed
    )

    # PIC
    X_pic = felsenstein_pic(X, tree)
    print(f"  Data: {X.shape}, PIC: {X_pic.shape}")

    # Correlations
    C_pic = np.corrcoef(X_pic.T)
    offdiag = C_pic[np.triu(np.ones((N_GENES, N_GENES)), 1).astype(bool)]
    bg_corr = float(np.abs(offdiag).mean())

    true_edges = list(zip(*np.where(np.abs(W_true) > 0)))
    true_cs = [abs(np.corrcoef(X_pic[:, i], X_pic[:, j])[0, 1]) for i, j in true_edges]
    true_c = float(np.mean(true_cs))
    print(f"  PIC bg_corr={bg_corr:.3f}, true_corr={true_c:.3f}, ratio={true_c/bg_corr:.1f}x")

    # ---- Method 1: Direct correlation threshold (simple baseline) ----
    # Select edges where |corr| exceeds a percentile-based threshold
    # Use the 95th percentile of absolute correlations
    abs_corr = np.abs(offdiag)
    p95 = np.percentile(abs_corr, 95)
    p99 = np.percentile(abs_corr, 99)

    for pct_name, thr in [("p95", p95), ("p99", p99), ("p99.5", np.percentile(abs_corr, 99.5))]:
        W_corr = np.zeros((N_GENES, N_GENES))
        for i in range(N_GENES):
            for j in range(i + 1, N_GENES):
                c = abs(np.corrcoef(X_pic[:, i], X_pic[:, j])[0, 1])
                if c > thr:
                    W_corr[i, j] = 1.0
        f1c = f1_score(W_corr, W_true, 0.5)
        print(f"  Corr {pct_name}(>{thr:.3f}): F1={f1c['F1']} ({f1c['tp']}tp/{f1c['edges']}edges)")

    # ---- Method 2: CT with many epochs ----
    W_pic_ct = run_causal_transformer(
        X_pic, d_model=64, n_heads=4,
        epochs=EPOCHS, threshold=0.15,  # lower threshold
        device=device
    )
    f1_ct = f1_score(W_pic_ct, W_true, 0.15)
    print(f"  CT(1200ep):  F1={f1_ct['F1']} ({f1_ct['tp']}tp/{f1_ct['edges']}edges)")

    # ---- Method 3: CT on raw ----
    W_raw_ct = run_causal_transformer(
        X, d_model=64, n_heads=4,
        epochs=EPOCHS, threshold=0.15, device=device
    )
    f1r_ct = f1_score(W_raw_ct, W_true, 0.15)
    print(f"  CT_raw(1200ep): F1={f1r_ct['F1']} ({f1r_ct['tp']}tp/{f1r_ct['edges']}edges)")

    elapsed = time.time() - t0
    print(f"  [{elapsed:.0f}s]")

    results.append({
        "seed": seed,
        "bg_corr": bg_corr, "true_corr": true_c,
        "corr_p95_f1": f1_score(W_corr, W_true).get("F1", 0),
        "ct_pic_f1": f1_ct['F1'], "ct_pic_edges": f1_ct['edges'],
        "ct_raw_f1": f1r_ct['F1'], "ct_raw_edges": f1r_ct['edges'],
        "time_s": elapsed,
    })

# Summary
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"  True edges in PIC space: corr={np.mean([r['true_corr'] for r in results]):.3f}")
print(f"  Background: corr={np.mean([r['bg_corr'] for r in results]):.3f}")
print(f"  CT_pic mean F1: {np.mean([r['ct_pic_f1'] for r in results]):.3f}")
print(f"  CT_raw mean F1: {np.mean([r['ct_raw_f1'] for r in results]):.3f}")

out = os.path.join(RES_DIR, "high_power.json")
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out}")
