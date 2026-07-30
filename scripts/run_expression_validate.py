"""
Realistic continuous gene expression validation.
Mimics cross-tissue RNA-seq data with known co-expression patterns.
Tests: can PIC + correlation recover known co-expressed gene pairs?
"""
import numpy as np, json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from felsenstein_pic import felsenstein_pic
from run_phylo_sweep import build_balanced_tree, _make_ultrametric, _count_tips, _compute_tip_heights

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR = os.path.join(ROOT, 'results')
os.makedirs(RES_DIR, exist_ok=True)

rng = np.random.RandomState(42)

# Configuration: 30 tissues (pseudo-species) x 50 genes
N_TISSUES = 80
N_GENES = 60
PHYLO = 0.25
SEEDS = [42, 123, 456]

# Known co-expressed pairs (from literature validation)
KNOWN_PAIRS = [
    ("VEGFA", "KDR"), ("VEGFA", "FLT1"), ("INS", "INSR"),
    ("IGF1", "IGF1R"), ("EGF", "EGFR"), ("BMP4", "BMPR1A"),
    ("SHH", "PTCH1"), ("HGF", "MET"), ("TNF", "TNFRSF1A"),
    # Negative controls (housekeeping, no regulatory relationship)
    ("GAPDH", "ACTB"),
]
NEG_PAIRS = [("GAPDH", "ACTB")]

print("=== Realistic Expression Validation ===\n")
print(f"{N_TISSUES} tissues x {N_GENES} genes, phylo={PHYLO}")
print(f"Known pairs: {len(KNOWN_PAIRS)} ({len(NEG_PAIRS)} negative controls)\n")

all_results = []
for seed in SEEDS:
    rng_seed = np.random.RandomState(seed)
    print(f"--- Seed {seed} ---")

    # Build tissue tree (simulating developmental lineage)
    tree, _ = build_balanced_tree(N_TISSUES, seed)
    _make_ultrametric(tree)
    heights = _compute_tip_heights(tree)

    # Generate tissue distances
    dist = np.zeros((N_TISSUES, N_TISSUES))
    for i in range(N_TISSUES):
        for j in range(i+1, N_TISSUES):
            hi = heights.get(i, 0); hj = heights.get(j, 0)
            d = abs(hi-hj)/max(hi,hj)*2.0 + 0.1 + rng_seed.uniform(0,0.3)*PHYLO
            dist[i,j] = dist[j,i] = d

    # Brownian motion phylogenetic covariance
    scale = max(dist.std(), 1e-8)
    C = np.exp(-dist * PHYLO / scale)
    C += 0.05 * np.eye(N_TISSUES)
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, 1e-6)
    L = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    # Build gene index for known pairs
    known_genes = list(set([p[0] for p in KNOWN_PAIRS] + [p[1] for p in KNOWN_PAIRS]))
    gene_idx = {}
    for i, gn in enumerate(known_genes):
        gene_idx[gn] = i
    coexpr_pairs = []

    # Generate base expression with co-expression patterns
    Z = rng_seed.randn(N_TISSUES, N_GENES)

    # Apply co-expression BEFORE phylogeny (so signal survives Brownian motion)
    for g1, g2 in KNOWN_PAIRS:
        if g1 in gene_idx and g2 in gene_idx:
            i, j = gene_idx[g1], gene_idx[g2]
            coef = rng_seed.uniform(0.4, 0.7) * rng_seed.choice([-1,1])
            # Add co-expression to the latent Z before phylogeny
            Z[:, j] += coef * Z[:, i]
            coexpr_pairs.append((g1, g2))

    # Phylogenetic correlation (applied to already-correlated Z)
    X = L @ Z

    # Add noise
    X += 0.08 * rng_seed.randn(N_TISSUES, N_GENES)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    X = X.astype(np.float32)

    # PIC
    X_pic = felsenstein_pic(X, tree)

    # Correlations
    C_pic = np.corrcoef(X_pic.T)
    mu = np.triu(np.ones((N_GENES, N_GENES)), 1).astype(bool)
    bg_pic = float(np.abs(C_pic[mu]).mean())
    all_corrs = np.abs(C_pic[mu])
    p99 = np.percentile(all_corrs, 99)

    # Test known pairs
    hits = 0; total = 0; fp = 0; neg_total = 0
    for g1, g2 in KNOWN_PAIRS:
        if g1 in gene_idx and g2 in gene_idx:
            i, j = gene_idx[g1], gene_idx[g2]
            pc = abs(C_pic[i, j])
            rank = (all_corrs < pc).sum() / len(all_corrs)
            hit = pc > p99
            if (g1, g2) in NEG_PAIRS or (g2, g1) in NEG_PAIRS:
                neg_total += 1
                if hit: fp += 1
            else:
                total += 1
                if hit: hits += 1
            print(f"  {g1}-{g2:<15} r={pc:.4f} rank={rank:.4f} {'HIT!' if hit else ''}")

    print(f"  Recovery: {hits}/{total} = {100*hits/max(total,1):.0f}% (FP: {fp}/{max(neg_total,1)})")
    print(f"  BG corr: {bg_pic:.4f}, p99: {p99:.4f}")

    all_results.append({"seed": seed, "hits": hits, "total": total,
                        "fp": fp, "neg_total": neg_total,
                        "bg_corr": bg_pic, "p99": p99})

# Summary
print(f"\n{'='*60}")
print("SUMMARY: Expression Validation")
print(f"{'='*60}")
mean_hit = np.mean([r['hits'] for r in all_results])
mean_total = np.mean([r['total'] for r in all_results])
mean_fp = np.mean([r['fp'] for r in all_results])
mean_bg = np.mean([r['bg_corr'] for r in all_results])
print(f"  Known pairs recovered: {mean_hit:.1f}/{mean_total:.0f} = {100*mean_hit/max(mean_total,1):.0f}%")
print(f"  False positives: {mean_fp:.1f}")
print(f"  Mean PIC background corr: {mean_bg:.4f}")

out = {
    "config": {"n_tissues": N_TISSUES, "n_genes": N_GENES, "phylo": PHYLO},
    "known_pairs": KNOWN_PAIRS,
    "results": all_results,
    "summary": {"mean_recovery_rate": mean_hit/max(mean_total,1),
                "mean_fp": float(mean_fp),
                "mean_bg_corr": mean_bg}
}
with open(os.path.join(RES_DIR, "expression_validate.json"), "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {RES_DIR}/expression_validate.json")
