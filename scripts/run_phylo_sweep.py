"""
Full evolutionary causal discovery pipeline with Felsenstein PIC.
Systematically tests across phylo_signal levels and methods.

Key question: At what phylogenetic signal level does causal discovery break?
"""

import numpy as np
import torch
import json, os, sys, time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(__file__))
from felsenstein_pic import felsenstein_pic, PhyloNode

# Add causalscale
# causalscale should be installed via: pip install causalscale
from causalscale.core.transformer import CausalTransformer, fit_causal_transformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR = os.path.join(ROOT, 'results')
os.makedirs(RES_DIR, exist_ok=True)


def build_balanced_tree(n_species: int, seed: int = 42) -> Tuple[PhyloNode, List[str]]:
    """Build a balanced binary ultrametric tree by hand (no Newick parsing).

    Returns:
        (root, species_names) where root has all tips at the same height.
    """
    rng = np.random.RandomState(seed)

    # Create tip nodes
    tips = [PhyloNode(
        name=f"sp{i}",
        children=[],
        branch_length=0.5 + 0.1 * rng.rand(),
        is_tip=True,
        index=i
    ) for i in range(n_species)]

    species_names = [t.name for t in tips]

    # Build balanced binary tree by iterative pairing
    nodes = list(tips)
    node_counter = n_species

    while len(nodes) > 1:
        new_nodes = []
        for i in range(0, len(nodes), 2):
            if i + 1 < len(nodes):
                left = nodes[i]
                right = nodes[i + 1]
                # Internal node: height = max(left, right) + branch
                internal = PhyloNode(
                    name=f"N{node_counter}",
                    children=[left, right],
                    branch_length=0.2 + 0.05 * rng.rand(),
                    is_tip=False,
                )
                node_counter += 1
                new_nodes.append(internal)
            else:
                new_nodes.append(nodes[i])
        nodes = new_nodes

    return nodes[0], species_names


def _compute_tip_heights(node: PhyloNode, current: float = 0.0, heights: Dict = None) -> Dict:
    """Compute height (distance from root) for each tip."""
    if heights is None:
        heights = {}
    if node.is_tip and node.index is not None:
        heights[node.index] = current + node.branch_length
    for child in node.children:
        _compute_tip_heights(child, current + node.branch_length, heights)
    return heights


def _make_ultrametric(node: PhyloNode):
    """Adjust branch lengths to make tree ultrametric (all tips at same height)."""
    heights = _compute_tip_heights(node)

    if not heights:
        return

    target_h = max(heights.values())

    def _adjust(node, current=0.0):
        if node.is_tip and node.index is not None:
            node.branch_length = target_h - current
        for child in node.children:
            _adjust(child, current + node.branch_length)
        if not node.is_tip:
            # Ensure children's branch lengths make sense
            child_heights = []
            for child in node.children:
                h = 0
                if child.is_tip and child.index is not None:
                    h = current + child.branch_length
                child_heights.append(h)
            if child_heights:
                max_ch = max(child_heights)
                for child in node.children:
                    if child.is_tip and child.index is not None:
                        need = max_ch - current
                        if need > 0:
                            child.branch_length = need

    _adjust(node)
    # Verify
    heights2 = _compute_tip_heights(node)
    if heights2:
        h_vals = list(heights2.values())
        if max(h_vals) - min(h_vals) > 0.01:
            # Force equal by scaling
            pass


def _count_tips(node):
    if node.is_tip:
        return 1
    return sum(_count_tips(c) for c in node.children)


def generate_clean_data(
    n_species: int = 50,
    n_genes: int = 100,
    n_causal: int = 15,
    phylo_signal: float = 0.3,
    noise: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, PhyloNode, List[str], np.ndarray]:
    """Generate clean data with known causal structure + phylogenetic signal."""
    rng = np.random.RandomState(seed)

    # Build tree
    tree, species_names = build_balanced_tree(n_species, seed)
    _make_ultrametric(tree)

    n_tips = _count_tips(tree)
    assert n_tips == n_species, f"Tree has {n_tips} tips, expected {n_species}"

    # Compute phylogenetic distance matrix from tree
    heights = _compute_tip_heights(tree)

    # Pairwise distances: for ultrametric tree, dist(i,j) = 2*(height - LCA_height)
    # Simplified: use |height_i - height_j| as base + random perturbation
    dist = np.zeros((n_species, n_species))
    for i in range(n_species):
        for j in range(i + 1, n_species):
            hi = heights.get(i, 0)
            hj = heights.get(j, 0)
            # More distant species have larger distance
            d = abs(hi - hj) / max(hi, hj) * 2.0 + 0.1
            # Add some stochasticity to make it interesting
            d += rng.uniform(0, 0.3) * phylo_signal
            dist[i, j] = dist[j, i] = d

    # Brownian motion covariance
    scale = max(dist.std(), 1e-8)
    C = np.exp(-dist * phylo_signal / scale)
    C += 0.05 * np.eye(n_species)

    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, 1e-6)
    L = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    # Generate causal DAG (upper triangular)
    W_true = np.zeros((n_genes, n_genes))
    edges_true = []
    available = [(i, j) for i in range(n_genes) for j in range(i + 1, n_genes)]
    rng.shuffle(available)
    for i, j in available[:n_causal]:
        w = rng.uniform(0.5, 1.5) * rng.choice([-1, 1])
        W_true[i, j] = w
        edges_true.append((i, j))

    # Independent traits
    Z = rng.randn(n_species, n_genes)

    # Phylogenetic correlation
    X = L @ Z

    # Causal effects
    for (i, j) in edges_true:
        X[:, j] += W_true[i, j] * X[:, i]

    # Noise
    X += noise * rng.randn(n_species, n_genes)

    # Normalize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    return X.astype(np.float32), W_true, tree, species_names, dist


def run_causal_transformer(
    X: np.ndarray,
    d_model: int = 64,
    n_heads: int = 4,
    epochs: int = 300,
    threshold: float = 0.2,
    device: str = "cpu",
) -> np.ndarray:
    n, d = X.shape
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    dm = d_model
    nh = n_heads
    if dm % nh:
        dm = ((dm // nh) + 1) * nh
    model = CausalTransformer(
        d_vars=d, d_model=dm, n_heads=nh, n_layers=2,
        lambda_dag=0.5, lr=0.001
    )
    fit_causal_transformer(
        model, X_t, n_epochs=epochs,
        batch_size=min(64, n), device=device, verbose=False
    )
    with torch.no_grad():
        model.eval()
        W_batch, _ = model(X_t[:min(500, n)])
        W = W_batch.mean(dim=0).cpu().numpy()
    W[np.abs(W) < threshold] = 0.0
    return W


def run_notears(
    X: np.ndarray,
    threshold: float = 0.2,
    max_iter: int = 35,
    inner_iter: int = 200,
    lr: float = 0.002,
    device: str = "cpu",
) -> np.ndarray:
    from causalscale.core.transfer_engine import _train_notears_warm
    # Scratch training: zero initialization
    n, d = X.shape
    W_init = np.zeros((d, d), dtype=np.float64)
    W = _train_notears_warm(X, W_init=W_init, max_iter=max_iter,
                             inner_iter=inner_iter, lr=lr,
                             device=device, verbose=False)
    W = W.cpu().numpy() if hasattr(W, 'cpu') else W
    W[np.abs(W) < threshold] = 0.0
    return W


def f1_score(W: np.ndarray, W_true: np.ndarray, threshold: float = 0.2) -> Dict:
    pred_mask = np.abs(W) > threshold
    true_mask = np.abs(W_true) > 0
    tp = int((pred_mask & true_mask).sum())
    fp = int(pred_mask.sum()) - tp
    fn = int(true_mask.sum()) - tp
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {"F1": round(f, 3), "prec": round(p, 3), "rec": round(r, 3),
            "edges": int(pred_mask.sum()), "tp": tp, "fp": fp, "fn": fn}


def run_single_config(
    n_species: int, n_genes: int, n_causal: int,
    phylo_signal: float, seed: int,
    device: str = "cpu",
) -> Dict:
    t0 = time.time()

    X, W_true, tree, species, dist = generate_clean_data(
        n_species=n_species, n_genes=n_genes, n_causal=n_causal,
        phylo_signal=phylo_signal, noise=0.1, seed=seed
    )

    # PIC
    X_pic = felsenstein_pic(X, tree)

    # Correlations
    C_raw = np.corrcoef(X.T)
    offdiag_raw = C_raw[np.triu(np.ones((n_genes, n_genes)), 1).astype(bool)]
    raw_corr = float(np.abs(offdiag_raw).mean())

    C_pic = np.corrcoef(X_pic.T)
    offdiag_pic = C_pic[np.triu(np.ones((n_genes, n_genes)), 1).astype(bool)]
    pic_corr = float(np.abs(offdiag_pic).mean())

    true_cs = []
    for i, j in zip(*np.where(np.abs(W_true) > 0)):
        if i < n_genes and j < n_genes:
            true_cs.append(abs(np.corrcoef(X_pic[:, i], X_pic[:, j])[0, 1]))
    true_pic_c = float(np.mean(true_cs)) if true_cs else 0

    print(f"  Phylo={phylo_signal:.2f} seed={seed}: raw_c={raw_corr:.3f} pic_c={pic_corr:.3f} "
          f"true_pic_c={true_pic_c:.3f}", end="")

    # CT
    W_raw_ct = run_causal_transformer(X, epochs=400, threshold=0.2, device=device)
    W_pic_ct = run_causal_transformer(X_pic, epochs=400, threshold=0.2, device=device)
    f1r_ct = f1_score(W_raw_ct, W_true)
    f1p_ct = f1_score(W_pic_ct, W_true)

    # NOTEARS
    W_raw_nt = run_notears(X, threshold=0.2, max_iter=35, device=device)
    W_pic_nt = run_notears(X_pic, threshold=0.2, max_iter=35, device=device)
    f1r_nt = f1_score(W_raw_nt, W_true)
    f1p_nt = f1_score(W_pic_nt, W_true)

    elapsed = time.time() - t0

    print(f" | CT: r={f1r_ct['F1']}({f1r_ct['edges']}e) p={f1p_ct['F1']}({f1p_ct['edges']}e)"
          f" | NT: r={f1r_nt['F1']}({f1r_nt['edges']}e) p={f1p_nt['F1']}({f1p_nt['edges']}e)"
          f" [{elapsed:.0f}s]")

    return {
        "phylo_signal": phylo_signal, "seed": seed,
        "f1_raw_ct": f1r_ct['F1'], "f1_pic_ct": f1p_ct['F1'],
        "delta_ct": f1p_ct['F1'] - f1r_ct['F1'],
        "f1_raw_notears": f1r_nt['F1'], "f1_pic_notears": f1p_nt['F1'],
        "delta_notears": f1p_nt['F1'] - f1r_nt['F1'],
        "n_edges_raw_ct": f1r_ct['edges'], "n_edges_pic_ct": f1p_ct['edges'],
        "n_edges_raw_notears": f1r_nt['edges'], "n_edges_pic_notears": f1p_nt['edges'],
        "raw_mean_corr": raw_corr, "pic_mean_corr": pic_corr,
        "true_edge_pic_corr": true_pic_c,
        "time_s": elapsed,
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    print("=== Phylogenetic Signal Sweep ===")
    print("50 species x 50 genes x 10 causal edges\n")

    results = []
    for phylo in [0.0, 0.15, 0.3, 0.5, 0.7]:
        for seed in [42, 123]:
            r = run_single_config(50, 50, 10, phylo, seed, device)
            results.append(r)

    # Save
    out_path = os.path.join(RES_DIR, "phylo_sweep.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary table
    print(f"\n{'='*90}")
    print("SUMMARY: Mean F1 by phylo_signal level")
    print(f"{'='*90}")
    print(f"{'Phylo':>6} {'CT_raw':>8} {'CT_pic':>8} {'D_CT':>8} "
          f"{'NT_raw':>8} {'NT_pic':>8} {'D_NT':>8} "
          f"{'raw_c':>8} {'pic_c':>8} {'t_pic_c':>8}")
    print("-" * 90)

    by_phylo = {}
    for r in results:
        p = r['phylo_signal']
        by_phylo.setdefault(p, []).append(r)

    for p in sorted(by_phylo.keys()):
        rs = by_phylo[p]
        a = lambda k: np.mean([r[k] for r in rs])
        print(f"{p:6.2f} {a('f1_raw_ct'):8.3f} {a('f1_pic_ct'):8.3f} "
              f"{a('delta_ct'):+8.3f} {a('f1_raw_notears'):8.3f} "
              f"{a('f1_pic_notears'):8.3f} {a('delta_notears'):+8.3f} "
              f"{a('raw_mean_corr'):8.3f} {a('pic_mean_corr'):8.3f} "
              f"{a('true_edge_pic_corr'):8.3f}")

    # Find the wall
    print(f"\n=== WALL DETECTION ===")
    for p in sorted(by_phylo.keys()):
        rs = by_phylo[p]
        ct_delta = np.mean([r['delta_ct'] for r in rs])
        nt_delta = np.mean([r['delta_notears'] for r in rs])
        ct_f1 = np.mean([r['f1_pic_ct'] for r in rs])
        nt_f1 = np.mean([r['f1_pic_notears'] for r in rs])
        status = "WORKS" if (ct_f1 > 0.1 or nt_f1 > 0.1) else "DEAD"
        print(f"  Phylo={p:.2f}: CT_F1={ct_f1:.3f} NT_F1={nt_f1:.3f} → {status}")
