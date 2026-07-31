# evo-causal — Replication Package

Replication package for **"The Phylogenetic Wall: Why Five Causal Discovery Methods Fail on Cross-Species Data."**

## Quick Start (Figures Only, ~5 seconds)

```bash
pip install numpy matplotlib scipy
python run_all.py
```

This verifies the PIC implementation, validates the checkpoint, and generates all three figures from the pre-computed results.

## Full Replication (Requires GPU, ~10 hours)

The Causal Transformer experiments use [**causalscale**](https://github.com/sgao-academics/causalscale) — our open-source engine for scaling gradient-based causal discovery to genome-wide resolution. If you find this work useful, please check it out.

```bash
pip install causalscale dagma-linear
python scripts/run_master_all.py          # Phase 1–5 experiments
python scripts/run_nonlinear_wall_penetration.py  # Nonlinear + penetration
python run_all.py                         # Generate figures
```

**Hardware**: NVIDIA RTX 5060 (8GB) or equivalent. CPU-only fallback works for some phases but is slow.

## Repository Structure

```
evo-causal/
├── run_all.py                              # One-command replication
├── data/                                   # Real gene ortholog data
│   ├── ensembl_real_*.json, *.npy
├── results/                                # Pre-computed checkpoints
│   ├── master_all.json                     # Main: 40 phase-diagram + all methods
│   ├── nonlinear_wall.json                 # Nonlinear robustness + penetration scan
│   └── blomberg_k_mapping.json             # Blomberg's K calibration
├── scripts/
│   ├── felsenstein_pic.py                  # Core PIC algorithm
│   ├── run_master_all.py                   # Full experiment pipeline
│   ├── run_bonus_fill.py                   # Supplementary experiments
│   ├── run_dagma_golem_full.py             # DAGMA + GOLEM sweep
│   ├── run_ensembl_real.py                 # Real gene data
│   ├── run_expression_validate.py          # Expression validation
│   ├── run_high_power.py                   # High-power CT experiments
│   ├── run_multimethod_addon.py            # Multi-method add-ons
│   ├── run_nonlinear_wall_penetration.py   # Nonlinear + wall penetration
│   ├── run_overnight_massive.py            # Overnight sweep scripts
│   ├── run_phylo_sweep.py                  # Phylogenetic parameter sweep
│   ├── run_reviewer_experiments.py         # Reviewer-requested experiments
│   ├── _compute_blomberg_k.py              # Blomberg's K calibration
│   └── figures/
│       ├── gen_fig1_phylogenetic_wall.py   # Fig 1: Phase diagram heatmap
│       ├── gen_fig2_ct_mechanism.py        # Fig 2: CT failure mechanism
│       └── gen_fig3_method_landscape.py    # Fig 3: Method comparison
└── figures/                               # Generated output (created by run_all.py)
```

## Paper

The manuscript is included as a separate LaTeX source package (`LaTeX_Source.zip`). The PDF is also available at the same repository root.

## License

MIT

## Author

Shuaidong Gao (ORCID: [0009-0004-5641-3581](https://orcid.org/0009-0004-5641-3581))
Chongqing Institute of Foreign Studies
