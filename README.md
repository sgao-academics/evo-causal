# evo_causal — Phylogenetically-Aware Causal Discovery on Cross-Species Data

**Question:** Can causal discovery recover regulatory relationships from evolutionary trait data, after removing phylogenetic signal?

**Pipeline:** Phylogenetic Independent Contrasts (PIC) → Causal Transformer / NOTEARS → Co-evolution Validation.

## Quick Start

```bash
cd scripts
python pipeline.py
```

## Project Structure

```
evo_causal/
├── data/           # Raw data: gene presence/absence, trait matrices, Newick trees
├── scripts/        # Causal discovery & PIC processing
│   └── pipeline.py # Main PIC + Causal Transformer pipeline
├── results/        # Learned graphs (.npy), validation reports (.json)
├── notebooks/      # Exploratory analysis
└── paper/          # Manuscript
```

## Data Sources (Planned)

| Source | Content | Size |
|:--|:--|:--|
| Ensembl Compara | 100+ vertebrate gene presence/absence | ~5000 genes |
| TimeTree | Divergence times for tree building | All species pairs |
| NCBI Taxonomy | Species tree topology | All vertebrates |
| OMA Orthology | Orthologous groups | ~1000 groups |

## Validation

- Known ligand-receptor co-evolution pairs (VEGFA-KDR, INS-INSR, etc.)
- Mitochondrial-nuclear compensatory mutations
- Ribosomal subunit co-evolution

## Author

Shuaidong Gao (ORCID: 0009-0004-5641-3581)
Chongqing Institute of Foreign Studies
