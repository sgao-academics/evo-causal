# Phylogenetic signal is not phylogenetic confounding

Replication package for the manuscript *Phylogenetic signal is not phylogenetic
confounding: what actually limits causal discovery across species*.

The package contains the code that produced every number in the paper and every
result file those numbers were read from. It does not contain the manuscript or
the figures: both are generated from what is here, and a second copy would only
be able to fall out of date.

## Requirements

```
pip install -r requirements.txt
```

Tested with Python 3.12.13, numpy 2.4.3, scipy 1.17.1, matplotlib 3.10.9 and
causal-learn 0.1.4.8. `causal-learn` is pinned because the PC and FCI entry
points changed between releases.

No GPU, no PyTorch, no external packaged causal-discovery library. The NOTEARS
solver is part of this repository (see *Implementation notes*).

## Layout

```
scripts/
  notears_impl.py            linear NOTEARS, restated from the published algorithm
  felsenstein_pic.py         recursive Felsenstein independent contrasts
  methods_unified.py         one interface to NOTEARS, PC, FCI and PPA
  run_revision.py            E1-E4: whitening identity, Blomberg's K, PIC vs raw
  run_revision2.py           E5-E9: dose-response, PC, path analysis, FCI, sample size
  _e1_n12.py                 E1 at the 12-tip tree Figure 1 draws
  _e10_threshold.py          E10: fixed-percentile vs top-k
  _e11_families.py           E11: four families on one grid
  _e11_auroc.py              E11: threshold-free AUROC for the same grid
  _e12_spearman.py           E12: Pearson, Spearman, O(D)GC
  _e13_odgc.py               E13: O(D)GC
  make_fig1.py               Figure 1
  make_fig2_fig4_fig5.py     Figures 2, 4 and 5
  make_fig3.py               Figure 3
  palette.py                 the single source of colour for every figure
  _fig1_arrays.npz           the matrices Figure 1 draws, written by _e1_n12.py
results/
  e1_whitening.json          E1, 40-tip tree
  e1_whitening_n12.json      E1, 12-tip tree (the one Figure 1 uses)
  e3_blomberg.json           E3, measured K under each process
  e4_methods.json            E4, NOTEARS on contrasts vs on raw values
  e5_k_axis.json             E5, dose-response on measured K
  e6_pc.json                 E6, PC at full conditioning depth
  e7_ppa.json                E7, path analysis with and without the true model
  e8_fci.json                E8, FCI and its endpoint marks
  e9_nsense.json             E9, sample size at two loadings
  e10_threshold.json         E10, fixed percentile vs top-k
  e11_families.json          E11, four families on one axis
  e11_auroc.json             E11, AUROC for the same grid
  e12_spearman.json          E12, rank transform
  e13_odgc.json              E13, O(D)GC
run_all.py                   runs everything and checks the result files
```

## Quick start

```
python run_all.py            # full grid; several hours on one CPU core
python run_all.py --fast     # one seed, reduced grid; a few minutes
python run_all.py --check    # verify the shipped result files only
```

`run_all.py --check` re-reads every result file, confirms the keys the
manuscript relies on are present, and prints the values that the text quotes.
It needs no computation and finishes in seconds, which makes it the quickest way
to see whether the archive and the paper agree.

## Reproducing the figures

```
python scripts/_e1_n12.py                  # writes _fig1_arrays.npz
python scripts/make_fig1.py
python scripts/make_fig2_fig4_fig5.py
python scripts/make_fig3.py
```

Figures are written to `figures/`, which is not tracked. Each script reads only
files in `results/`, so a figure cannot disagree with the tables.

## No input data files

Every dataset is a deterministic function of `(n, d, k, lambda, noise, seed)`.
`simulate()` in `run_revision.py` takes those six arguments and returns the same
array every time, and the tree is built by `build_ultrametric_tree(n, seed)` the
same way. There is therefore no data file to distribute: to obtain any dataset
the paper uses, call `simulate()` with the arguments printed in the
corresponding script.

The one exception is `_fig1_arrays.npz`, which is shipped because
`make_fig1.py` draws matrices rather than datasets, and regenerating it is
cheaper than re-deriving the PIC map each time the figure changes.

## Implementation notes

Three things in this package are worth stating explicitly, because the
manuscript makes claims about them.

**NOTEARS is restated, not wrapped.** `scripts/notears_impl.py` implements the
augmented Lagrangian of Zheng, Aragam, Ravikumar and Xing (2018) directly: an
L-BFGS-B inner solve, `rho` multiplied by ten whenever the constraint fails to
fall to a quarter of its previous value, dual ascent on `alpha`, and an outer
stopping tolerance of `h_tol = 1e-8`. It is not a call into the authors'
repository. That is deliberate: it is the only way the acyclicity residual can
be audited, and the residual is reported with every run (`h_final_mean` in
`results/e4_methods.json`, and in the text) at `2.2e-9` to `6.6e-9`, i.e. the
constraint holds to machine precision rather than merely to tolerance.

The solver is set up with `w_threshold = 0.0`. Every method in the paper is
scored under one common decision rule — keep the `k = |E_true|` largest scores —
and a threshold applied inside NOTEARS would give it a rule no other method has.
`notears_impl.notears_linear` still exposes `w_threshold`, so the `|W| > 0.3`
sensitivity reported in the manuscript can be reproduced by passing 0.3.

**PC and FCI come from one package at one significance level.**
`methods_unified.py` calls `causallearn.search.ConstraintBased.PC.pc` and
`...FCI.fci` at `ALPHA = 0.05`, and PC runs at full conditioning depth. Both
matter: at a fixed alpha, a shallower PC search and a different alpha for FCI
would turn a methodological comparison into a comparison of two tuning choices.

**PIC is implemented recursively.** `felsenstein_pic.py` standardises each
contrast by the square root of the summed branch lengths and computes the
ancestral value as the branch-length-weighted mean. The function writes adjusted
branch lengths back onto the nodes it is given, so it consumes the tree it is
handed; callers that need the tree afterwards must pass a fresh copy. The guard
`_pic_consumed` makes a second call on the same tree raise instead of returning
a silently wrong result.

## Reproducibility checks performed

* `_e1_n12.py` rerun against the archive: the regenerated
  `e1_whitening_n12.json` is identical to the shipped file, field by field
  (BM 0.000, OU(3) 0.083, OU(10) 0.113).
* `_e11_auroc.py` rerun on the full six-loading grid: all twelve AUROC values
  (NOTEARS and the correlation rule at six loadings) match the archive exactly,
  to the third decimal, at 0.0000 absolute difference.
* `make_fig3.py` rerun: the PNG is byte-identical to the shipped figure.
* `notears_impl.notears_linear` compared against the implementation the numbers
  were originally produced with: identical matrices, `max|dW| = 0.0`, at two
  penalty weights and at both threshold settings.

## Version note

An earlier version of this study, under a different title and with a different
central claim, is in the git history. The present package describes the
manuscript as submitted; the earlier analysis is not distributed, because its
data-generating model and its conclusions differ from the ones here.

## Contact

Shuaidong Gao — Chongqing Institute of Foreign Studies, Chongqing 401420, China
<gaoshuaidong@stu.cqifs.edu.cn> — ORCID 0009-0004-5641-3581
