# pic_screen

A one-call check for whether a comparative matrix can support an edge-level claim at all.

```python
from pic_screen import screen
res = screen(X, newick, species=names, labels=traits)
print(res)
```

```
<ScreenResult n=337 d=6  r1=3.29 (threshold 1.33, p=0.00025)  top=adult body mass (g) 0.27  deflated r1=2.04 (p=0.00)>
  one direction dominates and subtracting it does not clear the excess: the direction is a
  property of the trait set, and edge-level claims are not identified
```

## What it answers

Two questions, in this order.

1. **Is one direction doing the work?** `r1` is the leading eigenvalue of the contrast
   correlation matrix over the mean of the rest. Under independent Brownian traits the whitened
   data are independent standard normals, so the null distribution of `r1` depends on `(n, d)`
   only and is simulated exactly. No tree enters it, and no branch-length convention can bias it.
2. **Can that direction be attributed to one measured variable?** `share` gives the squared
   loadings. An `r1` of 3.5 spread over six traits and an `r1` of 3.5 sitting on one column mean
   opposite things, and the ratio alone cannot tell them apart.

`deflated_r1` reports the same ratio after subtracting the leading component, with its own
p-value. A screen that fires and a repair that works are different claims; this separates them.
On real life-history matrices the screen fires and the repair fails.

## Requirements and cost

`numpy` only. A screen with 4,000 null draws takes about a second at `d = 6` and a few seconds
at `d = 12`. The null draws are the whole cost and can be cached by passing a precomputed
covariance matrix instead of a tree.

## Input

| argument | meaning |
|:--|:--|
| `X` | `(n, d)` values, **rows are species, columns are traits**. Columns whose max/min ratio exceeds twenty are log10-transformed first. |
| `newick` | tree as a Newick string; pruning to `species` happens here, and degree-one ancestors are collapsed |
| `species` | tip labels in the row order of `X` |
| `labels` | trait names, one per column |
| `tree` | `'grafen'` (default), `'equal'`, or an `(n, n)` covariance matrix |
| `rho` | exponent of Grafen's power convention; 1.0 is the node-count convention |

## Branch lengths

Grafen (1989) node-count lengths place every tip the same distance from the root, which is what
makes a Brownian covariance matrix have a constant diagonal:

```
height h_v  = edges from the deepest tip below v up to v     (tip 0, root H)
depth  d_v  = (H - h_v) / H                                  (root 0, tip 1)
length(e_v) = d_v - d_parent                                 <- the increment
```

Writing `(H - h_v)/H` as the edge length is a frequent slip. It sends tips to `(H+1)/2` instead
of 1, so they stop being contemporaneous and the whitened data are not contrasts. `screen()`
checks the root-to-tip distances and raises rather than proceeding. `tree.check_ultrametric()`
shows the check in isolation; `python -m pic_screen` runs a self-test that includes it.

## Example

```bash
cd tool
python -m pic_screen                              # self-test, 8 checks
python -m pic_screen example/life_history.tsv example/life_history.nwk
```

`example/` carries the 337 mammal species, six life-history traits and the pruned synthesis
tree behind Figure S5 of the manuscript. The numbers it returns are the numbers in that figure.

## What it does not do

It does not estimate a graph, and it is not a substitute for one. It tells you whether the
matrix you are about to estimate a graph from has a single direction that no measured variable
accounts for. If it returns `no single direction dominates`, estimate away. If it returns the
other verdicts, no estimator run afterwards will fix the matrix, because the problem is in the
matrix.
