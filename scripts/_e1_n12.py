# -*- coding: utf-8 -*-
"""E1 at n = 12, the size Figure 1 actually draws.

The 40-tip run in e1_whitening.json is the robustness check. Figure 1 draws an 8-tip tree
and reported its own numbers (0.044 for OU), while the text reported the 40-tip number
(0.072). Two different trees quoted as one result is exactly the kind of mismatch a careful
reader finds. Running the same identity at the size the figure uses removes the mismatch
instead of explaining it away.

n = 12 is chosen because it is small enough to draw every tip legibly at 159 mm width and
large enough that the matrix is not degenerate.
"""
import os, sys, json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))



from run_revision import build_ultrametric_tree, mrca_heights, cov_brownian, cov_ou
from felsenstein_pic import felsenstein_pic

N = 12
SEED = 11

tree, T = build_ultrametric_tree(N, seed=SEED)
H, _, _, _ = mrca_heights(tree, N)

# PIC map A, recovered from unit vectors; a fresh tree per column because the recursion
# writes adjusted branch lengths back onto the nodes
A = np.zeros((N - 1, N))
for j in range(N):
    tr, _ = build_ultrametric_tree(N, seed=SEED)
    e = np.zeros((N, 1)); e[j] = 1.0
    A[:, j] = felsenstein_pic(e, tr)[:, 0]

out = {'n': N, 'tree_seed': SEED, 'tree_height': round(float(T), 4)}
for tag, ao in [('BM', 0.0), ('OU_alpha3', 3.0), ('OU_alpha10', 10.0)]:
    C = cov_brownian(H) if ao == 0 else cov_ou(H, ao)
    np.fill_diagonal(C, 1.0)
    M = A @ C @ A.T
    Mn = M / np.mean(np.diag(M))
    off = np.abs(Mn - np.diag(np.diag(Mn)))
    out[tag] = {'mean_offdiag': round(float(off.mean()), 5),
                'max_offdiag': round(float(off.max()), 5)}
    print(f'  n={N} {tag:10s} mean|off|={off.mean():.5f}  max|off|={off.max():.5f}')

# what the figure needs: the matrix for BM and for OU(3), plus the covariance V
Cs, Ms = {}, {}
for tag, ao in [('BM', 0.0), ('OU3', 3.0)]:
    C = cov_brownian(H) if ao == 0 else cov_ou(H, ao)
    np.fill_diagonal(C, 1.0)
    Cs[tag] = C
    M = A @ C @ A.T
    Ms[tag] = M / np.mean(np.diag(M))

np.savez(os.path.join(HERE, '_fig1_arrays.npz'),
         V=Cs['BM'], A=A, M_BM=Ms['BM'], M_OU=Ms['OU3'],
         H=H, total_height=np.array([T]))

p = os.path.join(os.path.dirname(HERE), 'results', 'e1_whitening_n12.json')
json.dump(out, open(p, 'w', encoding='utf-8'), indent=1)
print('saved', p)
