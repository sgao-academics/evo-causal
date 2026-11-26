# -*- coding: utf-8 -*-
"""`python -m pic_screen` runs the self-check; `python -m pic_screen FILE TREE` screens a matrix;
`python -m pic_screen --example` screens the bundled life-history matrix."""
import os
import sys

if len(sys.argv) > 1 and sys.argv[1].endswith('.py'):
    from .selfcheck import main
    sys.exit(main())

if len(sys.argv) == 1:
    from .selfcheck import main
    sys.exit(main())

if sys.argv[1] in ('--example', '-e'):
    ex = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'example')
    sys.argv = [sys.argv[0], os.path.join(ex, 'life_history.tsv'),
                os.path.join(ex, 'life_history.nwk')]

# screen a delimited matrix: first row headers, first column species labels, then a newick file
import csv, io, numpy as np
from .screen import screen

path = sys.argv[1]
tree_path = sys.argv[2] if len(sys.argv) > 2 else None
delim = '\t' if path.endswith(('.tsv', '.txt')) else ','
with io.open(path, encoding='utf-8') as f:
    rows = list(csv.reader(f, delimiter=delim))
traits = rows[0][1:]
species = [r[0] for r in rows[1:]]
X = np.array([[float(x) for x in r[1:]] for r in rows[1:]], float)
nw = io.open(tree_path, encoding='utf-8').read() if tree_path else None
res = screen(X, nw, species=species, labels=traits)
print(res)
print()
for k, v in res.share.items():
    print('  %-34s %.3f' % (k, v))
