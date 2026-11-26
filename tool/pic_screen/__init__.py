"""pic_screen — a one-call screen for phylogenetic comparative matrices.

    from pic_screen import screen
    res = screen(X, newick, labels=names)
    print(res)

The question the screen answers is narrow and, we think, the one that comes first:

    is this matrix dominated by a single direction, and if so can that direction be
    attributed to one measured variable?

If not, the matrix cannot support an edge-level causal claim, whatever estimator is used
afterwards.  The screen is a diagnostic, not a repair: subtracting the leading component is
reported, and on real life-history matrices it does not restore the matrix to the null.

No dependency beyond numpy.
"""

from .screen import screen, ScreenResult
from .tree import parse_newick, grafen_lengths, brownian_cov

__all__ = ['screen', 'ScreenResult', 'parse_newick', 'grafen_lengths', 'brownian_cov']
__version__ = '1.0.0'
