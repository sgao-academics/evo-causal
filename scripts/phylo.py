# -*- coding: utf-8 -*-
"""Minimal tree utilities: Newick parsing, pruning, Grafen branch lengths, and the
Brownian-motion covariance of an ultrametric tree.

The synthesis tree served by the Open Tree of Life carries topology only.  Branch lengths
are therefore not measurements and must not be presented as such; they are assigned by
Grafen's (1989) method, which makes every tip equidistant from the root and spaces nodes
by their number of descendant edges.  Anything computed from the tree downstream (Blomberg's
K, phylogenetic independent contrasts) is therefore conditional on that assignment, and the
experiment that uses it re-runs with equal branch lengths as a sensitivity check.
"""
import re
import numpy as np


class Node(object):
    __slots__ = ('name', 'children', 'length', 'parent')

    def __init__(self, name='', length=0.0):
        self.name = name
        self.children = []
        self.length = length
        self.parent = None


def parse_newick(s):
    """Iterative Newick parser.  Returns the root Node."""
    s = s.strip().rstrip(';')
    root = Node()
    cur = root
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == '(':
            child = Node()
            child.parent = cur
            cur.children.append(child)
            cur = child
            i += 1
        elif c == ',' :
            sib = Node()
            sib.parent = cur.parent
            cur.parent.children.append(sib)
            cur = sib
            i += 1
        elif c == ')':
            cur = cur.parent
            i += 1
        elif c == ':':
            j = i + 1
            while j < n and s[j] not in '(),:;':
                j += 1
            try:
                cur.length = float(s[i + 1:j])
            except ValueError:
                cur.length = 0.0
            i = j
        else:
            j = i
            while j < n and s[j] not in '(),:;':
                j += 1
            cur.name = s[i:j].strip()
            i = j
    return root


def leaves(root):
    out = []
    stack = [root]
    while stack:
        v = stack.pop()
        if not v.children:
            out.append(v)
        else:
            stack.extend(v.children)
    return out


def prune(root, keep):
    """Keep only the named tips in `keep`; drop everything that ends up childless."""
    keep = set(keep)
    order = []
    stack = [(root, False)]
    while stack:
        v, done = stack.pop()
        if done or not v.children:
            order.append(v)
        else:
            stack.append((v, True))
            for c in v.children:
                stack.append((c, False))
    alive = {}
    for v in order:
        if not v.children:
            alive[id(v)] = v.name in keep
        else:
            alive[id(v)] = any(alive[id(c)] for c in v.children)
            v.children = [c for c in v.children if alive[id(c)]]
    return root, [v.name for v in leaves(root)]


def heights(root):
    """Number of edges from each node down to the deepest tip below it (tips 0, root H)."""
    h = {}
    for v in postorder(root):
        h[id(v)] = 0 if not v.children else 1 + max(h[id(c)] for c in v.children)
    return h


def postorder(root):
    order = []
    stack = [(root, False)]
    while stack:
        v, done = stack.pop()
        if done or not v.children:
            order.append(v)
        else:
            stack.append((v, True))
            for c in v.children:
                stack.append((c, False))
    return order


def collapse_unary(root):
    """Splice out degree-one nodes, which a pruned synthesis tree is full of.

    They are artefacts of pruning, not ancestors of the retained sample, and they are not
    neutral: the node-count convention pays one step per node on the path, so a lineage that
    happens to keep more of them gets stretched more than its sister.  Collapsing first is
    what makes the length convention depend on the retained topology alone.
    """
    for v in list(postorder(root)):
        while len(v.children) == 1 and v.parent is not None:
            c = v.children[0]
            p = v.parent
            p.children[p.children.index(v)] = c
            c.parent = p
            v = c
    # a root with one child is the same tree; keep the child as root
    while len(root.children) == 1:
        root = root.children[0]
        root.parent = None
    return root


def grafen_lengths(root):
    """In-place Grafen (1989) node-count branch lengths.

    The edge into node v gets the INCREMENT of the normalised depth, (h_parent - h_v) / H,
    not the depth (H - h_v) / H itself.  The increment is what telescopes: summing it along
    any root-to-tip path gives (h_root - h_tip) / H = 1, so every tip is contemporaneous and
    the Brownian covariance has a constant diagonal.  Using the depth itself, as an earlier
    version of this file did, makes the root-to-tip distance depend on a tip's depth, the
    covariance is then not that of contemporaneous tips, and the whitened rows are not
    contrasts.
    """
    h = heights(root)
    H = h[id(root)] or 1
    for v in postorder(root):
        if v.parent is None:
            v.length = 0.0
        else:
            v.length = (h[id(v.parent)] - h[id(v)]) / float(H)
    return root


def power_lengths(root, rho):
    """Grafen's power convention: depth^rho, edge length the difference along the path.

    rho = 1 is the node-count convention, rho -> 0 approaches equal branch lengths and
    rho > 1 pushes the branching towards the tips.
    """
    h = heights(root)
    H = h[id(root)] or 1

    def depth(v):
        return ((H - h[id(v)]) / float(H)) ** rho

    for v in postorder(root):
        if v.parent is None:
            v.length = 0.0
        else:
            v.length = depth(v) - depth(v.parent)
    return root


def equal_lengths(root):
    order = []
    stack = [(root, False)]
    while stack:
        v, done = stack.pop()
        if done or not v.children:
            order.append(v)
        else:
            stack.append((v, True))
            for c in v.children:
                stack.append((c, False))
    for v in order:
        if v is not root:
            v.length = 1.0
    return root


def tip_depths(root):
    """Root-to-tip distance for every tip, and node depth, after lengths are set."""
    dist = {id(root): 0.0}
    stack = [root]
    while stack:
        v = stack.pop()
        for c in v.children:
            dist[id(c)] = dist[id(v)] + c.length
            stack.append(c)
    return {v.name: dist[id(v)] for v in leaves(root)}, dist


def brownian_cov(root, names):
    """Covariance matrix of Brownian motion on this tree, in the order of `names`.

    C_ij equals the root-to-MRCA distance, C_ii the root-to-tip distance.  Call after
    Grafen lengths so that the tips are contemporaneous and the diagonal is constant.
    """
    idx = {n: k for k, n in enumerate(names)}
    p = len(names)
    C = np.zeros((p, p))
    # ancestor path of each tip, with distances
    anc = {}
    for nm in names:
        v = None
        stack = [root]
        while stack:
            u = stack.pop()
            if not u.children and u.name == nm:
                v = u
                break
            stack.extend(u.children)
        if v is None:
            raise KeyError(nm)
        # walk up to the root, then back down accumulating root-to-node distance;
        # a node's own .length is the edge above it
        chain = []
        x = v
        while x is not None:
            chain.append(x)
            x = x.parent
        chain.reverse()
        path = {}
        h = 0.0
        for i, x in enumerate(chain):
            if i:
                h += x.length
            path[id(x)] = h
        anc[nm] = path
    for a, na in enumerate(names):
        for b in range(a, p):
            nb = names[b]
            common = set(anc[na]) & set(anc[nb])
            if not common:
                best = 0.0
            else:
                best = max(anc[na][k] for k in common)
            C[a, b] = C[b, a] = best
    return C
