# -*- coding: utf-8 -*-
"""Newick parsing and Grafen branch lengths, self-contained (no numpy needed to parse).

Grafen (1989) node-count lengths
--------------------------------
Every tip is placed the same distance from the root, which is what makes the covariance
matrix of Brownian motion have a constant diagonal.

    height h_v  = edges from the deepest tip below v up to v        (tip 0, root H)
    depth  d_v  = (H - h_v) / H                                     (root 0, tip 1)
    length(e_v) = d_v - d_parent

The last line is the increment, not the depth.  Writing (H - h_v)/H as the edge length is a
common slip: it makes the root the only node at distance 0, but sends tips to (H+1)/2 rather
than to 1, so tips are no longer contemporaneous and the "contrasts" are not contrasts.
`check_ultrametric` exists to make that slip impossible to miss.
"""


class Node(object):
    __slots__ = ('name', 'length', 'children', 'parent')

    def __init__(self, name=None, length=None):
        self.name = name
        self.length = length
        self.children = []
        self.parent = None

    def add(self, child):
        child.parent = self
        self.children.append(child)

    def __repr__(self):
        return 'Node(%s, %d children)' % (self.name, len(self.children))


def parse_newick(text):
    """Parse a Newick string with optional branch lengths and quoted labels.

    Iterative, so a deep tree does not hit the recursion limit.  The one rule that is easy to
    get wrong: a label that follows ')' belongs to the node that was just closed, while a label
    that follows '(' belongs to a new tip.  Getting that backwards silently turns tips into
    internal labels and the tip count comes out short.
    """
    s = text.strip()
    if s.endswith(';'):
        s = s[:-1]
    n = len(s)
    root = None
    stack = []
    i = 0
    pending = None          # a node closed by ')' that may still receive a label

    def read_label(j):
        # Synthesised trees carry stray single quotes as node labels (Open Tree does).  A quote
        # is only treated as a delimiter when a closing one exists; otherwise it is text.  The
        # naive version swallows the rest of the file on such a tree and the tip count comes
        # out short, which is a silent failure.
        if j < n and s[j] == "'":
            k = s.find("'", j + 1)
            if k > 0:
                return s[j + 1:k], k + 1
        start = j
        while j < n and s[j] not in ',():':
            j += 1
        return s[start:j].strip(), j

    def read_length(j):
        if j < n and s[j] == ':':
            k = j + 1
            while k < n and s[k] not in ',();':
                k += 1
            try:
                return float(s[j + 1:k]), k
            except ValueError:
                return None, k
        return None, j

    while i < n:
        c = s[i]
        if c in ' \t\r\n':
            i += 1
            continue
        if c == '(':
            v = Node()
            if stack:
                stack[-1].add(v)
            else:
                root = v
            stack.append(v)
            closed = None
            i += 1
        elif c == ',':
            closed = None
            i += 1
        elif c == ')':
            closed = stack.pop()
            i += 1
        elif c == ':':
            ln, i = read_length(i)
            target = closed
            if target is None and stack:
                target = stack[-1].children[-1] if stack[-1].children else stack[-1]
            if target is not None and ln is not None:
                target.length = ln
            closed = None
        else:
            name, i = read_label(i)
            if closed is not None:
                closed.name = name or None          # label after ')' names the closed node
            else:
                v = Node(name=name or None)
                if stack:
                    stack[-1].add(v)
                else:
                    root = v
                    stack.append(v)
    return root


def leaves(root):
    out, stack = [], [root]
    while stack:
        v = stack.pop()
        if v.children:
            stack.extend(v.children)
        else:
            out.append(v)
    return out


def postorder(root):
    order, stack = [], [(root, False)]
    while stack:
        v, done = stack.pop()
        if done or not v.children:
            order.append(v)
        else:
            stack.append((v, True))
            for c in v.children:
                stack.append((c, False))
    return order


def prune(root, keep):
    """Drop the tips not in `keep`, collapsing the unary nodes that this leaves behind."""
    have = set(l.name for l in leaves(root))
    missing = [k for k in keep if k not in have]
    if missing:
        raise KeyError('not in tree: %s%s' % (missing[:5], ' ...' if len(missing) > 5 else ''))
    def walk(v):
        if not v.children:
            return v if v.name in keep else None
        kids = [walk(c) for c in v.children]
        kids = [c for c in kids if c is not None]
        if not kids:
            return None
        v.children = kids
        for c in kids:
            c.parent = v
        return v
    out = walk(root)
    if out is None:
        raise ValueError('no tips left after pruning')
    # collapse single-child nodes, except at the root
    while True:
        changed = False
        for v in postorder(out):
            if v.children and len(v.children) == 1 and v.parent is not None:
                c = v.children[0]
                p = v.parent
                p.children = [c if x is v else x for x in p.children]
                c.parent = p
                changed = True
                break
        if not changed:
            break
    return out


def _heights(root):
    h = {}
    for v in postorder(root):
        h[id(v)] = 0 if not v.children else 1 + max(h[id(c)] for c in v.children)
    return h


def grafen_lengths(root, rho=1.0):
    """Set Grafen lengths in place (rho = 1 is the node-count convention)."""
    h = _heights(root)
    H = h[id(root)] or 1

    def depth(v):
        d = (H - h[id(v)]) / float(H)
        return d if rho == 1.0 else d ** rho

    for v in postorder(root):
        v.length = 0.0 if v.parent is None else depth(v) - depth(v.parent)
    return root


def equal_lengths(root):
    for v in postorder(root):
        v.length = 0.0 if v.parent is None else 1.0
    return root


def tip_depths(root):
    dist = {id(root): 0.0}
    stack = [root]
    while stack:
        v = stack.pop()
        for c in v.children:
            dist[id(c)] = dist[id(v)] + (c.length or 0.0)
            stack.append(c)
    return dict((l.name, dist[id(l)]) for l in leaves(root))


def check_ultrametric(root, tol=1e-8):
    """Return (min, max) root-to-tip distance.  They must agree for contrasts to be contrasts."""
    d = tip_depths(root)
    v = list(d.values())
    return min(v), max(v)


def brownian_cov(root, names):
    """Covariance of Brownian motion on this tree, in the order of `names`.

    C_ij is the root-to-MRCA distance and C_ii the root-to-tip distance.  Needs numpy.
    """
    import numpy as np
    want = set(names)
    anc = {}
    stack = [(root, [root])]
    while stack:
        v, path = stack.pop()
        if v.children:
            for c in v.children:
                stack.append((c, path + [c]))
        elif v.name in want:
            anc[v.name] = path
    miss = [n for n in names if n not in anc]
    if miss:
        raise KeyError('not in tree: %s' % miss[:5])
    p = len(names)
    C = np.zeros((p, p))
    for a in range(p):
        pa = anc[names[a]]
        for b in range(a, p):
            pb = anc[names[b]]
            k = 0
            while k < len(pa) and k < len(pb) and pa[k] is pb[k]:
                k += 1
            shared = sum((x.length or 0.0) for x in pa[1:k])
            C[a, b] = C[b, a] = shared
        C[a, a] = sum((x.length or 0.0) for x in pa[1:])
    return C
