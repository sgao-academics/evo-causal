"""
Full Felsenstein 1985 Phylogenetic Independent Contrasts (PIC).

Felsenstein, J. (1985). Phylogenies and the comparative method.
The American Naturalist, 125(1), 1-15.

This is NOT the heuristic neighbor-joining approximation used in pipeline.py.
This is the full recursive algorithm operating on a rooted phylogenetic tree
with branch lengths.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import re


@dataclass
class PhyloNode:
    """Node in a rooted phylogenetic tree."""
    name: str           # species name (leaf) or internal label
    children: List['PhyloNode']  # left and right child
    branch_length: float  # length of branch TO this node from parent
    is_tip: bool = False
    index: Optional[int] = None  # index in data matrix (for tips)

    def __post_init__(self):
        if not self.children:
            self.is_tip = True


def parse_newick(newick_str: str, name_map: Optional[Dict[str, int]] = None) -> PhyloNode:
    """Parse a Newick format tree string with branch lengths.

    Example: '((human:0.1,chimp:0.1):0.05,mouse:0.15);'

    Args:
        newick_str: Newick format string with branch lengths
        name_map: optional mapping from tip names to data matrix indices

    Returns:
        Root node of the phylogenetic tree
    """
    newick_str = newick_str.strip().rstrip(';')

    def _parse_subtree(s: str, pos: int) -> Tuple[PhyloNode, int]:
        """Recursive descent parser for Newick format."""
        if s[pos] == '(':
            # Internal node
            pos += 1  # skip '('
            children = []
            while True:
                child, pos = _parse_subtree(s, pos)
                children.append(child)
                if pos >= len(s):
                    break
                if s[pos] == ')':
                    pos += 1
                    break
                if s[pos] == ',':
                    pos += 1
                    continue

            # Read branch length and optional label
            name = ""
            bl = 0.0
            if pos < len(s) and s[pos] not in (',', ')', ';', ':'):
                # Read label
                start = pos
                while pos < len(s) and s[pos] not in (':', ',', ')', ';'):
                    pos += 1
                name = s[start:pos]

            if pos < len(s) and s[pos] == ':':
                pos += 1
                start = pos
                while pos < len(s) and s[pos] in '0123456789.eE+-':
                    pos += 1
                bl = float(s[start:pos])

            node = PhyloNode(name=name, children=children, branch_length=bl)
            return node, pos

        else:
            # Tip node
            start = pos
            while pos < len(s) and s[pos] not in (':', ',', ')', ';'):
                pos += 1
            name = s[start:pos]
            bl = 0.0
            if pos < len(s) and s[pos] == ':':
                pos += 1
                start = pos
                while pos < len(s) and s[pos] in '0123456789.eE+-':
                    pos += 1
                bl = float(s[start:pos])

            idx = name_map.get(name) if name_map else None
            node = PhyloNode(name=name, children=[], branch_length=bl, is_tip=True, index=idx)
            return node, pos

    root, _ = _parse_subtree(newick_str, 0)
    return root


def _assign_indices(node: PhyloNode, name_to_idx: Dict[str, int]):
    """Assign data matrix indices to tip nodes."""
    if node.is_tip:
        node.index = name_to_idx.get(node.name)
    else:
        for child in node.children:
            _assign_indices(child, name_to_idx)


def _compute_contrasts_recursive(
    node: PhyloNode,
    data: np.ndarray,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Recursive Felsenstein PIC computation.

    For each internal node:
    1. Recursively compute contrasts for children
    2. Compute contrast between children: (x_left - x_right) / sqrt(v_left + v_right)
    3. Compute ancestral state as weighted average
    4. Update branch length: v_anc = v_left*v_right/(v_left+v_right) + branch_length

    Args:
        node: current node in the tree
        data: (n_tips, d) trait matrix

    Returns:
        (ancestral_state, list_of_contrasts) where ancestral_state is (d,) array
        and list_of_contrasts is list of (d,) arrays
    """
    if node.is_tip:
        if node.index is None:
            raise ValueError(f"Tip '{node.name}' has no data index")
        return data[node.index].copy(), []

    # Handle single-child node: just pass through (no bifurcation = no contrast)
    if len(node.children) == 1:
        state, contrasts = _compute_contrasts_recursive(node.children[0], data)
        # Pass branch length up
        node.branch_length += node.children[0].branch_length
        return state, contrasts

    # Handle polytomy (>2 children): recursively binary-ize
    if len(node.children) > 2:
        # Pairwise: first child vs merged rest
        left_state, left_contrasts = _compute_contrasts_recursive(node.children[0], data)
        # Merge remaining children into a single state
        right_state, right_contrasts = _compute_contrasts_recursive(node.children[1], data)
        for extra_child in node.children[2:]:
            extra_state, extra_contrasts = _compute_contrasts_recursive(extra_child, data)
            right_state = (right_state + extra_state) / 2
            right_contrasts.extend(extra_contrasts)
        # Adjust branch length for the merged child
        avg_bl = np.mean([c.branch_length for c in node.children[1:]])
        node.children[0].branch_length = node.children[0].branch_length
        # Use average for the virtual right child
        v_left = node.children[0].branch_length
        v_right = avg_bl
    else:
        left_state, left_contrasts = _compute_contrasts_recursive(node.children[0], data)
        right_state, right_contrasts = _compute_contrasts_recursive(node.children[1], data)
        v_left = node.children[0].branch_length
        v_right = node.children[1].branch_length

    # Standardized contrast
    denom = np.sqrt(max(v_left + v_right, 1e-8))
    contrast = (left_state - right_state) / denom

    # Ancestral state: weighted average
    w_left = 1.0 / max(v_left, 1e-8)
    w_right = 1.0 / max(v_right, 1e-8)
    ancestral = (w_left * left_state + w_right * right_state) / (w_left + w_right)

    # Update branch length for ancestral node
    # v_anc = v_left * v_right / (v_left + v_right) + node.branch_length
    node.branch_length = v_left * v_right / max(v_left + v_right, 1e-8) + node.branch_length

    all_contrasts = left_contrasts + right_contrasts + [contrast]
    return ancestral, all_contrasts


def felsenstein_pic(data: np.ndarray, tree: PhyloNode) -> np.ndarray:
    """Compute Felsenstein's phylogenetic independent contrasts.

    Args:
        data: (n_tips, d) trait matrix
        tree: rooted phylogenetic tree with branch lengths

    Returns:
        contrasts: (n_tips - 1, d) matrix of independent contrasts

    Warning
    -------
    The recursion writes the contrast-adjusted branch length back onto each internal node
    (that is how the parent step learns the variance of the aggregated subtree). A second
    call on the same tree object therefore returns different, wrong contrasts. Rather than
    relying on callers to remember this, a tree is marked once used and a repeat call
    raises, so the failure is loud instead of silent. Build a fresh tree per call; that is
    one object construction and costs nothing next to the regression itself.
    """
    if getattr(tree, '_pic_consumed', False):
        raise RuntimeError(
            'felsenstein_pic mutates branch lengths: this tree object has already been '
            'used for contrasts. Rebuild the tree and call again.')
    tree._pic_consumed = True
    _, contrasts = _compute_contrasts_recursive(tree, data)
    return np.array(contrasts)


def build_vertebrate_tree() -> Tuple[PhyloNode, Dict[str, int], List[str]]:
    """Build a well-calibrated vertebrate species tree with divergence times.

    Uses TimeTree-derived divergence times (million years ago, MYA).
    Branch lengths = divergence time / total tree height.

    Returns:
        (tree, name_to_idx, species_names)
    """
    # Divergence times from TimeTree (MYA)
    # Build as Newick string with proper branch lengths
    # Normalize: total tree height ~ 450 MYA (human-zebrafish)

    T = 450.0  # total tree height in MYA

    # Mammals clade
    # Primates: human-chimp ~7, human-macaque ~29
    # Glires: mouse-rat ~25, human-mouse ~90
    # Laurasiatheria: cow-pig ~60, human-cow ~96

    # Birds: chicken-zebra finch ~80

    # Sauropsids: bird-lizard ~280

    # Amphibians: frog

    # Teleost fish: zebrafish-medaka ~200, human-zebrafish ~450

    newick = (
        "(((((human:3.5,chimp:3.5):11.0,macaque:14.5):14.5,"
        "(mouse:12.5,rat:12.5):16.5):32.0,"   # human-rodent ~90
        "((cow:30.0,pig:30.0):33.0,"           # cow-pig ~60
        "(dog:50.0,(cat:45.0,horse:45.0):5.0):13.0):27.0):3.0,"  # human-cow ~96
        "opossum:99.0):50.0,"                   # marsupial-placental ~150
        "platypus:149.0):160.0,"                # monotreme-therian ~310
        "((chicken:40.0,zebra_finch:40.0):120.0,"  # bird divergence
        "lizard:160.0):149.0,"                  # bird-lizard ~280
        "frog:309.0):60.0,"                     # amphibian-amniote ~370
        "(zebrafish:100.0,medaka:100.0):80.0)"  # teleost fish
        ";"
    )

    species_names = [
        "human", "chimp", "macaque",
        "mouse", "rat",
        "cow", "pig",
        "dog", "cat", "horse",
        "opossum", "platypus",
        "chicken", "zebra_finch",
        "lizard", "frog",
        "zebrafish", "medaka"
    ]

    name_to_idx = {name: i for i, name in enumerate(species_names)}
    tree = parse_newick(newick, name_to_idx)
    _assign_indices(tree, name_to_idx)

    return tree, name_to_idx, species_names


def generate_realistic_evo_data(*args, **kwargs):
    """REMOVED 2026-11-24.

    The former body built the between-species covariance from the column index:

        d = rng.uniform(0.1, 0.5) * phylo_signal
        if abs(i - j) > n_species // 2:
            d *= 2.0

    Nothing in that expression refers to the tree, so the resulting covariance is a
    phylogenetic decoration rather than a phylogenetic process. That is the precise
    criticism the Editor-in-Chief made of the rejected manuscript, and it is not a
    criticism worth inviting twice. Use run_revision.simulate, where the latent factor is
    drawn with covariance V taken from the tree (L_V L_V^T = V) and therefore has
    phylogenetic structure by construction.

    Raising rather than returning zeros, so an unnoticed import cannot produce numbers.
    """
    raise NotImplementedError(
        'generate_realistic_evo_data was removed: its covariance ignored the tree. '
        'Use run_revision.simulate.')


if __name__ == "__main__":
    print("PIC module: use run_revision.py for a runnable demo.")
