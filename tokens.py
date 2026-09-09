"""
IMASM TOKEN SPACE — 12 tokens in 4 algebraic families.

The iterator maps 12^8 = 429,981,696 arrangements of length 8
(plus variable-length arrangements from length 1 to 8, totaling ~469M).

Token index (0-11) is used internally for fast integer encoding.
"""

from enum import IntEnum
from typing import Tuple, List, Dict

class Family(IntEnum):
    LOGICAL = 0     # 6 tokens: VINIT, TANCH, AFWD, AREV, CLINK, IMSCRIB
    FROBENIUS = 1   # 2 tokens: FSPLIT, FFUSE
    DIALETHEIA = 2  # 3 tokens: EVALT, EVALF, ENGAGR
    LINEAR = 3      # 1 token:  IFIX

class Token(IntEnum):
    """The 12 IMASM tokens. Integer value = index (0-11)."""
    VINIT   = 0   # Logical: initial object (void)
    TANCH   = 1   # Logical: terminal object (boundary)
    AFWD    = 2   # Logical: forward morphism
    AREV    = 3   # Logical: reverse morphism
    CLINK   = 4   # Logical: composition of morphisms
    IMSCRIB = 5   # Logical: identity morphism
    FSPLIT  = 6   # Frobenius: split (δ)
    FFUSE   = 7   # Frobenius: fuse (μ)
    EVALT   = 8   # Dialetheia: evaluate-true
    EVALF   = 9   # Dialetheia: evaluate-false
    ENGAGR  = 10  # Dialetheia: engage/recognize paradox
    IFIX    = 11  # Linear: irreversible fixation (!)

# Token metadata
TOKEN_NAMES: List[str] = [t.name for t in Token]
TOKEN_COUNT: int = 12

# ─── DAG arity ─────────────────────────────────────────────────────────────────
# (in-degree, out-degree) for each token as a node in the arrangement DAG.
# All tokens are (1,1) linear except VINIT (source), FSPLIT (fork), FFUSE (join).
TOKEN_ARITY: Dict[Token, Tuple[int, int]] = {
    Token.VINIT:   (0, 1),  # source
    Token.TANCH:   (1, 1),
    Token.AFWD:    (1, 1),
    Token.AREV:    (1, 1),
    Token.CLINK:   (1, 1),
    Token.IMSCRIB: (1, 1),
    Token.FSPLIT:  (1, 2),  # fork: one in, two out (T-branch and F-branch)
    Token.FFUSE:   (2, 1),  # join: two in (T-branch and F-branch), one out
    Token.EVALT:   (1, 1),  # constrained to T-branch
    Token.EVALF:   (1, 1),  # constrained to F-branch
    Token.ENGAGR:  (1, 1),
    Token.IFIX:    (1, 1),
}

# Tokens that are constrained to a specific branch between FSPLIT and FFUSE
TOKEN_BRANCH: Dict[Token, str] = {
    Token.EVALT: 'T',
    Token.EVALF: 'F',
}

FORK_TOKENS: frozenset = frozenset({Token.FSPLIT})
JOIN_TOKENS: frozenset = frozenset({Token.FFUSE})

# Family membership
TOKEN_FAMILY: Dict[Token, Family] = {
    Token.VINIT:   Family.LOGICAL,
    Token.TANCH:   Family.LOGICAL,
    Token.AFWD:    Family.LOGICAL,
    Token.AREV:    Family.LOGICAL,
    Token.CLINK:   Family.LOGICAL,
    Token.IMSCRIB: Family.LOGICAL,
    Token.FSPLIT:  Family.FROBENIUS,
    Token.FFUSE:   Family.FROBENIUS,
    Token.EVALT:   Family.DIALETHEIA,
    Token.EVALF:   Family.DIALETHEIA,
    Token.ENGAGR:  Family.DIALETHEIA,
    Token.IFIX:    Family.LINEAR,
}

# Family sizes
FAMILY_SIZE: Dict[Family, int] = {
    Family.LOGICAL: 6,
    Family.FROBENIUS: 2,
    Family.DIALETHEIA: 3,
    Family.LINEAR: 1,
}

FAMILY_NAMES: Dict[Family, str] = {
    Family.LOGICAL: "Logical",
    Family.FROBENIUS: "Frobenius",
    Family.DIALETHEIA: "Dialetheia",
    Family.LINEAR: "Linear",
}

# Tokens per family
FAMILY_TOKENS: Dict[Family, List[Token]] = {
    Family.LOGICAL:    [Token.VINIT, Token.TANCH, Token.AFWD, Token.AREV, Token.CLINK, Token.IMSCRIB],
    Family.FROBENIUS:  [Token.FSPLIT, Token.FFUSE],
    Family.DIALETHEIA: [Token.EVALT, Token.EVALF, Token.ENGAGR],
    Family.LINEAR:     [Token.IFIX],
}

def token_name(idx: int) -> str:
    """Convert index 0-11 to token name string."""
    return Token(idx).name

def token_family(idx: int) -> Family:
    """Get family for token index."""
    return TOKEN_FAMILY[Token(idx)]

def signature(arr: Tuple[int, ...]) -> Tuple[int, int, int, int]:
    """Compute family signature (L, F, D, X) for an arrangement."""
    counts = [0, 0, 0, 0]
    for t in arr:
        counts[token_family(t)] += 1
    return (counts[0], counts[1], counts[2], counts[3])

def arrangement_str(arr: Tuple[int, ...]) -> str:
    """Pretty-print arrangement as flat token chain (laminated/serialized form)."""
    return " → ".join(token_name(t) for t in arr)


def rotat(arr: Tuple[int, ...], k: int = 1) -> Tuple[int, ...]:
    """ROTAT — the first op-opcode: the cyclic shift of an arrangement, by k (default 1).

    An op-opcode is an operator ON a word, not a token IN it (not one of the 12); it maps a
    whole arrangement to another. ROTAT is the ring automorphism, the Weyl-Heisenberg shift.
    The family `signature` is ROTAT-invariant (rotation permutes positions, never the
    multiset), so an arrangement and all its rotations share a fingerprint: the rotation
    ORBIT is the natural equivalence the 12 canonical classes are read up to. Between two
    arrangements being bound, ROTAT sets their relative phase.
    """
    if not arr:
        return arr
    s = k % len(arr)
    return arr[s:] + arr[:s]


def rotat_orbit(arr: Tuple[int, ...]) -> frozenset:
    """The full ROTAT orbit of an arrangement: every distinct cyclic rotation of it."""
    return frozenset(rotat(arr, k) for k in range(len(arr))) if arr else frozenset()


def dag_str(arr: Tuple[int, ...]) -> str:
    """Render arrangement as a DAG expression, unflattened at FSPLIT/FFUSE pairs.

    Sequential: A . B . C
    Fork block: ⟨ T-branch | F-branch ⟩
    Unmatched FSPLIT or FFUSE: rendered inline as named tokens.
    """
    return _render_segment(list(arr))


def _render_segment(tokens: List[int]) -> str:
    """Recursively render a token list, detecting matched FSPLIT/FFUSE pairs."""
    # Find first FSPLIT
    split_idx = next((i for i, t in enumerate(tokens) if t == Token.FSPLIT), None)
    if split_idx is None:
        return _linear(tokens)

    # Find first FFUSE after it
    fuse_idx = next(
        (i for i, t in enumerate(tokens) if i > split_idx and t == Token.FFUSE), None
    )
    if fuse_idx is None:
        # Unmatched FSPLIT — render linearly
        return _linear(tokens)

    pre   = tokens[:split_idx]
    block = tokens[split_idx + 1 : fuse_idx]
    post  = tokens[fuse_idx + 1 :]

    t_branch, f_branch = _split_branches(block)
    t_str = _linear(t_branch) if t_branch else "∅"
    f_str = _linear(f_branch) if f_branch else "∅"
    fork  = f"⟨ {t_str} | {f_str} ⟩"

    parts = []
    if pre:
        parts.append(_linear(pre))
    parts.append(fork)
    if post:
        parts.append(_render_segment(post))  # recurse for additional fork blocks

    return " . ".join(parts)


def _linear(tokens: List[int]) -> str:
    return " . ".join(token_name(t) for t in tokens)


def _split_branches(block: List[int]) -> Tuple[List[int], List[int]]:
    """Split fork-block tokens into (T-branch, F-branch) by branch-typed anchors.

    EVALT anchors the T-branch; EVALF anchors the F-branch.
    Tokens before the first anchor follow the first-encountered branch.
    If no anchors present: all tokens go to T-branch (single-sided fork).
    """
    t_anchor = next((i for i, t in enumerate(block) if t == Token.EVALT), None)
    f_anchor = next((i for i, t in enumerate(block) if t == Token.EVALF), None)

    if t_anchor is None and f_anchor is None:
        return block[:], []

    if t_anchor is not None and f_anchor is None:
        return block[:], []

    if t_anchor is None and f_anchor is not None:
        return [], block[:]

    # Both present — split at the later anchor
    if t_anchor < f_anchor:
        return block[:f_anchor], block[f_anchor:]
    else:
        return block[t_anchor:], block[:t_anchor]
