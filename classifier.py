"""
IMASM ARRANGEMENT CLASSIFIER — Structural fingerprinting for the 430M space.

Two-tier classification:
  Coarse key — groups arrangements by the properties used in the 12 canonical classes
  Fine key   — full structural fingerprint (for exact matching)

Each arrangement receives a structural fingerprint capturing all properties
used in the 12 canonical classes. Arrangements with identical fingerprints
belong to the same structural class.
"""

from typing import Tuple, List, Dict, NamedTuple, Optional, Set
from tokens import (
    Token, Family, TOKEN_FAMILY, FAMILY_TOKENS,
    token_family, signature, token_name, TOKEN_COUNT
)


class StructuralFingerprint(NamedTuple):
    """Compact fingerprint of an arrangement's structural properties."""
    length: int                    # 1-8
    sig_L: int                     # logical count
    sig_F: int                     # frobenius count
    sig_D: int                     # dialetheia count
    sig_X: int                     # linear count
    start_token: int               # 0-11
    end_token: int                 # 0-11
    self_ref: bool                 # start == end
    frobenius_order: int           # 0=none, 1=split→fuse, 2=fuse→split, 3=both
    dialetheia_complete: bool      # all 3 dialetheia present
    period: int                    # minimal period (1=constant)
    token_mask: int                # 12-bit bitmask
    fam_adj_mask: int              # 16-bit: family→family adjacency
    trans_sig: str                 # e.g. "LL:3,LF:1,FD:2,..."

    @property
    def signature(self) -> Tuple[int, int, int, int]:
        return (self.sig_L, self.sig_F, self.sig_D, self.sig_X)

    @property
    def token_diversity(self) -> int:
        return self.token_mask.bit_count()

    @property
    def is_periodic(self) -> bool:
        return self.period < self.length

    @property
    def is_constant(self) -> bool:
        return self.period == 1

    @property
    def has_frobenius_pair(self) -> bool:
        return self.frobenius_order > 0

    @property
    def frobenius_inverted(self) -> bool:
        return self.frobenius_order == 2

    def description(self) -> str:
        parts = []
        parts.append(f"sig=({self.sig_L},{self.sig_F},{self.sig_D},{self.sig_X})")
        parts.append(f"start={token_name(self.start_token)}")
        parts.append(f"end={token_name(self.end_token)}")
        if self.self_ref:
            parts.append("self-ref")
        if self.frobenius_order == 1:
            parts.append("Frobenius:split→fuse")
        elif self.frobenius_order == 2:
            parts.append("Frobenius:fuse→split(INVERTED)")
        elif self.frobenius_order == 3:
            parts.append("Frobenius:multiple")
        if self.dialetheia_complete:
            parts.append("Dialetheia:complete")
        if self.is_periodic:
            parts.append(f"period={self.period}")
        if self.is_constant:
            parts.append("constant")
        parts.append(f"diversity={self.token_diversity}/12")
        return " | ".join(parts)

    def coarse_key(self) -> str:
        """Coarse classification key — groups arrangements by the properties
        used to distinguish the 12 canonical classes.

        Properties: signature, start/end token, self-ref, frob order,
        dialetheia complete, period, token diversity.
        """
        return (
            f"{self.length}|{self.sig_L},{self.sig_F},{self.sig_D},{self.sig_X}|"
            f"{self.start_token}|{self.end_token}|{int(self.self_ref)}|"
            f"{self.frobenius_order}|{int(self.dialetheia_complete)}|"
            f"{self.period}|{self.token_diversity}"
        )

    def fine_key(self) -> str:
        """Full fingerprint key — unique to exact structural fingerprint."""
        return (
            f"{self.length}|{self.sig_L},{self.sig_F},{self.sig_D},{self.sig_X}|"
            f"{self.start_token}|{self.end_token}|{int(self.self_ref)}|"
            f"{self.frobenius_order}|{int(self.dialetheia_complete)}|"
            f"{self.period}|{self.token_mask:012b}|{self.fam_adj_mask:04b}|"
            f"{self.trans_sig}"
        )


def compute_fingerprint(arr: Tuple[int, ...]) -> StructuralFingerprint:
    """Compute the structural fingerprint for an arrangement."""
    n = len(arr)
    sig = signature(arr)

    start_tok = arr[0]
    end_tok = arr[-1]
    self_ref = (start_tok == end_tok)

    token_mask = 0
    for t in arr:
        token_mask |= (1 << t)

    frob_order = _frobenius_order(arr)
    dial_complete = _dialetheia_complete(arr)
    period = _minimal_period(arr)
    fam_adj_mask = _family_adjacency_mask(arr)
    trans_sig = _transition_signature(arr)

    return StructuralFingerprint(
        length=n,
        sig_L=sig[0], sig_F=sig[1], sig_D=sig[2], sig_X=sig[3],
        start_token=start_tok,
        end_token=end_tok,
        self_ref=self_ref,
        frobenius_order=frob_order,
        dialetheia_complete=dial_complete,
        period=period,
        token_mask=token_mask,
        fam_adj_mask=fam_adj_mask,
        trans_sig=trans_sig,
    )


def _frobenius_order(arr: Tuple[int, ...]) -> int:
    fsplit_positions = [i for i, t in enumerate(arr) if t == Token.FSPLIT]
    ffuse_positions = [i for i, t in enumerate(arr) if t == Token.FFUSE]
    if not fsplit_positions or not ffuse_positions:
        return 0
    first_split = min(fsplit_positions)
    first_fuse = min(ffuse_positions)
    last_split = max(fsplit_positions)
    last_fuse = max(ffuse_positions)
    has_split_first = first_split < max(ffuse_positions)
    has_fuse_first = first_fuse < max(fsplit_positions)
    if has_split_first and has_fuse_first:
        return 3
    elif first_split < first_fuse:
        return 1
    else:
        return 2


def _dialetheia_complete(arr: Tuple[int, ...]) -> bool:
    return Token.EVALT in arr and Token.EVALF in arr and Token.ENGAGR in arr


def _minimal_period(arr: Tuple[int, ...]) -> int:
    n = len(arr)
    for p in range(1, n + 1):
        if n % p != 0:
            continue
        is_period = True
        for i in range(p, n):
            if arr[i] != arr[i - p]:
                is_period = False
                break
        if is_period:
            return p
    return n


def _family_adjacency_mask(arr: Tuple[int, ...]) -> int:
    mask = 0
    for i in range(len(arr) - 1):
        f_from = token_family(arr[i])
        f_to = token_family(arr[i + 1])
        bit = f_from * 4 + f_to
        mask |= (1 << bit)
    return mask


def _transition_signature(arr: Tuple[int, ...]) -> str:
    counts = {}
    for i in range(len(arr) - 1):
        f_from = token_family(arr[i])
        f_to = token_family(arr[i + 1])
        key = f"{_family_abbrev(f_from)}{_family_abbrev(f_to)}"
        counts[key] = counts.get(key, 0) + 1
    parts = [f"{k}:{v}" for k, v in sorted(counts.items())]
    return ",".join(parts)


def _family_abbrev(f: Family) -> str:
    return {Family.LOGICAL: 'L', Family.FROBENIUS: 'F',
            Family.DIALETHEIA: 'D', Family.LINEAR: 'X'}[f]


# ─── Canonical Class Matching ───────────────────────────────────

# The 12 canonical arrangements from IMASM_ARRANGEMENT_CLASSES.md
CANONICAL_CLASSES: Dict[str, Tuple[int, ...]] = {
    "I_Dialetheic_Bootstrap": (5, 8, 6, 9, 7, 10, 11, 5),
    "II_Void_Genesis": (0, 1, 2, 6, 4, 7, 11, 5),
    "III_Anchor_Protocol": (1, 3, 0, 2, 1, 4, 11, 5),
    "IV_Dual_Bootstrap": (5, 2, 7, 6, 3, 4, 11, 5),
    "V_Linear_Chain": (11, 11, 11, 11, 11, 11, 11, 11),
    "VI_Empty_Bootstrap": (0, 5, 0, 5, 0, 5, 0, 5),
    "VII_Parakernel": (9, 3, 6, 8, 2, 7, 10, 11),
    "VIII_Frobenius_Kernel": (0, 6, 7, 1),
    "IX_Chiral_Pairs": (2, 3, 2, 3, 2, 3, 2, 3),
    "X_Truth_Machine": (5, 6, 8, 11, 5, 6, 9, 11),
    "XI_Eternal_Return": (5, 2, 3, 5, 2, 3, 5, 2),
    "XII_ROM_Burn": (8, 11, 9, 11, 10, 11, 5, 11),
}

CANONICAL_FINGERPRINTS: Dict[str, StructuralFingerprint] = {
    name: compute_fingerprint(arr)
    for name, arr in CANONICAL_CLASSES.items()
}


def match_canonical(arr: Tuple[int, ...]) -> Optional[str]:
    """Check if an arrangement exactly matches a canonical class."""
    for name, canon in CANONICAL_CLASSES.items():
        if arr == canon:
            return name
    return None


def fingerprint_key(fp: StructuralFingerprint) -> str:
    """Legacy: full fine-grained key (for backward compat)."""
    return fp.fine_key()
