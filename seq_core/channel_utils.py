#!/usr/bin/env python3
"""Pure-stdlib helpers for per-channel protection (no torch), so the
selection and effective-bit accounting can be unit-tested anywhere."""
from __future__ import annotations

import math
from typing import List, Optional, Sequence


def select_protected_channels(scores: Sequence[float], k_frac: float) -> List[int]:
    """Indices of the top ``k_frac`` fraction of channels by score (desc, ties by index).

    Non-finite scores are never selected. Returns a sorted index list.
    """
    n = len(scores)
    if n == 0 or k_frac <= 0.0:
        return []
    k = min(n, int(math.ceil(k_frac * n)))
    finite = [(i, float(s)) for i, s in enumerate(scores) if s is not None and _finite(s)]
    finite.sort(key=lambda t: (-t[1], t[0]))
    return sorted(i for i, _ in finite[:k])


def layer_effective_bits(
    in_features: int,
    num_protected: int,
    base_bits: int,
    protect_bits: int = 16,
) -> float:
    """Per-parameter effective bits for a column-split layer."""
    if in_features <= 0:
        return float(base_bits)
    num_protected = max(0, min(int(num_protected), int(in_features)))
    rest = in_features - num_protected
    return (rest * base_bits + num_protected * protect_bits) / in_features


def _finite(v: object) -> bool:
    try:
        return math.isfinite(float(v))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
