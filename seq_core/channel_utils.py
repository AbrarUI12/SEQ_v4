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


def bucket_by_rank(scores: Sequence[float], num_buckets: int) -> List[List[int]]:
    """Split channel indices into ``num_buckets`` contiguous groups by score rank.

    Bucket 0 = highest-scored channels ... bucket B-1 = lowest. Used by the
    sensitivity audit to measure the true protection value of each rank band and
    check whether a signal's ordering matches measured importance. Non-finite
    scores sort to the bottom.
    """
    n = len(scores)
    if n == 0 or num_buckets <= 0:
        return []
    order = sorted(range(n), key=lambda i: (_finite(scores[i]), scores[i] if _finite(scores[i]) else 0.0), reverse=True)
    buckets: List[List[int]] = []
    # even split with the remainder spread over the first buckets
    base, rem = divmod(n, num_buckets)
    start = 0
    for b in range(num_buckets):
        size = base + (1 if b < rem else 0)
        if size == 0:
            continue
        buckets.append(sorted(order[start:start + size]))
        start += size
    return buckets


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
