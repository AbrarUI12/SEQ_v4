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


def packed_storage_bits(
    in_features: int,
    out_features: int,
    base_bits: int,
    num_protected: int,
    *,
    group_size: int = 64,
    scale_zero_bits: int = 16,
    protect_bits: int = 16,
    index_bits: Optional[int] = None,
    count_base_scales: bool = True,
) -> float:
    """Honest *actual* bits/param for a channel-protected layer.

    Nominal effective bits (base_bits·(1−k) + 16·k) omits real overhead reviewers
    will recompute: FP16 residual columns, the channel index table, and the base
    quantizer's group scales/zero-points. This counts all of them.

    - protected columns cost ``protect_bits`` (FP16) instead of ``base_bits``,
    - each protected channel needs an index (``ceil(log2(in_features))`` bits),
    - the base quantizer stores 2 values (scale+zero) per group per output row.
    """
    params = max(1, in_features * out_features)
    rest = max(0, in_features - num_protected)
    weight_bits = out_features * (rest * base_bits + num_protected * protect_bits)
    if index_bits is None:
        index_bits = max(1, math.ceil(math.log2(max(2, in_features))))
    idx_bits = num_protected * index_bits
    scale_bits = 0
    if count_base_scales and group_size and group_size > 0:
        n_groups = math.ceil(rest / group_size) if rest > 0 else 0
        scale_bits = out_features * n_groups * 2 * scale_zero_bits
    return (weight_bits + idx_bits + scale_bits) / params


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


def normalize_minmax(arr: Sequence[float]) -> List[float]:
    """Per-array min-max normalize to [0,1]; non-finite -> 0; constant -> 0.5.

    Used to put per-channel signals on a common scale before combining them into
    a composite selection score (protection selects top-k *within* each layer,
    so this normalizes within a layer)."""
    vals = [float(v) if _finite(v) else None for v in arr]
    finite = [v for v in vals if v is not None]
    if not finite:
        return [0.0] * len(vals)
    lo, hi = min(finite), max(finite)
    if hi == lo:
        return [0.5 if v is not None else 0.0 for v in vals]
    span = hi - lo
    return [((v - lo) / span) if v is not None else 0.0 for v in vals]


def combine_scores(arrays: Sequence[Sequence[float]], op: str = "mul") -> List[float]:
    """Combine several equal-length per-channel score arrays elementwise.

    Each array is min-max normalized first; ``op`` is ``"mul"`` (favor channels
    high in *all* signals) or ``"add"`` (high in *any*). Empty -> []."""
    arrays = [a for a in arrays if a is not None and len(a) > 0]
    if not arrays:
        return []
    n = min(len(a) for a in arrays)
    norm = [normalize_minmax(a[:n]) for a in arrays]
    out: List[float] = []
    for i in range(n):
        if op == "add":
            out.append(sum(nrm[i] for nrm in norm))
        else:  # mul
            v = 1.0
            for nrm in norm:
                v *= nrm[i]
            out.append(v)
    return out


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
