#!/usr/bin/env python3
"""Pure-stdlib helpers for per-channel protection (no torch), so the
selection and effective-bit accounting can be unit-tested anywhere."""
from __future__ import annotations

import heapq
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


def parse_tiers(spec: str) -> List[tuple]:
    """Parse '16:0.02,8:0.08' -> [(16, 0.02), (8, 0.08)] sorted by bits desc.

    Each entry is (protect_bits, fraction-of-channels). Highest precision first
    so the top-ranked channels get the most bits."""
    tiers = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        b, f = part.split(":")
        tiers.append((int(b), float(f)))
    tiers.sort(key=lambda t: -t[0])
    return tiers


def assign_tiers(scores: Sequence[float], tiers: List[tuple]) -> dict:
    """Assign channels to precision tiers by score rank.

    Top ``tiers[0][1]`` fraction (highest score) -> tiers[0] bits, next fraction
    -> tiers[1] bits, ... ; the remainder stays at the base bit-width.
    Returns ``{bits: [sorted channel indices]}``."""
    n = len(scores)
    if n == 0 or not tiers:
        return {}
    order = sorted(
        range(n),
        key=lambda i: (scores[i] if _finite(scores[i]) else float("-inf")),
        reverse=True,
    )
    out: dict = {}
    start = 0
    for bits, frac in tiers:
        cnt = min(n - start, int(math.ceil(frac * n)) if frac > 0 else 0)
        if cnt <= 0:
            continue
        out.setdefault(bits, [])
        out[bits] = sorted(out[bits] + order[start:start + cnt])
        start += cnt
        if start >= n:
            break
    return out


def greedy_bit_alloc_by_value(
    dist_per_channel: Sequence[Sequence[float]],
    tier_bits: Sequence[int],
    budget_extra_bits: float,
    *,
    index_bits: float = 0.0,
) -> List[int]:
    """Error-per-byte greedy bit allocation across precision tiers.

    ``dist_per_channel[j][t]`` is the distortion of input channel ``j`` if it is
    quantized at tier ``t`` (e.g. ``E[x_j²]·‖W_:,j − Q_t(W_:,j)‖²``); ``tier_bits``
    lists the bit-widths **ascending** (``tier_bits[0]`` is the base, higher tiers
    cost more but distort less). Every channel starts at the base tier; we then
    repeatedly apply the single next-tier upgrade with the greatest distortion
    reduction *per extra bit*, ``(D_t − D_{t+1}) / (cost_{t+1} − cost_t)``, while the
    cumulative extra cost stays within the budget.

    ``budget_extra_bits`` is the mean extra bits/channel allowed over the all-base
    cost (so the total budget is ``budget_extra_bits · n_channels``). Tiers above
    the base carry ``index_bits`` for the protected-channel index table. Returns
    the chosen tier index per channel (0 = base).

    This replaces fixed ``--protect_tiers`` percentages: instead of "protect the
    top 2% at 16-bit and next 8% at 8-bit", it spends a bit budget where it buys
    the most error reduction, which differs per layer.
    """
    n = len(dist_per_channel)
    ntiers = len(tier_bits)
    chosen = [0] * n
    if n == 0 or ntiers <= 1 or budget_extra_bits <= 0:
        return chosen

    def _cost(t: int) -> float:
        return float(tier_bits[t]) + (float(index_bits) if t > 0 else 0.0)

    def _next_upgrade(j: int, from_t: int):
        """Best-value single-step upgrade (from_t -> from_t+1) or None."""
        to_t = from_t + 1
        if to_t >= ntiers:
            return None
        row = dist_per_channel[j]
        if to_t >= len(row):
            return None
        dred = float(row[from_t]) - float(row[to_t])  # distortion reduction (>0 = helps)
        if not _finite(dred) or dred <= 0.0:
            return None
        incr = _cost(to_t) - _cost(from_t)
        value = dred / incr if incr > 0 else float("inf")
        return (-value, incr, j, to_t)

    total_budget = float(budget_extra_bits) * n
    spent = 0.0
    heap: List[tuple] = []
    for j in range(n):
        up = _next_upgrade(j, 0)
        if up is not None:
            heapq.heappush(heap, up)

    while heap and spent < total_budget:
        neg_val, incr, j, to_t = heapq.heappop(heap)
        if to_t != chosen[j] + 1:
            continue  # stale (defensive; each channel has at most one live entry)
        if incr > 0 and spent + incr > total_budget:
            continue  # this upgrade doesn't fit; budget only shrinks, so skip it for good
        chosen[j] = to_t
        spent += max(0.0, incr)
        up = _next_upgrade(j, to_t)
        if up is not None:
            heapq.heappush(heap, up)
    return chosen


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


def layer_effective_bits_tiered(
    in_features: int,
    tier_counts: dict,
    base_bits: int,
) -> float:
    """Nominal effective bits for a multi-tier protected layer.

    ``tier_counts`` = {protect_bits: num_channels}; the remainder is at base_bits."""
    if in_features <= 0:
        return float(base_bits)
    protected = sum(int(c) for c in tier_counts.values())
    protected = max(0, min(protected, in_features))
    rest = in_features - protected
    total = rest * base_bits + sum(int(b) * int(c) for b, c in tier_counts.items())
    return total / in_features


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
