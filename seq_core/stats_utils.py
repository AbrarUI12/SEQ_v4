#!/usr/bin/env python3
"""Pure-stdlib statistics for SEQ signal-quality analysis.

This module deliberately has **no numpy / torch dependency** so it can be unit
tested anywhere and reused by the sensitivity harness. It provides:

- rank-correlation (Spearman rho, Kendall tau-b) between a candidate *signal*
  and a *ground-truth* sensitivity ranking, and
- greedy discrete bit-allocation that turns a per-unit signal into a precision
  map at a target effective-bit budget.

All functions operate on plain Python dicts / lists of floats.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# --------------------------------------------------------------------------- #
# Ranking helpers
# --------------------------------------------------------------------------- #
def rankdata(values: Sequence[float]) -> List[float]:
    """Fractional (average) ranks, 1-based, ties averaged. Mirrors scipy."""
    n = len(values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        # positions i..j (inclusive) are tied; average of 1-based ranks
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Pearson correlation; None if undefined (n<2 or a constant vector)."""
    n = len(x)
    if n != len(y) or n < 2:
        return None
    mx, my = _mean(x), _mean(y)
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    sxx = sum((xi - mx) ** 2 for xi in x)
    syy = sum((yi - my) ** 2 for yi in y)
    if sxx <= 0.0 or syy <= 0.0:
        return None
    return sxy / math.sqrt(sxx * syy)


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Spearman rho = Pearson on fractional ranks (tie-safe)."""
    if len(x) != len(y) or len(x) < 2:
        return None
    return pearson(rankdata(x), rankdata(y))


def kendall_tau(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Kendall tau-b (tie-corrected). O(n^2), fine for hundreds of units."""
    n = len(x)
    if n != len(y) or n < 2:
        return None
    concordant = discordant = 0
    tie_x = tie_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            s = (dx > 0) - (dx < 0)
            t = (dy > 0) - (dy < 0)
            prod = s * t
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
            else:
                if dx == 0:
                    tie_x += 1
                if dy == 0:
                    tie_y += 1
    n0 = concordant + discordant + tie_x + tie_y  # counts each shared tie once
    denom = math.sqrt((n0 - tie_x) * (n0 - tie_y))
    if denom <= 0.0:
        return None
    return (concordant - discordant) / denom


def aligned_correlation(
    signal: Dict[str, float],
    ground_truth: Dict[str, float],
) -> Dict[str, Optional[float]]:
    """Correlate a signal dict against a ground-truth dict on shared, finite keys.

    Higher signal should predict higher sensitivity, so positive rho/tau == good.
    """
    keys = [
        k
        for k in ground_truth
        if k in signal
        and _finite(signal[k])
        and _finite(ground_truth[k])
    ]
    xs = [float(signal[k]) for k in keys]
    ys = [float(ground_truth[k]) for k in keys]
    return {
        "n": len(keys),
        "spearman": spearman(xs, ys),
        "kendall_tau": kendall_tau(xs, ys),
        "pearson": pearson(xs, ys),
    }


def _finite(v: object) -> bool:
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


# --------------------------------------------------------------------------- #
# Bit allocation: signal -> precision map at a target effective-bit budget
# --------------------------------------------------------------------------- #
def average_bits(bits_map: Dict[str, int], param_counts: Dict[str, int]) -> float:
    """Parameter-weighted average bit-width."""
    total = 0
    weighted = 0.0
    for name, bits in bits_map.items():
        c = int(param_counts.get(name, 0))
        total += c
        weighted += float(bits) * c
    return weighted / total if total > 0 else 0.0


def greedy_bit_allocation(
    signal: Dict[str, float],
    param_counts: Dict[str, int],
    levels: Sequence[int],
    target_bits: float,
    protected: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    """Allocate discrete bit levels by signal, hitting ~target average bits.

    Strategy: start everyone at the lowest level, then greedily upgrade the
    highest-signal (most sensitive) units one level at a time, choosing at each
    step the upgrade with the best sensitivity-per-extra-bit-cost ratio, until
    the parameter-weighted average reaches ``target_bits``.

    ``protected`` pins specific units to a minimum bit-width (e.g. lm_head).
    Higher signal == more sensitive == deserves more bits.
    """
    levels = sorted(set(int(b) for b in levels))
    if not levels:
        raise ValueError("levels must be non-empty")
    protected = protected or {}
    lo = levels[0]

    # current level index per unit
    names = list(signal.keys())
    idx: Dict[str, int] = {}
    for name in names:
        floor_bits = protected.get(name, lo)
        # snap protected floor up to the nearest available level
        start = 0
        for li, b in enumerate(levels):
            if b >= floor_bits:
                start = li
                break
        else:
            start = len(levels) - 1
        idx[name] = start

    bits_of = lambda name: levels[idx[name]]

    if average_bits({n: bits_of(n) for n in names}, param_counts) >= target_bits:
        # already at/above budget from protection floors alone
        return {n: bits_of(n) for n in names}

    # Precompute an upgrade "value": prioritize sensitive units, normalize by the
    # extra parameter-bits an upgrade costs so large layers don't dominate.
    def upgrade_gain(name: str) -> float:
        if idx[name] >= len(levels) - 1:
            return float("-inf")
        extra_bits = levels[idx[name] + 1] - levels[idx[name]]
        cost = max(1, int(param_counts.get(name, 1))) * max(1, extra_bits)
        return float(signal.get(name, 0.0)) / cost

    total_params = sum(int(param_counts.get(n, 0)) for n in names) or 1
    guard = 0
    max_steps = len(names) * len(levels) + 1
    while guard < max_steps:
        guard += 1
        # candidate = highest gain among upgradable units
        best = None
        best_gain = float("-inf")
        for name in names:
            g = upgrade_gain(name)
            if g > best_gain:
                best_gain = g
                best = name
        if best is None or best_gain == float("-inf"):
            break  # nobody left to upgrade
        idx[best] += 1
        avg = average_bits({n: bits_of(n) for n in names}, param_counts)
        if avg >= target_bits:
            break
    return {n: bits_of(n) for n in names}


def bits_to_tier(bits: int) -> str:
    """Map a bit-width to the SEQ tier vocabulary used by the bnb backend."""
    if bits <= 4:
        return "int4"
    if bits <= 8:
        return "int8"
    return "fp16"
