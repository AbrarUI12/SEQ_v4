#!/usr/bin/env python3
"""Pure-stdlib tests for seq_core.stats_utils (runnable without torch/numpy)."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seq_core.stats_utils import (  # noqa: E402
    aligned_correlation,
    average_bits,
    bits_to_tier,
    correlation_pvalue,
    greedy_bit_allocation,
    kendall_tau,
    pearson,
    rankdata,
    sensitivity_reliability,
    spearman,
)

FAILS = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILS.append(msg)
        print("FAIL:", msg)
    else:
        print("ok  :", msg)


def approx(a, b, tol=1e-9):
    return a is not None and b is not None and abs(a - b) <= tol


# --- rankdata ---
check(rankdata([10, 20, 30]) == [1.0, 2.0, 3.0], "rankdata monotone")
check(rankdata([10, 10, 30]) == [1.5, 1.5, 3.0], "rankdata ties averaged")
check(rankdata([]) == [], "rankdata empty")

# --- perfect correlations ---
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
check(approx(pearson(x, y), 1.0), "pearson perfect +1")
check(approx(spearman(x, y), 1.0), "spearman perfect +1")
check(approx(kendall_tau(x, y), 1.0), "kendall perfect +1")
yr = [10, 8, 6, 4, 2]
check(approx(pearson(x, yr), -1.0), "pearson perfect -1")
check(approx(spearman(x, yr), -1.0), "spearman perfect -1")
check(approx(kendall_tau(x, yr), -1.0), "kendall perfect -1")

# --- spearman vs known value (monotone but nonlinear -> rho=1, pearson<1) ---
xs = [1, 2, 3, 4, 5]
ys = [1, 4, 9, 16, 25]
check(approx(spearman(xs, ys), 1.0), "spearman rank-monotone == 1")
check(pearson(xs, ys) < 1.0, "pearson < 1 for nonlinear monotone")

# --- kendall tau-b known small case ---
# a=[1,2,3,4], b=[1,2,4,3] -> 5 concordant, 1 discordant, tau = (5-1)/6
a = [1, 2, 3, 4]
b = [1, 2, 4, 3]
check(approx(kendall_tau(a, b), (5 - 1) / 6.0), "kendall tau-b known case")

# --- degenerate guards ---
check(pearson([1, 1, 1], [1, 2, 3]) is None, "pearson constant -> None")
check(spearman([1], [1]) is None, "spearman n<2 -> None")

# --- aligned_correlation on dicts with missing / non-finite keys ---
sig = {"a": 1.0, "b": 2.0, "c": 3.0, "d": float("nan"), "e": 5.0}
gt = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "z": 9.0}  # d dropped (nan), e/z unshared
res = aligned_correlation(sig, gt)
check(res["n"] == 3, "aligned_correlation intersects finite shared keys (n=3)")
check(approx(res["spearman"], 1.0), "aligned_correlation spearman == 1")

# --- average_bits ---
pc = {"x": 100, "y": 300}  # 25% / 75%
check(approx(average_bits({"x": 8, "y": 4}, pc), 0.25 * 8 + 0.75 * 4), "average_bits weighted")

# --- greedy_bit_allocation: sensitive units get more bits, budget ~respected ---
signal = {f"m{i}": float(i) for i in range(10)}       # m9 most sensitive
counts = {f"m{i}": 1000 for i in range(10)}           # equal size
alloc = greedy_bit_allocation(signal, counts, levels=[4, 8, 16], target_bits=6.0)
avg = average_bits(alloc, counts)
check(5.5 <= avg <= 6.6, f"alloc average near target (got {avg:.2f})")
check(alloc["m9"] >= alloc["m0"], "alloc gives >= bits to more sensitive unit")
check(min(alloc.values()) >= 4 and max(alloc.values()) <= 16, "alloc respects levels")

# protected floor honored
alloc2 = greedy_bit_allocation(
    signal, counts, levels=[4, 8, 16], target_bits=5.0, protected={"m0": 16}
)
check(alloc2["m0"] == 16, "alloc respects protected floor")

# target below floor: everyone already >= budget -> stays at floor
alloc3 = greedy_bit_allocation(
    {"a": 1.0, "b": 2.0}, {"a": 1, "b": 1}, levels=[4, 8], target_bits=3.0
)
check(min(alloc3.values()) == 4, "alloc floor when target below min level")

# --- correlation_pvalue (normal approx): matches hand values from the runs ---
# entropy on 1B: rho=0.164, n=113 -> not significant (p ~ 0.08)
p_1b = correlation_pvalue(0.164, 113)
check(p_1b is not None and 0.06 <= p_1b <= 0.10, f"pvalue weak rho not significant (got {p_1b})")
# entropy on 8B: rho=0.188, n=225 -> significant (p ~ 0.005)
p_8b = correlation_pvalue(0.188, 225)
check(p_8b is not None and p_8b <= 0.01, f"pvalue larger-n rho significant (got {p_8b})")
check(correlation_pvalue(1.0, 100) < 1e-6, "pvalue perfect corr ~0")
check(correlation_pvalue(0.0, 100) is not None and abs(correlation_pvalue(0.0, 100) - 1.0) < 1e-9, "pvalue zero corr == 1")
check(correlation_pvalue(None, 100) is None, "pvalue None rho -> None")

# --- sensitivity_reliability: catches a noise-dominated ground truth ---
noisy = [-0.05, -0.01, 0.0, 0.001, 0.002, 0.005, -0.003, 2.3, 0.004, -0.02]
rel = sensitivity_reliability(noisy, noise_threshold=0.02)
check(rel["n"] == 10, "reliability n")
check(rel["frac_negative"] == 0.4, "reliability frac_negative")
check(rel["top1_share"] > 0.85, "reliability top1 share dominated by one unit")
check(0.0 <= rel["frac_below_noise"] <= 1.0, "reliability frac_below_noise in range")

# --- bits_to_tier ---
check(bits_to_tier(3) == "int4" and bits_to_tier(4) == "int4", "tier int4")
check(bits_to_tier(5) == "int8" and bits_to_tier(8) == "int8", "tier int8")
check(bits_to_tier(16) == "fp16", "tier fp16")

print("\n%d checks, %d failures" % (CHECKS, len(FAILS)))
if __name__ == "__main__":
    sys.exit(1 if FAILS else 0)
