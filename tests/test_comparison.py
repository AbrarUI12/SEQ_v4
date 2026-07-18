#!/usr/bin/env python3
"""Pure-stdlib test for the comparison-table Pareto logic."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.build_comparison import pareto_frontier  # noqa: E402

FAILS = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    (print("ok  :", msg) if cond else (FAILS.append(msg), print("FAIL:", msg)))


# (bits, ppl); frontier = non-dominated toward (min bits, min ppl)
pts = [(4.0, 11.0), (5.0, 10.3), (6.0, 10.4), (4.0, 12.0), (8.0, 9.8)]
f = set(pareto_frontier(pts))
check(f == {0, 1, 4}, "frontier keeps non-dominated points")
check(2 not in f, "(6,10.4) dominated by (5,10.3)")
check(3 not in f, "(4,12) dominated by (4,11)")

# a strictly dominating point removes all others at higher bits+ppl
pts2 = [(4.0, 10.0), (5.0, 11.0), (6.0, 12.0)]
check(set(pareto_frontier(pts2)) == {0}, "single dominator")

# ties: equal points both non-dominated (no strict improvement)
pts3 = [(4.0, 10.0), (4.0, 10.0)]
check(len(pareto_frontier(pts3)) == 2, "identical points both kept")

check(pareto_frontier([]) == [], "empty -> empty")

print("\n%d checks, %d failures" % (CHECKS, len(FAILS)))
if __name__ == "__main__":
    sys.exit(1 if FAILS else 0)
