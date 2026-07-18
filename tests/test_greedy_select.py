#!/usr/bin/env python3
"""Pure-stdlib tests for the greedy OMP channel selector (no torch).

Exercises the pure reference implementation, which mirrors the torch entry point
exactly. Ground truth for the greedy step is the module's own ``residual_energy``
(``tr(ΔWᵀ ΔW H)``): the best single channel to protect is the one whose removal
reduces that energy most, and greedy must (a) match that for the first pick and
(b) reduce the residual monotonically at every accepted step.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seq_core.greedy_select import (  # noqa: E402
    greedy_independent_reference,
    greedy_select_reference,
    residual_energy,
)

FAILS = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    (print("ok  :", msg) if cond else (FAILS.append(msg), print("FAIL:", msg)))


def zero_col(dw, j):
    return [[(0.0 if c == j else v) for c, v in enumerate(row)] for row in dw]


def eye(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def brute_force_best_single(dw, H, in_f):
    """Channel whose protection (column zeroing) most reduces residual energy."""
    base = residual_energy(dw, H)
    best_j, best_red = -1, float("-inf")
    for j in range(in_f):
        red = base - residual_energy(zero_col(dw, j), H)
        if red > best_red:
            best_j, best_red = j, red
    return best_j, best_red


# --- residual_energy sanity ------------------------------------------------- #
# A = I2 ; H = I2 -> tr(AᵀA H) = 2
check(abs(residual_energy([[1.0, 0.0], [0.0, 1.0]], eye(2)) - 2.0) < 1e-9, "residual energy I,I = 2")
# A = I2 ; H = diag(2,3) -> 1*2 + 1*3 = 5
check(abs(residual_energy([[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 3.0]]) - 5.0) < 1e-9,
      "residual energy weighted diag = 5")
# zeroing a column lowers energy under identity H
dw0 = [[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]]  # col energies 1,4,9,16
check(abs(residual_energy(dw0, eye(4)) - 30.0) < 1e-9, "sum of column energies = 30")

# --- diagonal H: greedy == sort by column energy desc ----------------------- #
order = greedy_select_reference(dw0, eye(4), 4)
check(order == [3, 2, 1, 0], "identity H -> greedy picks highest-energy columns first")
# monotone strictly-decreasing residual along the greedy path
energies = []
cur = [list(r) for r in dw0]
for j in order:
    energies.append(residual_energy(cur, eye(4)))
    cur = zero_col(cur, j)
energies.append(residual_energy(cur, eye(4)))
check(all(energies[i] > energies[i + 1] for i in range(len(energies) - 1)), "identity H -> residual strictly decreasing")
check(abs(energies[-1]) < 1e-9, "all columns protected -> zero residual")

# --- non-diagonal (interaction) H: first pick matches brute force ------------ #
# H = XᵀX for X (in=3, T=2): X rows = channels, cols = tokens
X = [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]  # 3 channels x 2 tokens
H = [[sum(X[a][t] * X[b][t] for t in range(2)) for b in range(3)] for a in range(3)]
# H = [[2,1,1],[1,1,0],[1,0,1]]  (coupled channels)
check(H == [[2.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]], "constructed coupled Hessian")
dw1 = [[0.5, 1.0, 0.3], [0.2, -0.4, 0.9]]  # out=2, in=3
bf_j, bf_red = brute_force_best_single(dw1, H, 3)
g1 = greedy_select_reference(dw1, H, 1)
check(g1 == [bf_j], "coupled H -> greedy first pick == brute-force best single channel")
check(bf_red > 0, "best single-channel protection reduces residual")

# --- monotone decrease under coupled H, and full protection empties residual - #
order3 = greedy_select_reference(dw1, H, 3)
check(sorted(order3) == [0, 1, 2], "greedy selects all channels when k=in")
cur = [list(r) for r in dw1]
path = [residual_energy(cur, H)]
for j in order3:
    cur = zero_col(cur, j)
    path.append(residual_energy(cur, H))
check(all(path[i] > path[i + 1] - 1e-12 for i in range(len(path) - 1)), "coupled H -> residual non-increasing along path")
check(abs(path[-1]) < 1e-9, "coupled H -> all protected empties residual")

# --- greedy k=2 residual beats every single-channel choice ------------------- #
best_single_energy = min(residual_energy(zero_col(dw1, j), H) for j in range(3))
o2 = greedy_select_reference(dw1, H, 2)
cur = [list(r) for r in dw1]
for j in o2:
    cur = zero_col(cur, j)
check(residual_energy(cur, H) <= best_single_energy + 1e-12, "greedy k=2 residual <= best k=1 residual")

# --- edge cases ------------------------------------------------------------- #
check(greedy_select_reference([[1.0, 2.0]], H=[[1.0, 0.0], [0.0, 1.0]], k=0) == [], "k=0 -> none")
check(greedy_select_reference([], [], 3) == [], "empty weight -> none")
# zero error weight -> no channel reduces residual -> stop early (nothing to protect)
check(greedy_select_reference([[0.0, 0.0]], eye(2), 2) == [], "zero ΔW -> no protection needed")

# --- selection-order regression guards -------------------------------------- #
# The first implementation accidentally sorted the selected IDs before
# returning them.  This synthetic diagonal case has priority [2, 1, 0], which
# is intentionally different from channel-index order [0, 1, 2].
priority = greedy_select_reference([[1.0, 2.0, 3.0]], eye(3), 3)
check(priority == [2, 1, 0], "greedy return preserves selection priority, not index order")
prefix = greedy_select_reference([[1.0, 2.0, 3.0]], eye(3), 2)
check(priority[:2] == prefix, "greedy prefixes are consistent across k")
check(len(priority) == len(set(priority)), "greedy order has no duplicate channels")
check(greedy_select_reference([[1.0, 2.0, 3.0]], eye(3), 0) == [], "greedy k=0 returns no channels")
# Ties are deterministic because the first maximum encountered wins.
tie_a = greedy_select_reference([[1.0, 1.0, 1.0]], eye(3), 3)
tie_b = greedy_select_reference([[1.0, 1.0, 1.0]], eye(3), 3)
check(tie_a == tie_b == [0, 1, 2], "greedy ties are deterministic")


# --- greedy_independent (interaction-free ablation) ------------------------- #
def first_step_gains(dw, H, in_f, out_f):
    """G_j^0 = 2<A_:,j, (A H)_:,j> - ||A_:,j||^2 H_jj, computed once."""
    RX = [[sum(dw[o][a] * H[a][j] for a in range(in_f)) for j in range(in_f)] for o in range(out_f)]
    G = []
    for j in range(in_f):
        dot = sum(dw[o][j] * RX[o][j] for o in range(out_f))
        nrm = sum(dw[o][j] * dw[o][j] for o in range(out_f))
        G.append(2.0 * dot - nrm * H[j][j])
    return G

# independent ranking == argsort of the one-shot gains (desc), positive only
Gind = first_step_gains(dw1, H, 3, 2)
brute_order = [j for j in sorted(range(3), key=lambda j: Gind[j], reverse=True) if Gind[j] > 0]
check(greedy_independent_reference(dw1, H, 3) == brute_order, "greedy_indep == argsort of first-step gains")
# its top pick equals greedy's first pick (identical first-step objective)...
check(greedy_independent_reference(dw1, H, 1)[0] == greedy_select_reference(dw1, H, 1)[0],
      "greedy_indep top pick == greedy first pick")
# ...but the FULL orders can diverge, because greedy re-evaluates after each pick.
# Construct a coupled case where the interaction-free order differs from greedy's.
Wd = [[3.0, 2.9, 0.1], [0.0, 0.0, 3.0]]
Hc = [[1.0, 0.95, 0.0], [0.95, 1.0, 0.0], [0.0, 0.0, 1.0]]  # cols 0,1 strongly coupled
gi = greedy_independent_reference(Wd, Hc, 2)
gg = greedy_select_reference(Wd, Hc, 2)
check(gi[0] == gg[0], "coupled: same first pick")
check(gi != gg, "coupled: interaction-free order differs from iterative greedy")
# greedy's 2-set has <= residual energy than the independent 2-set (greedy optimizes it)
def energy_of(order, dw, H):
    cur = [list(r) for r in dw]
    for j in order:
        cur = zero_col(cur, j)
    return residual_energy(cur, H)
check(energy_of(gg, Wd, Hc) <= energy_of(gi, Wd, Hc) + 1e-12,
      "greedy 2-set residual <= independent 2-set residual")
# edge cases
check(greedy_independent_reference([], [], 3) == [], "greedy_indep empty -> none")
check(greedy_independent_reference([[0.0, 0.0]], eye(2), 2) == [], "greedy_indep zero ΔW -> none")
check(greedy_independent_reference(dw1, H, 0) == [], "greedy_indep k=0 -> none")

print("\n%d checks, %d failures" % (CHECKS, len(FAILS)))
if __name__ == "__main__":
    sys.exit(1 if FAILS else 0)
