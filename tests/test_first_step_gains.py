#!/usr/bin/env python3
"""Regression guard for the published alignment-analysis error (2026-07-28).

`scripts/measure_objective_alignment.py` once scored the greedy selector as
``‖ΔW_j‖²·H_jj`` and claimed that was its objective. The true first-step gain is
``2⟨ΔW_j,(ΔW H)_j⟩ − ‖ΔW_j‖²H_jj``; the diagonal expression drops the off-diagonal coupling and
ranks channels differently. These tests pin both facts:

  1. `first_step_gains` agrees with what the selector actually picks first.
  2. The diagonal proxy demonstrably does not — so nobody can quietly reintroduce it.
"""
import sys

import torch

sys.path.insert(0, ".")
from seq_core.greedy_select import first_step_gains, greedy_select_channels  # noqa: E402

CHECKS = 0
FAILS = []


def check(cond, label):
    global CHECKS
    CHECKS += 1
    print(("ok  : " if cond else "FAIL: ") + label)
    if not cond:
        FAILS.append(label)


def diag_proxy(dW, H):
    """The superseded expression. Kept only so the divergence stays measurable."""
    return (dW * dW).sum(dim=0) * torch.diagonal(H)


def spd(n, seed, off=0.6):
    """A correlated (strongly off-diagonal) SPD Hessian, like a real activation covariance."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(4 * n, n, generator=g)
    X[:, : max(1, n // 6)] *= 4.0
    H = X.T @ X / X.shape[0]
    H = H + off * torch.outer(torch.randn(n, generator=g), torch.randn(n, generator=g))
    return H @ H.T


# --- 1. the gain vector must agree with the selector's own first choice ---------------- #
agree = 0
for seed in range(12):
    g = torch.Generator().manual_seed(1000 + seed)
    n = 48
    H = spd(n, seed)
    dW = torch.randn(16, n, generator=g) * 0.05
    picked_first = greedy_select_channels(dW, H, 4, exact_k=True)[0]
    gains = first_step_gains(dW, H)
    agree += int(picked_first == int(gains.argmax()))
check(agree == 12, f"first_step_gains argmax == selector's first pick ({agree}/12 seeds)")

# --- 2. the diagonal proxy must NOT be treated as equivalent --------------------------- #
disagree = 0
for seed in range(20):
    g = torch.Generator().manual_seed(2000 + seed)
    n = 64
    H = spd(n, seed)
    dW = torch.randn(24, n, generator=g) * 0.05
    if int(first_step_gains(dW, H).argmax()) != int(diag_proxy(dW, H).argmax()):
        disagree += 1
check(disagree >= 10,
      f"diagonal proxy disagrees with the true gain on most correlated layers "
      f"({disagree}/20) -- it is NOT the greedy objective")

# --- 3. deterministic counterexample (audit reproduction) ------------------------------ #
dW = torch.tensor([[1.0, 0.2, 0.1], [0.0, 1.0, 0.3]])
H = torch.tensor([[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 2.0]])
tg = first_step_gains(dW, H)
check(torch.isfinite(tg).all(), "true gain is finite on the audit counterexample")
check(abs(float(tg[2]) - 0.2) < 1e-6,
      "isolated channel gain equals 2*e*H_jj - e*H_jj = e*H_jj for a decoupled column")

# --- 4. the cross-term is what distinguishes them: with a DIAGONAL H they coincide ------ #
Hd = torch.diag(torch.tensor([1.0, 2.0, 3.0]))
check(torch.allclose(first_step_gains(dW, Hd).float(), diag_proxy(dW, Hd), atol=1e-5),
      "with a diagonal Hessian the true gain reduces to the proxy (isolates the cross-term)")

print("\n%d checks, %d failures" % (CHECKS, len(FAILS)))
if __name__ == "__main__":
    sys.exit(1 if FAILS else 0)
