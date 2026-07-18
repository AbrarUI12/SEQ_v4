#!/usr/bin/env python3
"""Greedy residual-reduction channel selection (interaction-aware).

The per-channel audit found that independent local scores (``act_max``,
``act_scale``, reconstruction sensitivity) miss cross-channel interactions: the
quantization-error residual ``R = (W − Wq) X`` couples input channels through the
input Hessian ``H = XᵀX``, so the best *set* of channels to protect is not the
top-k by any scalar signal.

This module selects channels by orthogonal-matching-pursuit on the layer output
residual. Protecting a channel (restoring it to FP16) zeroes its column of the
error weight ``ΔW = W − Wq``. The reduction in residual energy
``‖ΔW X‖²_F = tr(ΔWᵀ ΔW H)`` from protecting channel ``j``, given the already
protected columns (zeroed in the running ``ΔW``), is

    G_j = 2·⟨ΔW_:,j, (ΔW H)_:,j⟩ − ‖ΔW_:,j‖²·H_jj .

We greedily pick ``argmax_j G_j`` (with ``G_j > 0``), zero that column of ``ΔW``,
update the running product ``RX = ΔW H`` by a rank-1 correction
``RX ← RX − ΔW_:,j ⊗ H_j,:``, and repeat. This captures interactions the scalar
signals cannot and directly optimizes the quantity that maps to output error
(hence PPL). Setup is ``O(out·in²)`` (one ``ΔW H``); each of the ``k`` steps is
``O(out·in)``.

The torch entry point :func:`greedy_select_channels` runs on real layers; the
pure-Python :func:`greedy_select_reference` and :func:`residual_energy` mirror it
on nested lists so the algorithm is unit-tested without torch (see
``tests/test_greedy_select.py``). Both selectors return channels in *selection
order*, so a prefix ``order[:k]`` is the greedy set of size ``k`` (nested across
protection fractions — one pass serves every ``k`` in a sweep).
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence


# --------------------------------------------------------------------------- #
# Pure-Python reference (nested lists) — the algorithm, unit-testable w/o torch.
# --------------------------------------------------------------------------- #
def residual_energy(delta_w: Sequence[Sequence[float]], H: Sequence[Sequence[float]]) -> float:
    """``tr(ΔWᵀ ΔW H) = ‖ΔW X‖²_F`` for ``delta_w`` [out][in] and ``H`` [in][in]."""
    out_f = len(delta_w)
    in_f = len(delta_w[0]) if out_f else 0
    energy = 0.0
    for o in range(out_f):
        row = delta_w[o]
        # s[b] = Σ_a ΔW[o,a] H[a,b]; energy += Σ_b ΔW[o,b] s[b]
        for b in range(in_f):
            s = 0.0
            for a in range(in_f):
                s += row[a] * H[a][b]
            energy += row[b] * s
    return energy


def _gains_reference(
    A: Sequence[Sequence[float]],
    RX: Sequence[Sequence[float]],
    Hdiag: Sequence[float],
    in_f: int,
    out_f: int,
) -> List[float]:
    """``G_j = 2·⟨A_:,j, RX_:,j⟩ − ‖A_:,j‖²·H_jj`` for all j."""
    G = [0.0] * in_f
    for j in range(in_f):
        dot = 0.0
        nrm = 0.0
        for o in range(out_f):
            a = A[o][j]
            dot += a * RX[o][j]
            nrm += a * a
        G[j] = 2.0 * dot - nrm * Hdiag[j]
    return G


def greedy_select_reference(
    delta_w: Sequence[Sequence[float]],
    H: Sequence[Sequence[float]],
    k: int,
) -> List[int]:
    """Pure reference OMP selection; returns channel indices in *selection order*."""
    out_f = len(delta_w)
    in_f = len(delta_w[0]) if out_f else 0
    if in_f == 0 or k <= 0:
        return []
    k = min(int(k), in_f)
    A = [list(row) for row in delta_w]                      # working copy; protected cols zeroed
    RX = [[sum(A[o][a] * H[a][j] for a in range(in_f)) for j in range(in_f)] for o in range(out_f)]
    Hdiag = [float(H[j][j]) for j in range(in_f)]
    protected = set()
    order: List[int] = []
    for _ in range(k):
        G = _gains_reference(A, RX, Hdiag, in_f, out_f)
        best_j, best_g = -1, 0.0
        for j in range(in_f):
            if j in protected:
                continue
            if best_j < 0 or G[j] > best_g:
                best_j, best_g = j, G[j]
        if best_j < 0 or best_g <= 0.0:
            break  # no remaining channel reduces the residual -> stop early
        col = [A[o][best_j] for o in range(out_f)]
        hrow = H[best_j]
        for o in range(out_f):
            c = col[o]
            if c != 0.0:
                RXo = RX[o]
                for j in range(in_f):
                    RXo[j] -= c * hrow[j]
            A[o][best_j] = 0.0
        protected.add(best_j)
        order.append(best_j)
    return order


def greedy_independent_reference(
    delta_w: Sequence[Sequence[float]],
    H: Sequence[Sequence[float]],
    k: int,
) -> List[int]:
    """Pure reference for the interaction-free ablation (first-step gains only).

    Ranks columns by ``G_j⁰ = 2⟨ΔW_:,j,(ΔW H)_:,j⟩ − ‖ΔW_:,j‖²·H_jj`` computed once
    (no iterative residual update) and returns up to ``k`` positive-gain columns in
    descending gain order. Mirrors :func:`greedy_independent_order`. Its top pick
    equals :func:`greedy_select_reference`'s first pick (identical first-step
    objective); the two orders diverge afterwards because greedy re-evaluates.
    """
    out_f = len(delta_w)
    in_f = len(delta_w[0]) if out_f else 0
    if in_f == 0 or k <= 0:
        return []
    k = min(int(k), in_f)
    RX = [[sum(delta_w[o][a] * H[a][j] for a in range(in_f)) for j in range(in_f)] for o in range(out_f)]
    Hdiag = [float(H[j][j]) for j in range(in_f)]
    G = _gains_reference(delta_w, RX, Hdiag, in_f, out_f)
    order = sorted(range(in_f), key=lambda j: G[j], reverse=True)
    return [j for j in order if G[j] > 0.0][:k]


# --------------------------------------------------------------------------- #
# Torch entry point (lazy import) — runs on real layers.
# --------------------------------------------------------------------------- #
def greedy_select_channels(
    delta_w: "Any",
    H: "Any",
    k: int,
) -> List[int]:
    """OMP residual-reduction selection on a real layer.

    ``delta_w`` = ``W − Wq`` [out, in] (the quantization error weight), ``H`` =
    ``XᵀX`` [in, in] input Hessian, ``k`` = channels to protect. Returns the
    selected input-channel indices in selection order (take ``[:k']`` for a
    smaller fraction). Mirrors :func:`greedy_select_reference` exactly.
    """
    import torch

    if delta_w.ndim != 2:
        raise ValueError(f"delta_w must be [out, in], got {tuple(delta_w.shape)}")
    out_f, in_f = int(delta_w.shape[0]), int(delta_w.shape[1])
    if in_f == 0 or k <= 0:
        return []
    k = min(int(k), in_f)
    # float64 throughout: on an error-compensated (GPTQ) base the residual ΔW is
    # small and H is ill-conditioned, so the rank-1 ``RX -= col ⊗ H_j`` update
    # accumulates float32 drift over hundreds of steps and corrupts late gains
    # (the source of the k=0.10 blow-up). float64 + a periodic exact recompute of
    # ``RX = A @ H`` keeps the gains accurate; non-finite gains are masked out.
    A = delta_w.detach().to(dtype=torch.float64).clone()
    Hf = H.detach().to(dtype=torch.float64, device=A.device)
    if not bool(torch.isfinite(A).all()) or not bool(torch.isfinite(Hf).all()):
        A = torch.nan_to_num(A)
        Hf = torch.nan_to_num(Hf)
    Hdiag = torch.diagonal(Hf).clone()
    RX = A @ Hf                                              # [out, in]
    avail = torch.ones(in_f, dtype=torch.bool, device=A.device)
    neg_inf = torch.tensor(float("-inf"), dtype=RX.dtype, device=A.device)
    recompute_every = 64                                    # cancel accumulated drift
    order: List[int] = []
    for step in range(k):
        dot = (A * RX).sum(dim=0)                            # [in]
        nrm = (A * A).sum(dim=0)                             # [in]
        G = 2.0 * dot - nrm * Hdiag                          # [in]
        G = torch.where(avail & torch.isfinite(G), G, neg_inf)
        gmax, jstar = torch.max(G, dim=0)
        if not bool(torch.isfinite(gmax)) or float(gmax) <= 0.0:
            break
        j = int(jstar)
        col = A[:, j].clone()                                # [out]
        RX -= torch.outer(col, Hf[j, :])                     # rank-1: RX ← RX − ΔW_:,j ⊗ H_j,:
        A[:, j] = 0.0
        avail[j] = False
        order.append(j)
        if recompute_every and (step + 1) % recompute_every == 0:
            RX = A @ Hf                                      # exact refresh kills drift
    return order


def greedy_independent_order(
    delta_w: "Any",
    H: "Any",
    k: int,
) -> List[int]:
    """Independent (first-step) column ranking — the interaction-free ablation.

    Ranks columns by the greedy objective's *marginal* gains evaluated ONCE, with
    no iterative residual update::

        G_j⁰ = 2·⟨ΔW_:,j, (ΔW H)_:,j⟩ − ‖ΔW_:,j‖²·H_jj

    This uses the full Hessian in the score but ignores how protecting one column
    changes the value of protecting another. Comparing this against
    :func:`greedy_select_channels` at identical actual weight bits isolates exactly
    the contribution of the iterative cross-column interactions (the paper's
    novelty claim). Returns up to ``k`` channels with positive gain, in descending
    gain order; the top pick coincides with greedy's first pick.
    """
    import torch

    if delta_w.ndim != 2:
        raise ValueError(f"delta_w must be [out, in], got {tuple(delta_w.shape)}")
    in_f = int(delta_w.shape[1])
    if in_f == 0 or k <= 0:
        return []
    k = min(int(k), in_f)
    A = delta_w.detach().to(dtype=torch.float64)
    Hf = H.detach().to(dtype=torch.float64, device=A.device)
    if not bool(torch.isfinite(A).all()) or not bool(torch.isfinite(Hf).all()):
        A = torch.nan_to_num(A)
        Hf = torch.nan_to_num(Hf)
    Hdiag = torch.diagonal(Hf)
    RX = A @ Hf                                              # [out, in]
    dot = (A * RX).sum(dim=0)                                # [in]
    nrm = (A * A).sum(dim=0)                                 # [in]
    G = 2.0 * dot - nrm * Hdiag                              # [in]
    G = torch.where(torch.isfinite(G), G, torch.full_like(G, float("-inf")))
    vals, idx = torch.sort(G, descending=True)
    # keep only positive-gain columns (mirrors greedy's "protect only if it helps")
    order = [int(j) for v, j in zip(vals.tolist(), idx.tolist()) if v > 0.0][:k]
    return order


def greedy_protected_map(
    weights: "Any",
    bases: "Any",
    hessians: "Any",
    k_by_layer: "Any",
    *,
    mode: str = "greedy",
    device: str = "cuda",
    logger: "Any" = None,
) -> dict:
    """Run greedy selection for many layers -> ``{layer: [protected idx]}``.

    ``weights``/``bases`` map layer name -> weight tensor [out, in] (FP16 ``W`` and
    the quantized base ``Wq``); ``hessians`` maps layer name -> (H [in,in], count)
    as returned by ``gptq.collect_gptq_hessians``; ``k_by_layer`` maps layer name
    -> number of channels to protect. ``mode`` selects ``"greedy"`` (interaction-
    aware, iterative) or ``"independent"`` (first-step gains, no update). Layers
    missing a base or Hessian are skipped.
    """
    import torch

    select = greedy_select_channels if mode == "greedy" else greedy_independent_order
    out: dict = {}
    for name, w in weights.items():
        wq = bases.get(name) if hasattr(bases, "get") else None
        hess = hessians.get(name) if hasattr(hessians, "get") else None
        k = int(k_by_layer.get(name, 0)) if hasattr(k_by_layer, "get") else 0
        if wq is None or hess is None or k <= 0:
            continue
        H = hess[0] if isinstance(hess, (tuple, list)) else hess
        wf = w.detach().to(device=device, dtype=torch.float32)
        wqf = wq.detach().to(device=device, dtype=torch.float32)
        if wqf.shape != wf.shape and wqf.t().shape == wf.shape:
            wqf = wqf.t().contiguous()
        if wqf.shape != wf.shape:
            if logger is not None:
                logger.warning("greedy: base/weight shape mismatch for %s; skipping", name)
            continue
        try:
            order = select(wf - wqf, H.to(device), k)
        except Exception as exc:  # noqa: BLE001
            if logger is not None:
                logger.warning("greedy: selection failed for %s: %s", name, exc)
            continue
        # Preserve *selection priority* order: callers slice ``order[:k']`` to get
        # the top-k' greedy set for a smaller fraction. Sorting by index here would
        # make every prefix protect the lowest-index channels instead of the most
        # important ones (ChannelProtectedLinear sorts the final set internally).
        out[name] = list(order)
    return out
