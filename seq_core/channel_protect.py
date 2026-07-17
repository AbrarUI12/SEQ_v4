#!/usr/bin/env python3
"""Per-channel (input-column) mixed precision — the pivot after run 3.

Run 3 showed module-level allocation cannot beat uniform. Within a layer,
uniform is *not* the ceiling: a small fraction of input channels (outlier
features) genuinely need more precision (LLM.int8 / AWQ). We test that directly.

For a Linear ``W`` of shape ``[out, in]`` we keep the top-k% input channels
(chosen by a per-input-channel signal) in FP16 and quantize the remaining
columns to ``base_bits`` with the backend — the LLM.int8-style column split::

    y = Q(W[:, rest]) · x[rest]  +  W[:, prot] · x[prot]  +  b

The decisive comparison is signal-chosen vs. random-chosen protected channels
at the same k: if the signal wins, per-channel importance is real.

``select_protected_channels`` and ``layer_effective_bits`` are pure functions
(unit-tested); ``ChannelProtectedLinear`` realizes the split for eval.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .channel_utils import layer_effective_bits, select_protected_channels
from .quantizers.base import QuantBackend, get_module_by_name, set_module_by_name, tag_quantized

LOGGER = logging.getLogger(__name__)


def _quantize_columns(cols: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-input-column symmetric quantization of a [out, k] weight block to ``bits``."""
    qmax = float(2 ** (int(bits) - 1) - 1)  # 127 for 8-bit
    if qmax < 1:
        qmax = 1.0
    scale = cols.abs().amax(dim=0, keepdim=True).clamp_min(1e-8) / qmax
    return torch.round(cols / scale).clamp(-qmax, qmax) * scale


class ChannelProtectedLinear(nn.Module):
    """Full-layer quantization + exact FP16 correction on protected columns.

    The whole weight is quantized at ``base_bits`` (identical to a plain
    uniform run, always group-size-divisible), then protected input columns are
    restored to FP16 via a low-rank correction ``C = (W − dequant(Q(W)))[:, prot]``::

        y = Q(W)·x + x[prot] · Cᵀ + b

    This avoids sub-matrix group-size issues and keeps the base numerically
    identical to the uniform baseline, isolating the effect of protection.
    ``num_protected`` reflects channels *actually* corrected (0 if the backend
    cannot dequantize).
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        protected_idx: Sequence[int],
        backend: Optional[QuantBackend],
        base_bits: int,
        *,
        device: str,
        compute_dtype: torch.dtype,
        group_size: Optional[int] = 64,
        precomputed_base: Optional[torch.Tensor] = None,
        tiers: Optional[Dict[int, Sequence[int]]] = None,
        **backend_kwargs: Any,
    ) -> None:
        super().__init__()
        out_f, in_f = int(weight.shape[0]), int(weight.shape[1])
        self.in_features, self.out_features = in_f, out_f
        self.base_bits = int(base_bits)
        # tiers: {protect_bits: [channel idx]}. Single-tier FP16 protection is the
        # default (protected_idx -> the 16-bit tier).
        if tiers is None:
            tiers = {16: sorted(set(int(i) for i in protected_idx))} if len(protected_idx) else {}
        else:
            tiers = {int(b): sorted(set(int(i) for i in idx)) for b, idx in tiers.items() if len(idx)}
        w = weight.detach()

        # Base = a precomputed fake-quant weight (e.g. GPTQ) or a data-free
        # backend quantization (HQQ/bnb). Both expose a dequantized weight for
        # the protection correction.
        if precomputed_base is not None:
            base_w = precomputed_base.detach().to(device=device, dtype=compute_dtype)
            full = nn.Linear(in_f, out_f, bias=False)
            full.weight = nn.Parameter(base_w.contiguous(), requires_grad=False)
            self.q_full = full.to(device)
            dequant = base_w
        else:
            if backend is None:
                raise ValueError("ChannelProtectedLinear needs a backend or precomputed_base")
            full = nn.Linear(in_f, out_f, bias=False)
            full.weight = nn.Parameter(w.contiguous(), requires_grad=False)
            self.q_full = backend.quantize_linear(
                full, int(base_bits), device=device, compute_dtype=compute_dtype,
                group_size=group_size, **backend_kwargs,
            )
            dequant = backend.dequantize_weight(self.q_full)

        # Build one correction per tier: for a channel protected at ``b`` bits,
        # restore it from the base to a b-bit representation via a residual
        # C_b = (Q_b(W) − Q_base(W))[:, idx].  b=16 is exact FP16 (Q_16 = W).
        self._corr_names = []  # list of (idx_buffer, corr_buffer, bits)
        self.tier_counts: Dict[int, int] = {}
        wq = dequant
        if tiers and wq is not None:
            wq = wq.to(device=device, dtype=torch.float32)
            wf = w.to(device=device, dtype=torch.float32)
            if wq.shape != wf.shape and wq.t().shape == wf.shape:
                wq = wq.t().contiguous()
            if wq.shape == wf.shape:
                for ti, (bits, idx_list) in enumerate(sorted(tiers.items(), key=lambda kv: -kv[0])):
                    if not idx_list:
                        continue
                    idx = torch.tensor(idx_list, dtype=torch.long, device=device)
                    target = wf.index_select(1, idx)
                    if int(bits) < 16:
                        target = _quantize_columns(target, int(bits))
                    corr = (target - wq.index_select(1, idx)).to(compute_dtype)
                    self.register_buffer(f"corr_idx_{ti}", idx)
                    self.register_buffer(f"corr_w_{ti}", corr.contiguous())
                    self._corr_names.append((f"corr_idx_{ti}", f"corr_w_{ti}"))
                    self.tier_counts[int(bits)] = self.tier_counts.get(int(bits), 0) + len(idx_list)
            elif tiers:
                LOGGER.warning("channel protect: dequant shape mismatch; protection disabled for a layer")
        elif tiers and wq is None:
            LOGGER.warning("channel protect: backend cannot dequantize; protection disabled for a layer")

        self.num_protected = sum(self.tier_counts.values())
        self.bias = (
            nn.Parameter(bias.detach().to(device=device, dtype=compute_dtype), requires_grad=False)
            if bias is not None else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.q_full(x)
        for iname, wname in self._corr_names:
            idx = getattr(self, iname)
            corr = getattr(self, wname)
            xp = x.index_select(-1, idx).to(corr.dtype)
            y = y + torch.matmul(xp, corr.t()).to(y.dtype)
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y


def apply_channel_protection(
    model: nn.Module,
    layer_channel_scores: Dict[str, Sequence[float]],
    k_frac: float,
    backend: Optional[QuantBackend],
    base_bits: int,
    *,
    device: str,
    compute_dtype: torch.dtype,
    group_size: Optional[int] = 64,
    skip: Optional[Sequence[str]] = None,
    protect_bits: int = 16,
    explicit_protected: Optional[Dict[str, Sequence[int]]] = None,
    precomputed_base: Optional[Dict[str, torch.Tensor]] = None,
    tier_fracs: Optional[List[tuple]] = None,
    explicit_tiers: Optional[Dict[str, Dict[int, Sequence[int]]]] = None,
    **backend_kwargs: Any,
) -> Dict[str, Any]:
    """Replace each scored Linear with a column-split protected version.

    ``explicit_protected`` protects exactly the given channel indices per layer
    (audit / greedy selection). ``explicit_tiers`` assigns exact per-layer
    precision tiers ``{bits: [idx]}`` (value-based bit allocation). ``precomputed_base``
    supplies a per-layer fake-quant base weight (e.g. GPTQ). ``tier_fracs`` =
    [(bits, frac), ...] enables percentage-based multi-precision protection (top
    frac -> highest bits, etc.). Precedence: explicit_tiers > tier_fracs >
    explicit_protected > top-k by score. Returns effective bits.
    """
    from .channel_utils import assign_tiers, layer_effective_bits_tiered
    skip_set = set(skip or [])
    total_params = 0
    weighted_bits = 0.0
    errors: List[Dict[str, Any]] = []
    per_layer: Dict[str, Dict[str, Any]] = {}

    for name, scores in layer_channel_scores.items():
        if name in skip_set:
            continue
        try:
            module = get_module_by_name(model, name)
        except Exception as exc:  # noqa: BLE001
            errors.append({"module": name, "error": f"lookup: {exc}"})
            continue
        if not isinstance(module, nn.Linear):
            continue
        in_f = module.in_features
        out_f = module.out_features
        tiers = None
        prot = []
        if explicit_tiers is not None:
            tiers = {int(b): list(idx) for b, idx in (explicit_tiers.get(name) or {}).items() if len(idx)} or None
        elif tier_fracs:
            tiers = assign_tiers(scores, tier_fracs)
        elif explicit_protected is not None:
            prot = list(explicit_protected.get(name, []))
        else:
            prot = select_protected_channels(scores, k_frac)
        base_w = precomputed_base.get(name) if precomputed_base else None
        # a layer with no precomputed base (e.g. skipped by GPTQ) stays FP16-quantized
        # by the backend if available; if neither, skip it.
        if base_w is None and backend is None:
            continue
        try:
            new_module = ChannelProtectedLinear(
                module.weight, module.bias, prot, backend, base_bits,
                device=device, compute_dtype=compute_dtype, group_size=group_size,
                precomputed_base=base_w, tiers=tiers,
                **backend_kwargs,
            )
            tag_quantized(new_module, base_bits, f"{backend.name}+chprot")
            set_module_by_name(model, name, new_module)
        except Exception as exc:  # noqa: BLE001
            errors.append({"module": name, "error": str(exc)})
            continue
        # account for channels the module *actually* corrected (dequant may disable)
        if new_module.tier_counts and any(b != 16 for b in new_module.tier_counts):
            eff = layer_effective_bits_tiered(in_f, new_module.tier_counts, base_bits)
        else:
            eff = layer_effective_bits(in_f, new_module.num_protected, base_bits, protect_bits)
        params = in_f * out_f + (out_f if module.bias is not None else 0)
        total_params += params
        weighted_bits += eff * params
        per_layer[name] = {"in_features": in_f, "num_protected": new_module.num_protected,
                           "tier_counts": dict(new_module.tier_counts), "effective_bits": eff}

    effective_bits = weighted_bits / total_params if total_params else float(base_bits)
    if errors:
        LOGGER.warning("apply_channel_protection: %d layer(s) failed", len(errors))
    return {
        "k_frac": k_frac,
        "base_bits": base_bits,
        "effective_bits": effective_bits,
        "num_layers": len(per_layer),
        "errors": errors,
        "per_layer": per_layer,
    }
