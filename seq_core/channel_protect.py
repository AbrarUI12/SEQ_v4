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
        backend: QuantBackend,
        base_bits: int,
        *,
        device: str,
        compute_dtype: torch.dtype,
        group_size: Optional[int] = 64,
        **backend_kwargs: Any,
    ) -> None:
        super().__init__()
        out_f, in_f = int(weight.shape[0]), int(weight.shape[1])
        self.in_features, self.out_features = in_f, out_f
        self.base_bits = int(base_bits)
        prot = sorted(set(int(i) for i in protected_idx))
        w = weight.detach()

        full = nn.Linear(in_f, out_f, bias=False)
        full.weight = nn.Parameter(w.contiguous(), requires_grad=False)
        self.q_full = backend.quantize_linear(
            full, int(base_bits), device=device, compute_dtype=compute_dtype,
            group_size=group_size, **backend_kwargs,
        )

        self._has_correction = False
        if prot:
            wq = backend.dequantize_weight(self.q_full)
            if wq is not None:
                wq = wq.to(device=device, dtype=torch.float32)
                wf = w.to(device=device, dtype=torch.float32)
                if wq.shape != wf.shape and wq.t().shape == wf.shape:
                    wq = wq.t().contiguous()
                if wq.shape == wf.shape:
                    idx = torch.tensor(prot, dtype=torch.long, device=device)
                    corr = (wf.index_select(1, idx) - wq.index_select(1, idx)).to(compute_dtype)
                    self.register_buffer("protected_idx", idx)
                    self.register_buffer("correction", corr.contiguous())  # [out, k]
                    self._has_correction = True
                else:
                    LOGGER.warning("channel protect: dequant shape mismatch; protection disabled for a layer")
            else:
                LOGGER.warning("channel protect: backend cannot dequantize; protection disabled for a layer")

        self.num_protected = len(prot) if self._has_correction else 0
        self.bias = (
            nn.Parameter(bias.detach().to(device=device, dtype=compute_dtype), requires_grad=False)
            if bias is not None else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.q_full(x)
        if self._has_correction:
            xp = x.index_select(-1, self.protected_idx).to(self.correction.dtype)
            y = y + torch.matmul(xp, self.correction.t()).to(y.dtype)
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y


def apply_channel_protection(
    model: nn.Module,
    layer_channel_scores: Dict[str, Sequence[float]],
    k_frac: float,
    backend: QuantBackend,
    base_bits: int,
    *,
    device: str,
    compute_dtype: torch.dtype,
    group_size: Optional[int] = 64,
    skip: Optional[Sequence[str]] = None,
    protect_bits: int = 16,
    explicit_protected: Optional[Dict[str, Sequence[int]]] = None,
    **backend_kwargs: Any,
) -> Dict[str, Any]:
    """Replace each scored Linear with a column-split protected version.

    If ``explicit_protected`` is given, each named layer protects exactly those
    channel indices (``k_frac``/scores ignored) — used by the sensitivity audit
    to protect a specific rank bucket. Returns effective-bit accounting.
    """
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
        if explicit_protected is not None:
            prot = list(explicit_protected.get(name, []))
        else:
            prot = select_protected_channels(scores, k_frac)
        try:
            new_module = ChannelProtectedLinear(
                module.weight, module.bias, prot, backend, base_bits,
                device=device, compute_dtype=compute_dtype, group_size=group_size,
                **backend_kwargs,
            )
            tag_quantized(new_module, base_bits, f"{backend.name}+chprot")
            set_module_by_name(model, name, new_module)
        except Exception as exc:  # noqa: BLE001
            errors.append({"module": name, "error": str(exc)})
            continue
        # account for channels the module *actually* corrected (dequant may disable)
        eff = layer_effective_bits(in_f, new_module.num_protected, base_bits, protect_bits)
        params = in_f * out_f + (out_f if module.bias is not None else 0)
        total_params += params
        weighted_bits += eff * params
        per_layer[name] = {"in_features": in_f, "num_protected": len(prot), "effective_bits": eff}

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
