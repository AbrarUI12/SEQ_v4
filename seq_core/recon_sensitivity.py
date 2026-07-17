#!/usr/bin/env python3
"""Local reconstruction-error sensitivity — a deterministic ground truth.

Run 1 (docs/FINDINGS_run1.md) showed the PPL-based one-hot degrade is
noise-dominated at module level. The standard, noise-free alternative is the
*local reconstruction error* — the GPTQ/AWQ objective — measured with the real
calibration activations:

    E‖(W − Q(W)) x‖²  ≈  Σ_j  E[x_j²] · ‖ΔW_:,j‖²       (diagonal in x)

where ΔW = W − dequant(quant(W)) is the *real* quantizer error (outliers, group
size and all), not the ``w²`` proxy the signals use. This gives, in one
calibration pass plus one quantize per module and **no PPL eval loop**:

- a per-module sensitivity (sum over input channels), and
- a per-input-channel sensitivity vector — the granularity RQ1 asks for,

both deterministic. Correlating the signals against this is a fair test of the
``w²`` proxy against the true error, and it scales to channels.
"""
from __future__ import annotations

import copy
import gc
import logging
from typing import Any, Dict, List, Optional, Sequence

import torch

from .quantizers.base import QuantBackend, get_module_by_name
from .signals import collect_input_stats
from .stats_utils import aligned_correlation

LOGGER = logging.getLogger(__name__)


def _free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def reconstruction_sensitivity(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    backend: QuantBackend,
    *,
    bits: int,
    group_size: Optional[int] = 64,
    device: str = "cuda",
    compute_dtype: torch.dtype = torch.float16,
    seq_len: int = 2048,
    max_prompts: Optional[int] = 64,
    module_names: Optional[Sequence[str]] = None,
    return_channels: bool = False,
    **backend_kwargs: Any,
) -> Dict[str, Any]:
    """Per-module (and optional per-channel) reconstruction sensitivity at ``bits``."""
    accs = collect_input_stats(
        model, tokenizer, prompts, seq_len=seq_len, device=device, max_prompts=max_prompts
    )
    names = list(module_names) if module_names is not None else [
        n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
    ]

    per_unit: Dict[str, Dict[str, Any]] = {}
    skipped: List[str] = []
    for name in names:
        module = get_module_by_name(model, name)
        acc = accs.get(name)
        if acc is None or not isinstance(module, torch.nn.Linear):
            continue
        try:
            clone = copy.deepcopy(module)
            q = backend.quantize_linear(
                clone, bits, device=device, compute_dtype=compute_dtype,
                group_size=group_size, **backend_kwargs,
            )
            wq = backend.dequantize_weight(q)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("recon: quantize %s failed: %s", name, exc)
            skipped.append(name)
            continue
        if wq is None:
            skipped.append(name)
            continue

        w = module.weight.detach().to(dtype=torch.float32)
        wq = wq.to(dtype=torch.float32, device=w.device)
        if wq.shape != w.shape:
            if wq.t().shape == w.shape:
                wq = wq.t().contiguous()
            else:
                LOGGER.warning("recon: shape mismatch for %s (%s vs %s)", name, tuple(wq.shape), tuple(w.shape))
                skipped.append(name)
                continue
        dw = w - wq                                    # [out, in]  real quant error
        col_err = (dw * dw).sum(dim=0)                 # [in]  ‖ΔW_:,j‖²
        act_sq = acc.act_sq().to(w.device)             # [in]  E[x_j²]
        per_in = act_sq * col_err                      # [in]  diagonal recon sensitivity
        row: Dict[str, Any] = {
            "module": float(per_in.sum().item()),
            "weight_mse": float((dw * dw).mean().item()),
            "act_weighted": float(per_in.sum().item()),
            "bits": int(bits),
        }
        if return_channels:
            row["in_channel"] = [float(v) for v in per_in.detach().cpu().tolist()]
        per_unit[name] = row
        del clone, q
        _free()

    if skipped:
        LOGGER.warning("recon: %d modules skipped (no dequant / failure)", len(skipped))
    return {
        "mode": "reconstruction",
        "bits": int(bits),
        "backend": backend.name,
        "num_units": len(per_unit),
        "skipped": skipped,
        "per_unit": per_unit,
    }


def channel_residual_scores(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    backend: QuantBackend,
    *,
    bits: int,
    group_size: Optional[int] = 64,
    device: str = "cuda",
    compute_dtype: torch.dtype = torch.float16,
    seq_len: int = 2048,
    max_prompts: Optional[int] = 64,
    precomputed_base: Optional[Dict[str, torch.Tensor]] = None,
    module_names: Optional[Sequence[str]] = None,
    **backend_kwargs: Any,
) -> Dict[str, Dict[str, List[float]]]:
    """Per-input-channel *residual-aware* selection scores against the built base.

    Unlike the scalar activation signals (which look only at ``x`` or ``W``), these
    weight each channel by the **real** quantization error it carries, ``ΔW = W − Wq``:

        residual_rms_j = E[x_j²] · ‖ΔW_:,j‖²             (mean-energy weighted)
        residual_max_j = (max_t |x_{t,j}|)² · ‖ΔW_:,j‖²  (worst-case / outlier weighted)

    ``Wq`` is the precomputed base (``gptq`` / ``gptq_llmc``) when supplied for the
    layer, else the backend's dequantized ``bits``-bit quantization (``hqq``). The
    activation moments reuse ``collect_input_stats`` — the same pass and padding as
    the ``act_max`` signal, so the comparison is apples-to-apples. Returns
    ``{layer: {"residual_rms": [...], "residual_max": [...]}}`` (per input channel).
    """
    accs = collect_input_stats(
        model, tokenizer, prompts, seq_len=seq_len, device=device, max_prompts=max_prompts
    )
    names = list(module_names) if module_names is not None else [
        n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
    ]
    base = precomputed_base or {}
    out: Dict[str, Dict[str, List[float]]] = {}
    skipped: List[str] = []
    for name in names:
        module = get_module_by_name(model, name)
        acc = accs.get(name)
        if acc is None or not isinstance(module, torch.nn.Linear):
            continue
        w = module.weight.detach().to(dtype=torch.float32)
        # ΔW: prefer the precomputed base for this layer; else quantize with backend.
        if name in base:
            wq = base[name].detach().to(dtype=torch.float32, device=w.device)
        else:
            try:
                clone = copy.deepcopy(module)
                q = backend.quantize_linear(
                    clone, bits, device=device, compute_dtype=compute_dtype,
                    group_size=group_size, **backend_kwargs,
                )
                wq = backend.dequantize_weight(q)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("residual: quantize %s failed: %s", name, exc)
                skipped.append(name)
                continue
            if wq is None:
                skipped.append(name)
                continue
            wq = wq.to(dtype=torch.float32, device=w.device)
            del clone, q
        if wq.shape != w.shape:
            if wq.t().shape == w.shape:
                wq = wq.t().contiguous()
            else:
                LOGGER.warning("residual: shape mismatch for %s (%s vs %s)", name, tuple(wq.shape), tuple(w.shape))
                skipped.append(name)
                continue
        dw = w - wq                                    # [out, in]  real quant error
        col_err = (dw * dw).sum(dim=0)                 # [in]  ‖ΔW_:,j‖²
        act_sq = acc.act_sq().to(w.device)             # [in]  E[x_j²]
        act_max = acc.act_max().to(w.device)           # [in]  max_t |x_{t,j}|
        residual_rms = (act_sq * col_err)              # [in]
        residual_max = (act_max * act_max * col_err)   # [in]
        out[name] = {
            "residual_rms": [float(v) for v in residual_rms.detach().cpu().tolist()],
            "residual_max": [float(v) for v in residual_max.detach().cpu().tolist()],
        }
        _free()
    if skipped:
        LOGGER.warning("residual: %d modules skipped (no dequant / failure)", len(skipped))
    return out


def ground_truth_scores(result: Dict[str, Any], key: str = "module") -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, row in result.get("per_unit", {}).items():
        val = row.get(key)
        if val is not None:
            out[name] = float(val)
    return out


def correlate_signals(
    signal_scalars: Dict[str, Dict[str, float]],
    ground_truth: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Rank signals by Spearman rho against reconstruction sensitivity (desc)."""
    rows: List[Dict[str, Any]] = []
    for signal_name, scores in signal_scalars.items():
        rows.append({"signal": signal_name, **aligned_correlation(scores, ground_truth)})
    rows.sort(key=lambda r: (r["spearman"] if r["spearman"] is not None else float("-inf")), reverse=True)
    return rows
