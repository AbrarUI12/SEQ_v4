#!/usr/bin/env python3
"""Per-module and per-channel importance signals for SEQ.

This generalizes the current *module-level entropy* signal into a family of
signals that can be scored against a ground-truth sensitivity ranking
(``seq_core/sensitivity.py``) to answer:

    RQ1: how good is the entropy signal, and at what granularity?
    RQ2: do magnitude / kurtosis / activation-scale / Hessian-diagonal predict
         quantization sensitivity better?

Granularity for a Linear with ``weight`` of shape ``[out_features, in_features]``:

- ``module``       one scalar for the whole matrix.
- ``out_channel``  one value per output row  (reduces over inputs).
- ``in_channel``   one value per input column (reduces over outputs). This is
                   the granularity that matters for weight quantization because
                   activation outliers hit specific *input* channels.

Weight-only signals (entropy, magnitude, kurtosis, outlier fraction) need only
the weights. Activation-aware signals (act-scale, Hessian-diagonal, salience)
accumulate per-input-channel statistics over calibration data via forward-pre
hooks — the same inputs GPTQ/AWQ use.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch

LOGGER = logging.getLogger(__name__)

WEIGHT_SIGNALS = ("entropy", "magnitude", "kurtosis", "outlier_frac")
# Extensive (sum over the matrix) vs. per-parameter (``_pp``, intensive) forms.
# Extensive predicts *whole-module* ΔLoss; ``_pp`` is comparable across module
# sizes (see docs/FINDINGS_run1.md — the extensive forms rank by size / flag lm_head).
ACT_SIGNALS = ("act_scale", "hessian_diag", "hessian_diag_pp", "salience", "salience_pp")
ALL_SIGNALS = WEIGHT_SIGNALS + ACT_SIGNALS


# --------------------------------------------------------------------------- #
# Row-wise (batched) weight statistics. Each helper takes a [U, N] matrix and
# returns a length-U vector; "module" scalars pass the flattened weight as [1,N].
# --------------------------------------------------------------------------- #
def _row_entropy(mat: torch.Tensor, bins: int, clip: float, eps: float) -> torch.Tensor:
    """Shannon entropy (bits) of a per-row z-standardized histogram."""
    mat = mat.to(dtype=torch.float32)
    mean = mat.mean(dim=1, keepdim=True)
    std = mat.std(dim=1, unbiased=False, keepdim=True)
    z = torch.clamp((mat - mean) / (std + eps), -clip, clip)
    edges = torch.linspace(-clip, clip, bins + 1, device=mat.device, dtype=torch.float32)
    idx = torch.clamp(torch.bucketize(z, edges, right=False) - 1, 0, bins - 1)
    counts = torch.zeros((mat.shape[0], bins), device=mat.device, dtype=torch.float32)
    counts.scatter_add_(1, idx, torch.ones_like(z))
    probs = counts / counts.sum(dim=1, keepdim=True).clamp_min(1.0)
    ent = -(probs * torch.log2(probs.clamp_min(1e-12))).sum(dim=1)
    nan = torch.full_like(ent, float("nan"))
    return torch.where(std.squeeze(1) < eps, nan, ent)


def _row_magnitude(mat: torch.Tensor) -> torch.Tensor:
    return mat.to(dtype=torch.float32).abs().mean(dim=1)


def _row_kurtosis(mat: torch.Tensor, eps: float) -> torch.Tensor:
    """Excess kurtosis per row (0 for Gaussian; >0 heavy-tailed / outliers)."""
    mat = mat.to(dtype=torch.float32)
    mean = mat.mean(dim=1, keepdim=True)
    diff = mat - mean
    var = (diff * diff).mean(dim=1)
    m4 = (diff.pow(4)).mean(dim=1)
    return m4 / (var * var + eps) - 3.0


def _row_outlier_frac(mat: torch.Tensor, eps: float, k: float) -> torch.Tensor:
    """Fraction of entries with |z| > k after per-row standardization."""
    mat = mat.to(dtype=torch.float32)
    mean = mat.mean(dim=1, keepdim=True)
    std = mat.std(dim=1, unbiased=False, keepdim=True)
    z = (mat - mean) / (std + eps)
    return (z.abs() > k).to(dtype=torch.float32).mean(dim=1)


def _scalar(x: torch.Tensor) -> float:
    try:
        return float(x.reshape(-1)[0].item())
    except Exception:
        return float("nan")


def _vec(x: torch.Tensor) -> List[float]:
    return [float(v) for v in x.detach().to(dtype=torch.float32).cpu().tolist()]


# --------------------------------------------------------------------------- #
# Weight signals
# --------------------------------------------------------------------------- #
def weight_signals_for_matrix(
    weight: torch.Tensor,
    *,
    bins: int = 256,
    clip: float = 6.0,
    eps: float = 1e-5,
    outlier_k: float = 4.0,
    return_channels: bool = False,
) -> Dict[str, Any]:
    """Compute all weight-only signals for one [out, in] matrix, all granularities."""
    w = weight.detach().to(dtype=torch.float32)
    flat = w.reshape(1, -1)
    out: Dict[str, Any] = {}

    # module-level scalars (computed directly on the flattened matrix)
    out["entropy"] = {"module": _scalar(_row_entropy(flat, bins, clip, eps))}
    out["magnitude"] = {"module": _scalar(_row_magnitude(flat))}
    out["kurtosis"] = {"module": _scalar(_row_kurtosis(flat, eps))}
    out["outlier_frac"] = {"module": _scalar(_row_outlier_frac(flat, eps, outlier_k))}

    if return_channels:
        wt = w.t().contiguous()  # [in, out] -> rows are input channels
        out["entropy"]["out_channel"] = _vec(_row_entropy(w, bins, clip, eps))
        out["entropy"]["in_channel"] = _vec(_row_entropy(wt, bins, clip, eps))
        out["magnitude"]["out_channel"] = _vec(_row_magnitude(w))
        out["magnitude"]["in_channel"] = _vec(_row_magnitude(wt))
        out["kurtosis"]["out_channel"] = _vec(_row_kurtosis(w, eps))
        out["kurtosis"]["in_channel"] = _vec(_row_kurtosis(wt, eps))
        out["outlier_frac"]["out_channel"] = _vec(_row_outlier_frac(w, eps, outlier_k))
        out["outlier_frac"]["in_channel"] = _vec(_row_outlier_frac(wt, eps, outlier_k))
    return out


def extract_weight_signals(
    model: torch.nn.Module,
    *,
    bins: int = 256,
    clip: float = 6.0,
    eps: float = 1e-5,
    outlier_k: float = 4.0,
    return_channels: bool = False,
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not hasattr(module, "weight") or module.weight is None:
            continue
        if module.weight.numel() == 0:
            continue
        results[name] = weight_signals_for_matrix(
            module.weight,
            bins=bins,
            clip=clip,
            eps=eps,
            outlier_k=outlier_k,
            return_channels=return_channels,
        )
    return results


# --------------------------------------------------------------------------- #
# Activation-aware signals: per-input-channel stats over calibration data.
# We hook the *input* of each Linear (forward_pre_hook) — the X that multiplies
# W and drives quantization sensitivity (GPTQ diag(XᵀX), AWQ mean|x|).
# --------------------------------------------------------------------------- #
@dataclass
class _InputAccumulator:
    in_features: int
    device: torch.device
    sum_abs: torch.Tensor = field(default=None)  # type: ignore[assignment]
    sum_sq: torch.Tensor = field(default=None)  # type: ignore[assignment]
    count: int = 0

    def __post_init__(self) -> None:
        self.sum_abs = torch.zeros(self.in_features, dtype=torch.float64, device=self.device)
        self.sum_sq = torch.zeros(self.in_features, dtype=torch.float64, device=self.device)

    def update(self, x: torch.Tensor) -> None:
        flat = x.detach().reshape(-1, self.in_features).to(dtype=torch.float64)
        mask = torch.isfinite(flat).all(dim=1)
        flat = flat[mask]
        if flat.numel() == 0:
            return
        self.sum_abs += flat.abs().sum(dim=0)
        self.sum_sq += (flat * flat).sum(dim=0)
        self.count += int(flat.shape[0])

    def act_scale(self) -> torch.Tensor:  # E|x| per input channel  (AWQ)
        c = max(1, self.count)
        return (self.sum_abs / c).to(dtype=torch.float32)

    def act_sq(self) -> torch.Tensor:  # E[x^2] per input channel  (Hessian diag)
        c = max(1, self.count)
        return (self.sum_sq / c).to(dtype=torch.float32)


def collect_input_stats(
    model: torch.nn.Module,
    tokenizer,
    prompts: Sequence[str],
    *,
    seq_len: int,
    device: str,
    max_prompts: Optional[int] = None,
) -> Dict[str, _InputAccumulator]:
    """One calibration pass accumulating per-input-channel activation moments."""
    model.eval()
    accs: Dict[str, _InputAccumulator] = {}
    handles = []
    dev = torch.device(device)

    def make_hook(name: str):
        def pre_hook(_module, inputs):
            if not inputs:
                return
            x = inputs[0]
            if not isinstance(x, torch.Tensor):
                return
            if name not in accs:
                accs[name] = _InputAccumulator(in_features=x.shape[-1], device=dev)
            accs[name].update(x.to(dev))
        return pre_hook

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_pre_hook(make_hook(name)))

    used = list(prompts) if max_prompts is None else list(prompts)[: int(max_prompts)]
    with torch.no_grad():
        for prompt in used:
            if not isinstance(prompt, str) or not prompt.strip():
                continue
            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=seq_len,
            )
            enc = {k: v.to(next(model.parameters()).device) for k, v in enc.items()}
            model(**enc)

    for h in handles:
        h.remove()
    return accs


def activation_signals_for_module(
    weight: torch.Tensor,
    acc: _InputAccumulator,
    *,
    eps: float = 1e-12,
    return_channels: bool = False,
) -> Dict[str, Any]:
    """Combine per-input-channel activation moments with weights.

    Hessian-diag sensitivity (2nd-order, GPTQ):  s_ij ≈ E[x_j²]·w_ij²
    Salience (weight-aware AWQ):                  q_j  ≈ E|x_j|·‖w_:,j‖₂
    """
    w = weight.detach().to(dtype=torch.float32)
    if w.device != acc.act_sq().device:
        act_sq = acc.act_sq().to(w.device)
        act_scale = acc.act_scale().to(w.device)
    else:
        act_sq = acc.act_sq()
        act_scale = acc.act_scale()

    w2 = w * w                                  # [out, in]
    col_w2 = w2.sum(dim=0)                       # [in]  ‖w_:,j‖²
    hess = w2 * act_sq.unsqueeze(0)              # [out, in]
    sal_in = act_scale * torch.sqrt(col_w2 + eps)  # [in]

    out: Dict[str, Any] = {
        "act_scale": {"module": float(act_scale.mean().item())},
        # extensive: total whole-module second-order ΔLoss / salience
        "hessian_diag": {"module": float(hess.sum().item())},
        "salience": {"module": float(sal_in.sum().item())},
        # per-parameter (intensive): comparable across module sizes
        "hessian_diag_pp": {"module": float(hess.mean().item())},
        "salience_pp": {"module": float(sal_in.mean().item())},
    }
    if return_channels:
        out["act_scale"]["in_channel"] = _vec(act_scale)
        out["hessian_diag"]["in_channel"] = _vec(act_sq * col_w2)
        out["hessian_diag"]["out_channel"] = _vec(hess.sum(dim=1))
        out["salience"]["in_channel"] = _vec(sal_in)
    return out


def extract_activation_signals(
    model: torch.nn.Module,
    tokenizer,
    prompts: Sequence[str],
    *,
    seq_len: int,
    device: str,
    max_prompts: Optional[int] = None,
    return_channels: bool = False,
) -> Dict[str, Dict[str, Any]]:
    accs = collect_input_stats(
        model, tokenizer, prompts, seq_len=seq_len, device=device, max_prompts=max_prompts
    )
    results: Dict[str, Dict[str, Any]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        acc = accs.get(name)
        if acc is None:
            continue
        results[name] = activation_signals_for_module(
            module.weight, acc, return_channels=return_channels
        )
    return results


# --------------------------------------------------------------------------- #
# Top-level: all signals as module-level scalar table (for correlation study).
# --------------------------------------------------------------------------- #
def extract_all_signals(
    model: torch.nn.Module,
    tokenizer=None,
    prompts: Optional[Sequence[str]] = None,
    *,
    seq_len: int = 2048,
    device: str = "cuda",
    bins: int = 256,
    clip: float = 6.0,
    eps: float = 1e-5,
    outlier_k: float = 4.0,
    max_prompts: Optional[int] = None,
    include_activation: bool = True,
    return_channels: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Return ``{module_name: {signal_name: {granularity: value}}}``.

    Weight signals are always computed. Activation signals require a tokenizer
    and calibration prompts.
    """
    table = extract_weight_signals(
        model, bins=bins, clip=clip, eps=eps, outlier_k=outlier_k, return_channels=return_channels
    )
    if include_activation and tokenizer is not None and prompts:
        act = extract_activation_signals(
            model,
            tokenizer,
            prompts,
            seq_len=seq_len,
            device=device,
            max_prompts=max_prompts,
            return_channels=return_channels,
        )
        for name, sig in act.items():
            table.setdefault(name, {}).update(sig)
    else:
        LOGGER.info("Activation signals skipped (need tokenizer + prompts).")
    return table


def module_scalar_table(
    signals: Dict[str, Dict[str, Any]],
    granularity: str = "module",
) -> Dict[str, Dict[str, float]]:
    """Flatten to ``{signal_name: {module_name: scalar}}`` for correlation.

    For channel granularities, each module is summarized by the mean of its
    per-channel values so it can be correlated against a module-level ground
    truth. (Channel-vs-channel correlation is handled separately.)
    """
    out: Dict[str, Dict[str, float]] = {}
    for module_name, sig in signals.items():
        for signal_name, by_gran in sig.items():
            if not isinstance(by_gran, dict):
                continue
            if granularity in by_gran:
                val = by_gran[granularity]
                if isinstance(val, list):
                    finite = [v for v in val if isinstance(v, (int, float)) and math.isfinite(v)]
                    scalar = sum(finite) / len(finite) if finite else float("nan")
                else:
                    scalar = float(val)
            elif "module" in by_gran:
                scalar = float(by_gran["module"])
            else:
                continue
            out.setdefault(signal_name, {})[module_name] = scalar
    return out
