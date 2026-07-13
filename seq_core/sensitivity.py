#!/usr/bin/env python3
"""Ground-truth quantization-sensitivity harness (the keystone of RQ1/RQ2).

We *measure* how much each unit matters by perturbing it and watching PPL:

- ``one_hot_degrade``: from FP16, quantize only unit ``u`` to ``bits``, keep the
  rest FP16, record ``ΔPPL_u``. Ranking units by ΔPPL is the ground truth a good
  signal should reproduce.
- ``one_hot_protect``: from an all-quantized model, restore only ``u`` to FP16,
  record the PPL recovered — the marginal value of protecting ``u``.

Both use a single loaded model and a **swap-and-restore** loop: deepcopy the
original Linear, quantize the copy, evaluate, then put the pristine original
back. That keeps everything else untouched and is robust to backends that free
the source weights (e.g. HQQ ``del_orig``).

Candidate signals (``seq_core/signals.py``) are then rank-correlated against the
measured sensitivity with ``seq_core/stats_utils.py``.
"""
from __future__ import annotations

import copy
import gc
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

from .quantizers.base import QuantBackend, get_module_by_name, set_module_by_name
from .stats_utils import aligned_correlation

LOGGER = logging.getLogger(__name__)

PplFn = Callable[[torch.nn.Module, Any], float]


def make_ppl_fn(
    *,
    dataset_name: str = "wikitext2",
    split: str = "test",
    seq_len: int = 2048,
    device: str = "cuda",
    dtype: Optional[torch.dtype] = None,
    mode: str = "canonical",
    max_examples: Optional[int] = None,
    full_corpus: Optional[bool] = None,
    seed: int = 1234,
) -> PplFn:
    """Wrap ``benchmarks.core.compute_ppl`` into ``ppl_fn(model, tokenizer)->float``.

    A ``proxy`` mode with a small ``max_examples`` is recommended for the
    per-unit sweep (hundreds of evals); use ``canonical`` for final numbers.
    """
    from benchmarks.core import compute_ppl  # lazy to avoid heavy import at load

    def ppl_fn(model: torch.nn.Module, tokenizer: Any) -> float:
        result = compute_ppl(
            model,
            tokenizer,
            dataset_name=dataset_name,
            split=split,
            seq_len=seq_len,
            max_examples=max_examples,
            device=device,
            dtype=dtype,
            seed=seed,
            mode=mode,
            full_corpus=full_corpus,
        )
        ppl = result.get("ppl") if isinstance(result, dict) else None
        return float(ppl) if ppl is not None else float("nan")

    return ppl_fn


def _free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def linear_module_names(model: torch.nn.Module) -> List[str]:
    return [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]


def _quantize_copy(
    model: torch.nn.Module,
    name: str,
    bits: int,
    backend: QuantBackend,
    *,
    device: str,
    compute_dtype: torch.dtype,
    group_size: Optional[int],
    **backend_kwargs: Any,
) -> torch.nn.Module:
    """Quantize a deepcopy of module ``name`` and swap it in; return the original."""
    original = get_module_by_name(model, name)
    clone = copy.deepcopy(original)
    q = backend.quantize_linear(
        clone,
        bits,
        device=device,
        compute_dtype=compute_dtype,
        group_size=group_size,
        **backend_kwargs,
    )
    set_module_by_name(model, name, q)
    return original


def one_hot_degrade(
    model: torch.nn.Module,
    tokenizer: Any,
    ppl_fn: PplFn,
    backend: QuantBackend,
    *,
    bits: int,
    module_names: Optional[Sequence[str]] = None,
    device: str = "cuda",
    compute_dtype: torch.dtype = torch.float16,
    group_size: Optional[int] = 64,
    baseline_ppl: Optional[float] = None,
    progress_every: int = 10,
    **backend_kwargs: Any,
) -> Dict[str, Any]:
    """ΔPPL when only one module is dropped to ``bits`` (others FP16)."""
    names = list(module_names) if module_names is not None else linear_module_names(model)
    if baseline_ppl is None:
        baseline_ppl = ppl_fn(model, tokenizer)
        LOGGER.info("baseline FP16 ppl = %.4f", baseline_ppl)

    per_unit: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(names):
        try:
            original = _quantize_copy(
                model, name, bits, backend,
                device=device, compute_dtype=compute_dtype,
                group_size=group_size, **backend_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("skip %s: quantize failed: %s", name, exc)
            per_unit[name] = {"ppl": float("nan"), "delta_ppl": float("nan"), "error": str(exc)}
            continue
        try:
            ppl = ppl_fn(model, tokenizer)
        finally:
            set_module_by_name(model, name, original)  # restore pristine original
            _free()
        per_unit[name] = {
            "ppl": ppl,
            "delta_ppl": ppl - baseline_ppl,
            "rel_delta_ppl": (ppl - baseline_ppl) / baseline_ppl if baseline_ppl else float("nan"),
            "bits": int(bits),
        }
        if progress_every and (i + 1) % progress_every == 0:
            LOGGER.info("degrade %d/%d done (last Δppl=%.4f)", i + 1, len(names), per_unit[name]["delta_ppl"])

    return {
        "mode": "one_hot_degrade",
        "bits": int(bits),
        "backend": backend.name,
        "baseline_ppl": baseline_ppl,
        "per_unit": per_unit,
    }


def one_hot_protect(
    model: torch.nn.Module,
    tokenizer: Any,
    ppl_fn: PplFn,
    backend: QuantBackend,
    *,
    bits: int,
    module_names: Optional[Sequence[str]] = None,
    device: str = "cuda",
    compute_dtype: torch.dtype = torch.float16,
    group_size: Optional[int] = 64,
    progress_every: int = 10,
    **backend_kwargs: Any,
) -> Dict[str, Any]:
    """PPL recovered when only one module is restored to FP16 in an all-``bits`` model.

    Implemented by swap-restore in reverse: for each ``u`` we temporarily restore
    ``u`` to FP16 while all others are quantized, and measure the drop from the
    all-quantized baseline. Requires a saved FP16 copy of every module first.
    """
    names = list(module_names) if module_names is not None else linear_module_names(model)

    # Save pristine FP16 originals, then quantize everything in place.
    originals: Dict[str, torch.nn.Module] = {n: copy.deepcopy(get_module_by_name(model, n)) for n in names}
    for name in names:
        try:
            q = backend.quantize_linear(
                copy.deepcopy(originals[name]), bits,
                device=device, compute_dtype=compute_dtype,
                group_size=group_size, **backend_kwargs,
            )
            set_module_by_name(model, name, q)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("protect-setup: quantize %s failed: %s", name, exc)
    _free()
    baseline_all_ppl = ppl_fn(model, tokenizer)
    LOGGER.info("all-%dbit ppl = %.4f", bits, baseline_all_ppl)

    per_unit: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(names):
        quantized_here = get_module_by_name(model, name)
        set_module_by_name(model, name, originals[name])  # restore this one to FP16
        try:
            ppl = ppl_fn(model, tokenizer)
        finally:
            set_module_by_name(model, name, quantized_here)  # put quantized back
            _free()
        per_unit[name] = {
            "ppl": ppl,
            "recovered_ppl": baseline_all_ppl - ppl,
            "bits": int(bits),
        }
        if progress_every and (i + 1) % progress_every == 0:
            LOGGER.info("protect %d/%d done", i + 1, len(names))

    return {
        "mode": "one_hot_protect",
        "bits": int(bits),
        "backend": backend.name,
        "baseline_all_ppl": baseline_all_ppl,
        "per_unit": per_unit,
    }


# --------------------------------------------------------------------------- #
# Correlate candidate signals against measured sensitivity.
# --------------------------------------------------------------------------- #
def ground_truth_scores(result: Dict[str, Any], key: str = "delta_ppl") -> Dict[str, float]:
    """Extract ``{module: sensitivity}`` from a harness result."""
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
    """Rank signals by how well they predict measured sensitivity.

    ``signal_scalars`` == ``signals.module_scalar_table(...)`` output:
    ``{signal_name: {module_name: value}}``.
    Returns a list sorted by Spearman rho (desc), most predictive first.
    """
    rows: List[Dict[str, Any]] = []
    for signal_name, scores in signal_scalars.items():
        corr = aligned_correlation(scores, ground_truth)
        rows.append({"signal": signal_name, **corr})
    rows.sort(
        key=lambda r: (r["spearman"] if r["spearman"] is not None else float("-inf")),
        reverse=True,
    )
    return rows
