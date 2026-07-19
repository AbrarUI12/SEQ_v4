#!/usr/bin/env python3
"""Load a saved LightCompress (LLMC) fake-quant model as a per-layer base.

From-scratch GPTQ (``seq_core/gptq.py``) never produced a working full-model base
(≈5 attempts, garbage k=0 PPL) and cannot be debugged in a no-GPU environment, so
the strong-base path uses **LightCompress**, whose GPTQ works (Llama-3.2-1B W4g128
≈ 10.39 PPL). Run LightCompress with ``save_fake`` (via ``run_compare_matrix.py
--llmc_save_mode fake``) and it writes a standard ``transformers`` checkpoint whose
Linear weights are already fake-quantized (dequantized GPTQ values).

This module reloads that checkpoint and returns ``{layer_name: weight}`` so
``channel_sweep`` can pass it as ``precomputed_base`` to ``apply_channel_protection``
— exactly the reuse point the protection path already supports. The protection
sweep (FP16 residual columns, greedy selection, tier allocation) then runs on top
of the GPTQ base, so we can measure the decisive comparison: does protection
improve a GPTQ base over plain GPTQ-4 at matched actual bits?
"""
from __future__ import annotations

import gc
import logging
from typing import Any, Dict, Optional, Sequence

import torch


def _load_state_tensors(model_path: str) -> Optional[Dict[str, torch.Tensor]]:
    """Load raw LLMC tensors, including GPTQ qparam buffers.

    ``transformers.from_pretrained`` intentionally drops the unexpected
    ``buf_*`` tensors written by LLMC.  Those buffers are required to reproduce
    the fake-quant forward pass, so read the safetensors checkpoint directly
    when available.
    """
    try:
        from safetensors.torch import load_file
        import glob, json, os
        files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        index = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.isfile(index):
            mapping = json.load(open(index, encoding="utf-8")).get("weight_map", {})
            files = sorted({os.path.join(model_path, f) for f in mapping.values()})
        if not files:
            return None
        out: Dict[str, torch.Tensor] = {}
        for f in files:
            out.update(load_file(f, device="cpu"))
        return out
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("could not read raw safetensors from %s: %s", model_path, exc)
        return None


def _dequantize_llmc_gptq(weight: torch.Tensor, state: Dict[str, torch.Tensor], name: str) -> torch.Tensor:
    """Reproduce LLMC GPTQ ``w_qdq`` for one [out,in] linear weight."""
    prefix = name + "."
    scales = state.get(prefix + "buf_scales")
    zeros = state.get(prefix + "buf_zeros")
    qmin = state.get(prefix + "buf_qmin")
    qmax = state.get(prefix + "buf_qmax")
    perm = state.get(prefix + "buf_perm")
    invperm = state.get(prefix + "buf_invperm")
    if scales is None or qmin is None or qmax is None:
        return weight
    groups = int(weight.shape[1] // 128)
    if groups <= 0 or int(weight.shape[1]) % 128:
        return weight
    wp = weight
    if perm is not None:
        wp = wp.index_select(1, perm.to(torch.long))
    flat = wp.reshape(-1, 128).float()
    sc = scales.reshape(-1, 1).float()
    ze = zeros.reshape(-1, 1).float() if zeros is not None else torch.zeros_like(sc)
    if sc.shape[0] != flat.shape[0]:
        return weight
    q = torch.round(flat / sc.clamp_min(1e-9) + ze)
    q = q.clamp(float(qmin.item()), float(qmax.item()))
    out = ((q - ze) * sc).reshape_as(wp).to(weight.dtype)
    if invperm is not None:
        out = out.index_select(1, invperm.to(torch.long))
    return out

LOGGER = logging.getLogger(__name__)


def load_llmc_fake_quant_base(
    model_path: str,
    in_features: Dict[str, int],
    *,
    skip: Optional[Sequence[str]] = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float16,
    trust_remote_code: bool = False,
) -> Dict[str, torch.Tensor]:
    """Return ``{layer_name: fake-quant weight}`` from a saved LightCompress model.

    ``in_features`` is the target model's ``{Linear name: in_features}`` map; only
    layers whose name is present there (and not in ``skip``), with a matching
    ``in_features``, are returned — so a shape/name mismatch is dropped with a
    warning rather than silently corrupting the base. Weights are detached onto
    ``device`` (default CPU, to avoid doubling GPU memory) and the reloaded model
    is freed before returning.
    """
    from transformers import AutoModelForCausalLM

    raw_state = _load_state_tensors(model_path)

    skip_set = set(skip or [])
    try:
        saved = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=bool(trust_remote_code)
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"could not load LightCompress fake-quant model from '{model_path}': {exc}. "
            "Produce it with run_compare_matrix.py --methods gptq_llmc --llmc_save_mode fake."
        ) from exc
    saved.eval()

    base: Dict[str, torch.Tensor] = {}
    missing_shape = 0
    for name, module in saved.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if name in skip_set or name not in in_features:
            continue
        w = module.weight
        if w is None:
            continue
        if int(w.shape[1]) != int(in_features[name]):
            missing_shape += 1
            LOGGER.warning("llmc base: in_features mismatch for %s (%d vs %d); skipping",
                           name, int(w.shape[1]), int(in_features[name]))
            continue
        if raw_state is not None and name + ".weight" in raw_state:
            raw_w = raw_state[name + ".weight"]
            w = _dequantize_llmc_gptq(raw_w, raw_state, name)
        base[name] = w.detach().to(device=device, dtype=dtype).contiguous()

    matched = len(base)
    target = len([n for n in in_features if n not in skip_set])
    LOGGER.info("llmc base: matched %d/%d target Linear layers (%d shape mismatches)",
                matched, target, missing_shape)
    if matched == 0:
        raise RuntimeError(
            f"no layers matched between the LightCompress model and the target model; "
            f"check that '{model_path}' is the same architecture/model."
        )
    if matched < target:
        LOGGER.warning("llmc base: %d target layers had no fake-quant weight (left to backend/FP16)",
                       target - matched)

    del saved
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return base
