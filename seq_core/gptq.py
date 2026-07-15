#!/usr/bin/env python3
"""GPTQ base quantization (error-compensated) for SEQ.

Run 6 showed round-to-nearest bases (bitsandbytes / HQQ) are the Pareto ceiling:
4-bit already sits ~1.4 PPL above FP16 and sub-4-bit collapses, and per-channel
protection cannot close that. GPTQ compensates quantization error using the
layer's input Hessian, so a GPTQ 4-bit base is far closer to FP16 — the base a
competitive Pareto needs.

This is a faithful, dependency-free reimplementation of the GPTQ algorithm
(Frantar et al. 2022; mirrors auto-gptq's `fasterquant`), producing **fake-
quantized** weights (dequantized fp16) — exact for PPL measurement and directly
usable by the per-channel protection correction. It supports arbitrary bits and
group sizes.

Two steps:
- ``collect_gptq_hessians``: one calibration pass accumulating H = XᵀX per Linear.
- ``gptq_quantize_weight`` / ``gptq_quantize_model``: the error-compensated quant.

Memory: H is [in, in] per layer; ``gptq_quantize_model`` frees each H right after
use, but the returned fake-quant weights are full fp16 — run on 1B/3B first.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence, Tuple

import torch

LOGGER = logging.getLogger(__name__)


def collect_gptq_hessians(
    model: torch.nn.Module,
    tokenizer,
    prompts: Sequence[str],
    *,
    seq_len: int,
    device: str,
    max_prompts: Optional[int] = None,
    hessian_device: str = "cpu",
) -> Dict[str, Tuple[torch.Tensor, int]]:
    """Accumulate the input Hessian H = XᵀX per Linear over calibration data.

    Hessians are stored on ``hessian_device`` (default ``"cpu"``). Holding every
    layer's [in, in] Hessian on the GPU simultaneously OOMs large models (8B down
    projections alone are ~37 GiB); accumulating on CPU bounds GPU use to the
    model plus one transient XᵀX, and ``gptq_quantize_model`` moves each Hessian
    to the GPU only when it quantizes that layer.
    """
    accs: Dict[str, list] = {}
    handles = []
    hdev = torch.device(hessian_device)

    def make_hook(name: str):
        def pre_hook(_module, inputs):
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).to(dtype=torch.float32)
            mask = torch.isfinite(x).all(dim=1)
            x = x[mask]
            if x.numel() == 0:
                return
            xtx = (x.t() @ x).to(hdev)  # compute on the activation's device, store on hdev
            if name not in accs:
                accs[name] = [torch.zeros(x.shape[1], x.shape[1], device=hdev, dtype=torch.float32), 0]
            accs[name][0].add_(xtx)
            accs[name][1] += int(x.shape[0])
        return pre_hook

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_pre_hook(make_hook(name)))

    used = list(prompts) if max_prompts is None else list(prompts)[: int(max_prompts)]
    with torch.no_grad():
        for prompt in used:
            if not isinstance(prompt, str) or not prompt.strip():
                continue
            # NO padding: the Hessian H = XᵀX sums over all positions, so padding
            # every short prompt to seq_len would make H ~97% pad-token statistics
            # and GPTQ would compensate for the wrong distribution (garbage base).
            # Batch=1 natural length = real tokens only.
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=seq_len)
            enc = {k: v.to(next(model.parameters()).device) for k, v in enc.items()}
            model(**enc)
    for h in handles:
        h.remove()
    return {name: (H, n) for name, (H, n) in accs.items()}


def _find_params(x: torch.Tensor, maxq: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row asymmetric affine params over a column group. x: [out, g]."""
    xmax = x.max(dim=1).values.clamp(min=0)
    xmin = x.min(dim=1).values.clamp(max=0)
    scale = (xmax - xmin) / maxq
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    zero = torch.round(-xmin / scale)
    return scale, zero


def _quantize_affine(w: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, maxq: float) -> torch.Tensor:
    """Fake-quantize a column w [out] with per-row scale/zero -> dequantized [out]."""
    q = torch.clamp(torch.round(w / scale) + zero, 0, maxq)
    return scale * (q - zero)


def gptq_quantize_weight(
    weight: torch.Tensor,
    H: torch.Tensor,
    bits: int,
    *,
    group_size: int = 128,
    percdamp: float = 0.01,
    blocksize: int = 128,
) -> torch.Tensor:
    """GPTQ fake-quantized weight [out, in] using input Hessian H [in, in].

    Faithful to auto-gptq: damped inverse Cholesky of H drives per-column error
    compensation; per-(row, group) asymmetric quantization.
    """
    W = weight.detach().to(dtype=torch.float32).clone()
    out_f, in_f = W.shape
    maxq = float(2 ** int(bits) - 1)
    H = H.detach().to(dtype=torch.float32).clone()

    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0

    damp = percdamp * torch.mean(torch.diag(H))
    diag_idx = torch.arange(in_f, device=W.device)
    H[diag_idx, diag_idx] += damp

    # Hinv = upper Cholesky of H^{-1}
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    Q = torch.zeros_like(W)
    gs = int(group_size)
    scale = zero = None
    if gs == -1:  # per-tensor (single group over all columns)
        scale, zero = _find_params(W, maxq)
    for i1 in range(0, in_f, blocksize):
        i2 = min(i1 + blocksize, in_f)
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(count):
            col = i1 + i
            w = W1[:, i]
            d = Hinv1[i, i]
            if gs != -1 and col % gs == 0:
                g = W[:, col:min(col + gs, in_f)]
                scale, zero = _find_params(g, maxq)
            q = _quantize_affine(w, scale, zero, maxq)
            Q1[:, i] = q
            err = (w - q) / d
            W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        if i2 < in_f:
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return Q


def gptq_quantize_model(
    model: torch.nn.Module,
    tokenizer,
    prompts: Sequence[str],
    *,
    bits: int,
    group_size: int = 128,
    seq_len: int,
    device: str,
    max_prompts: Optional[int] = None,
    percdamp: float = 0.01,
    skip: Optional[Sequence[str]] = None,
    out_dtype: torch.dtype = torch.float16,
    out_device: str = "cpu",
    hessian_device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Return ``{layer_name: fake_quantized_weight}`` for every Linear.

    NB: this quantizes each layer independently from the FP16 activations (no
    sequential re-capture between layers). That is the standard "one-shot" GPTQ
    setup used for calibration-cheap PTQ and is sufficient here. Hessians live on
    ``hessian_device`` (CPU by default) and are moved to the layer's device only
    while that layer is quantized, then freed — so GPU peak is model + one
    Hessian, not all of them.
    """
    accs = collect_gptq_hessians(
        model, tokenizer, prompts, seq_len=seq_len, device=device,
        max_prompts=max_prompts, hessian_device=hessian_device,
    )
    skip_set = set(skip or [])
    result: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or name in skip_set or name not in accs:
            continue
        H, n = accs[name]
        try:
            H_dev = H.to(module.weight.device)  # bring just this layer's Hessian to the GPU
            Wq = gptq_quantize_weight(module.weight, H_dev, bits, group_size=group_size, percdamp=percdamp)
            # store off the GPU by default: the whole fake-quant model would
            # otherwise coexist with each reloaded model during the sweep (OOM on 8B)
            result[name] = Wq.to(dtype=out_dtype, device=torch.device(out_device))
            del H_dev, Wq
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("gptq: layer %s failed (%s); leaving FP16", name, exc)
        finally:
            accs[name] = None  # free the CPU Hessian immediately
            del H
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    LOGGER.info("gptq: quantized %d layers to %d-bit", len(result), bits)
    return result
