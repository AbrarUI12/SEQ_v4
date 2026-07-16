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

Memory: H is [in, in] per layer. The one-shot path stores Hessians on CPU and
moves them to the weight device individually. The sequential path additionally
offloads inactive decoder blocks and calibration activations to CPU. Returned
fake-quant weights are still full fp16 and therefore require substantial RAM.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence, Tuple

import torch

LOGGER = logging.getLogger(__name__)


class _StopForward(Exception):
    pass


_CACHE_KWARGS = {"past_key_value", "past_key_values"}


def _is_transformers_cache(value) -> bool:
    """Recognize HF cache objects without importing a version-specific class."""
    if value is None:
        return False
    cls = type(value)
    if cls.__module__.startswith("transformers.cache_utils"):
        return True
    return (
        callable(getattr(value, "update", None))
        and callable(getattr(value, "get_seq_length", None))
    )


def _move_replay_value(value, device: torch.device):
    """Detach tensor trees for replay and discard mutable attention caches."""
    if _is_transformers_cache(value):
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {key: _move_replay_value(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        moved = tuple(_move_replay_value(item, device) for item in value)
        if hasattr(value, "_fields"):  # preserve namedtuple types
            return type(value)(*moved)
        return moved
    if isinstance(value, list):
        return [_move_replay_value(item, device) for item in value]
    return value


def _prepare_replay_call(args, kwargs, device: torch.device):
    """Make an independent, cache-free copy of one decoder-block call."""
    clean_kwargs = {key: value for key, value in kwargs.items() if key not in _CACHE_KWARGS}
    if "use_cache" in clean_kwargs:
        clean_kwargs["use_cache"] = False
    return (
        _move_replay_value(args, device),
        _move_replay_value(clean_kwargs, device),
    )


def _find_decoder_layers(model: torch.nn.Module):
    """Locate the ModuleList of transformer blocks and its dotted name prefix."""
    import torch.nn as nn

    for path in ("model.layers", "model.model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"):
        obj = model
        ok = True
        for p in path.split("."):
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                ok = False
                break
        if ok and isinstance(obj, nn.ModuleList) and len(obj) > 0:
            return obj, path
    # fallback: the largest ModuleList whose blocks contain Linears
    best = None
    best_len = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > best_len:
            if any(isinstance(m, nn.Linear) for m in mod[0].modules()):
                best, best_len = (mod, name), len(mod)
    if best is not None:
        return best[0], best[1]
    raise RuntimeError("could not locate decoder layers for sequential GPTQ")


@torch.no_grad()
def gptq_quantize_model_sequential(
    model: torch.nn.Module,
    tokenizer,
    prompts,
    *,
    bits: int,
    group_size: int = 128,
    seq_len: int,
    device: str,
    max_prompts: Optional[int] = 32,
    percdamp: float = 0.01,
    skip: Optional[Sequence[str]] = None,
    out_dtype: torch.dtype = torch.float16,
    out_device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Sequential GPTQ: quantize block-by-block, feeding each block's *quantized*
    output to the next so Hessians reflect the true (quantized) input distribution.

    This is the correct fix for the one-shot failure (which miscalibrates because
    upstream layers change the activations). Weights are replaced in-place as we
    go; returns ``{layer_name: fake_quant_weight}`` for the protection step.

    Calibration activations and their per-sample arguments live on CPU. On CUDA,
    decoder blocks are also offloaded and processed one at a time. Attention
    caching is forcibly disabled, because replaying a mutable KV cache across
    independent samples makes the effective sequence length grow without bound.
    """
    import torch.nn as nn

    skip_set = set(skip or [])
    layers, prefix = _find_decoder_layers(model)
    dev = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    cpu = torch.device("cpu")
    replay: list = []
    result: Dict[str, torch.Tensor] = {}
    original_use_cache = getattr(getattr(model, "config", None), "use_cache", None)
    original_devices = []
    for block in layers:
        param = next(block.parameters(), None)
        original_devices.append(param.device if param is not None else dev)

    # Keeping an 8B model resident defeats sequential GPTQ's memory bound. The
    # first-block catcher stops before any decoder computation, so all blocks can
    # be on CPU during capture and only the active block needs to return to CUDA.
    offload_blocks = dev.type == "cuda"
    if offload_blocks:
        for block in layers:
            block.to(cpu)
        torch.cuda.empty_cache()

    class _Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, hidden_states, *args, **kwargs):  # noqa: D401
            replay_args, replay_kwargs = _prepare_replay_call(args, kwargs, cpu)
            replay.append((hidden_states.detach().to(cpu), replay_args, replay_kwargs))
            raise _StopForward

    try:
        if original_use_cache is not None:
            model.config.use_cache = False

        # ---- capture first-block inputs and each sample's own call metadata - #
        first_block = layers[0]
        layers[0] = _Catcher(first_block)
        try:
            used = list(prompts) if max_prompts is None else list(prompts)[: int(max_prompts)]
            for prompt in used:
                if not isinstance(prompt, str) or not prompt.strip():
                    continue
                enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=seq_len)
                enc = {key: value.to(dev) for key, value in enc.items()}
                try:
                    model(**enc, use_cache=False)
                except _StopForward:
                    pass
        finally:
            layers[0] = first_block

        if not replay:
            raise ValueError("sequential GPTQ captured no usable calibration samples")
        LOGGER.info(
            "seq-gptq: captured %d samples on CPU; processing %d decoder blocks",
            len(replay), len(layers),
        )

        # ---- accumulate, quantize, and advance one decoder block at a time -- #
        for i, block in enumerate(layers):
            LOGGER.info("seq-gptq: block %d/%d", i + 1, len(layers))
            if offload_blocks:
                block.to(dev)
            sub = {n: m for n, m in block.named_modules() if isinstance(m, nn.Linear)}
            accs: Dict[str, list] = {}

            def make_hook(nm):
                def pre_hook(_m, inputs):
                    if not inputs or not isinstance(inputs[0], torch.Tensor):
                        return
                    x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).to(torch.float32)
                    finite_rows = torch.isfinite(x).all(dim=1)
                    if not bool(finite_rows.all()):
                        x = x[finite_rows]
                    if x.numel() == 0:
                        return
                    if nm not in accs:
                        accs[nm] = [
                            torch.zeros(x.shape[1], x.shape[1], device=x.device, dtype=torch.float32),
                            0,
                        ]
                    accs[nm][0].add_(x.t() @ x)
                    accs[nm][1] += int(x.shape[0])
                return pre_hook

            handles = []
            for name, module in sub.items():
                if f"{prefix}.{i}.{name}" not in skip_set:
                    handles.append(module.register_forward_pre_hook(make_hook(name)))
            try:
                for hidden_states, cpu_args, cpu_kwargs in replay:
                    args = _move_replay_value(cpu_args, dev)
                    kwargs = _move_replay_value(cpu_kwargs, dev)
                    block(hidden_states.to(dev), *args, **kwargs)
            finally:
                for handle in handles:
                    handle.remove()

            # Free persistent block Hessians from CUDA before Cholesky. Each one
            # is transferred back individually for quantization below.
            if dev.type == "cuda":
                for entry in accs.values():
                    entry[0] = entry[0].to(cpu)
                torch.cuda.empty_cache()

            for name, lin in sub.items():
                full = f"{prefix}.{i}.{name}"
                if full in skip_set or name not in accs:
                    continue
                H_dev = None
                Wq = None
                try:
                    H_dev = accs[name][0].to(lin.weight.device)
                    Wq = gptq_quantize_weight(
                        lin.weight, H_dev, bits, group_size=group_size,
                        percdamp=percdamp, clone_hessian=False,
                    )
                    lin.weight.data.copy_(Wq)
                    result[full] = Wq.to(dtype=out_dtype, device=torch.device(out_device))
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("seq-gptq: layer %s failed (%s)", full, exc)
                finally:
                    accs[name] = None
                    del H_dev, Wq
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()

            # Re-run the quantized block and retain only CPU outputs for the
            # next block. Per-sample args/kwargs remain independent and on CPU.
            next_replay = []
            for hidden_states, cpu_args, cpu_kwargs in replay:
                args = _move_replay_value(cpu_args, dev)
                kwargs = _move_replay_value(cpu_kwargs, dev)
                out = block(hidden_states.to(dev), *args, **kwargs)
                next_hidden = out[0] if isinstance(out, (tuple, list)) else out
                next_replay.append((next_hidden.detach().to(cpu), cpu_args, cpu_kwargs))
            replay = next_replay

            if offload_blocks:
                block.to(cpu)
                torch.cuda.empty_cache()

    finally:
        if original_use_cache is not None:
            model.config.use_cache = original_use_cache
        for block, original_device in zip(layers, original_devices):
            param = next(block.parameters(), None)
            if param is not None and param.device != original_device:
                block.to(original_device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    LOGGER.info("seq-gptq: quantized %d layers to %d-bit across %d blocks", len(result), bits, len(layers))
    return result


def build_gptq_calibration(
    tokenizer,
    *,
    n_samples: int = 128,
    seq_len: int = 2048,
    dataset_name: str = "wikitext2",
    split: str = "train",
    seed: int = 1234,
) -> list:
    """Standard GPTQ calibration: ``n_samples`` contiguous ``seq_len``-token chunks
    of real text (decoded back to strings for the tokenizer-based collector).

    GPTQ inverts a per-layer Hessian H = XᵀX of size [in, in] (in up to ~14k);
    with too few tokens H is rank-deficient and the inverse is garbage. This
    yields ~n_samples*seq_len tokens (e.g. 128*2048 = 262k), enough for full rank.
    """
    import random as _random

    from benchmarks.core import _load_text_dataset

    ds, field = _load_text_dataset(dataset_name, split)
    text = "\n\n".join(t for t in ds[field] if isinstance(t, str) and t.strip())
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
    n = int(ids.shape[0])
    rng = _random.Random(seed)
    texts = []
    for _ in range(int(n_samples)):
        if n <= seq_len:
            chunk = ids
        else:
            start = rng.randint(0, n - seq_len - 1)
            chunk = ids[start:start + seq_len]
        texts.append(tokenizer.decode(chunk, skip_special_tokens=True))
    return texts


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
    clone_hessian: bool = True,
) -> torch.Tensor:
    """GPTQ fake-quantized weight [out, in] using input Hessian H [in, in].

    Faithful to auto-gptq: damped inverse Cholesky of H drives per-column error
    compensation; per-(row, group) asymmetric quantization. Set
    ``clone_hessian=False`` only when the caller owns and can discard ``H``;
    this avoids retaining two large Hessian buffers during Cholesky.
    """
    W = weight.detach().to(dtype=torch.float32).clone()
    out_f, in_f = W.shape
    maxq = float(2 ** int(bits) - 1)
    H = H.detach().to(dtype=torch.float32)
    if clone_hessian:
        H = H.clone()

    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0

    damp = percdamp * torch.mean(torch.diag(H))
    diag_idx = torch.arange(in_f, device=W.device)
    H[diag_idx, diag_idx] += damp

    # Hinv = upper Cholesky of H^{-1}
    L = torch.linalg.cholesky(H)
    del H
    Hinv = torch.cholesky_inverse(L)
    del L
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
            Wq = gptq_quantize_weight(
                module.weight, H_dev, bits, group_size=group_size,
                percdamp=percdamp, clone_hessian=False,
            )
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
