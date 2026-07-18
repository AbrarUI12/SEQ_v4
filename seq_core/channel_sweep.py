#!/usr/bin/env python3
"""Per-channel protection sweep: does protecting signal-chosen channels beat random?

For a base bit-width and a set of protection fractions k, quantize each layer's
weight columns to ``base_bits`` while keeping the top-k% input channels (by a
per-input-channel signal) in FP16, then measure real PPL. The decisive line is
signal vs. `random` at the *same* k (same effective bits): if the signal wins,
per-channel importance is real (the positive result run 3 sent us to find).

Run on a GPU box. Example::

    python -m seq_core.channel_sweep \
        --model meta-llama/Llama-3.2-1B --backend hqq \
        --base_bits 4 --protect_fracs 0,0.02,0.05,0.1,0.2 \
        --signals act_scale,hessian_diag,salience,magnitude,random \
        --ppl_mode canonical --calibration_prompts calibration_prompts.json \
        --out_dir runs/channel/Llama-3.2-1B
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger("channel_sweep")


def _load_prompts(path: Optional[str]) -> List[str]:
    if not path:
        return []
    from seq_core.pipeline import load_prompts

    return load_prompts(Path(path))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SEQ per-channel protection sweep")
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--calibration_prompts", default="calibration_prompts.json")
    p.add_argument("--calib_seq_len", type=int, default=2048)
    p.add_argument("--max_calib_prompts", type=int, default=64)

    p.add_argument("--backend", default="hqq")
    p.add_argument("--group_size", type=int, default=64)
    p.add_argument("--base_bits", type=int, default=4)
    p.add_argument("--protect_bits", type=int, default=16, help="16=FP16 columns, or 8 for int8 columns")
    p.add_argument("--protect_fracs", default="0,0.02,0.05,0.1,0.2")
    p.add_argument("--protect_tiers", default="",
                   help="multi-precision protection instead of --protect_fracs; ';'-separated "
                        "configs, each 'bits:frac,...' e.g. '16:0.02,8:0.08;16:0.01,8:0.04'")
    p.add_argument("--signals", default="act_scale,act_max,act_kurt,hessian_diag,magnitude,random",
                   help="scalar per-channel signals; also accepts 'residual_rms'/'residual_max' "
                        "(HQQ/GPTQ-residual-aware) which are computed against the built base")
    p.add_argument("--select", default="topk", choices=["topk", "greedy"],
                   help="topk = protect the top-k channels by signal (per-channel independent); "
                        "greedy = OMP residual-reduction selector (interaction-aware, ignores --signals)")
    p.add_argument("--tier_alloc", default="", choices=["", "value"],
                   help="'value' = allocate protection bits by error-per-byte (greedy benefit/cost) "
                        "across {base,8,16} instead of fixed --protect_tiers percentages")
    p.add_argument("--channel_entropy", action="store_true",
                   help="also compute per-channel activation entropy as signal 'act_entropy' (memory-heavy; try 1B/3B first)")
    p.add_argument("--entropy_bins", type=int, default=32)
    p.add_argument("--skip_lm_head", action="store_true")
    p.add_argument("--base_quantizer", default="hqq", choices=["hqq", "gptq", "gptq_llmc"],
                   help="base quantizer under the protection: hqq (data-free RTN), gptq "
                        "(from-scratch, diagnostics only), gptq_llmc (load a saved LightCompress "
                        "fake-quant model — the working strong base for the decisive comparison)")
    p.add_argument("--gptq_model_path", default="",
                   help="for --base_quantizer gptq_llmc: directory of the saved LightCompress "
                        "fake-quant HF model (produced with --llmc_save_mode fake)")
    p.add_argument("--gptq_group_size", type=int, default=128)
    p.add_argument("--gptq_percdamp", type=float, default=0.01)
    p.add_argument("--gptq_calib_samples", type=int, default=128,
                   help="GPTQ needs many tokens for a full-rank Hessian; build this many "
                        "seq_len chunks of real text (0 = reuse --calibration_prompts, usually too small)")
    p.add_argument("--gptq_mode", default="sequential", choices=["sequential", "oneshot"],
                   help="sequential (correct: feeds each block's quantized output forward) or "
                        "oneshot (broken for full models; kept for diagnostics)")
    p.add_argument("--gptq_hessian_device", default="cpu", choices=["cpu", "cuda"],
                   help="cpu (default) accumulates Hessians off-GPU so 3B/8B don't OOM; cuda is faster for 1B")
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--ppl_mode", default="canonical", choices=["proxy", "canonical"])
    p.add_argument("--ppl_dataset", default="wikitext2")
    p.add_argument("--ppl_seq_len", type=int, default=2048)
    p.add_argument("--ppl_max_examples", type=int, default=64)
    p.add_argument("--out_dir", default="runs/channel")
    p.add_argument("--save_model_path", default="",
                   help="optional Hugging Face directory for one selected fake-quant evaluation checkpoint")
    p.add_argument("--save_signal", default="",
                   help="signal label to export (required with --save_model_path)")
    p.add_argument("--save_k_frac", type=float, default=None,
                   help="protection fraction to export (required with --save_model_path)")
    return p.parse_args()


def _in_channel_scores(signals: Dict[str, Any], signal_name: str) -> Dict[str, List[float]]:
    """Pull per-layer per-input-channel arrays for a signal from extract_all_signals."""
    out: Dict[str, List[float]] = {}
    for layer, sig in signals.items():
        by = sig.get(signal_name)
        if isinstance(by, dict) and isinstance(by.get("in_channel"), list):
            out[layer] = by["in_channel"]
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    if args.save_model_path and (not args.save_signal or args.save_k_frac is None):
        raise ValueError("--save_model_path requires --save_signal and --save_k_frac")

    import torch  # noqa: F401

    from seq_core.pipeline import load_model_and_tokenizer, resolve_device, resolve_dtype, unload_model
    from seq_core.signals import collect_channel_activation_entropy, extract_all_signals
    from seq_core.channel_protect import apply_channel_protection
    from seq_core.quantizers import get_backend
    from seq_core.sensitivity import make_ppl_fn

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = _load_prompts(args.calibration_prompts)
    fracs = [float(x) for x in str(args.protect_fracs).split(",") if x.strip() != ""]
    signal_names = [s.strip() for s in str(args.signals).split(",") if s.strip()]

    backend = get_backend(args.backend)
    if not backend.is_available():
        raise RuntimeError(f"backend '{args.backend}' not available")

    ppl_fn = make_ppl_fn(
        dataset_name=args.ppl_dataset,
        split="test" if args.ppl_mode == "canonical" else "validation",
        seq_len=args.ppl_seq_len, device=device, dtype=dtype, mode=args.ppl_mode,
        max_examples=None if args.ppl_mode == "canonical" else args.ppl_max_examples,
        full_corpus=(args.ppl_mode == "canonical"), seed=args.seed,
    )

    # ---- pass 1: per-channel signals + baseline PPL ----------------------- #
    model, tokenizer = load_model_and_tokenizer(args.model, device, dtype, trust_remote_code=bool(args.trust_remote_code))
    signals = extract_all_signals(
        model, tokenizer=tokenizer, prompts=prompts, seq_len=args.calib_seq_len,
        device=device, max_prompts=args.max_calib_prompts, include_activation=bool(prompts),
        return_channels=True,
    )
    in_features = {n: m.in_features for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    model_parameter_count = sum(int(p.numel()) for p in model.parameters())
    linear_weight_ids = {id(m.weight) for m in model.modules() if isinstance(m, torch.nn.Linear)}
    tied_embedding_extra_count = sum(
        int(m.weight.numel()) for m in model.modules()
        if isinstance(m, torch.nn.Embedding) and id(m.weight) in linear_weight_ids
    )

    channel_entropy: Dict[str, List[float]] = {}
    if args.channel_entropy or "act_entropy" in signal_names:
        LOGGER.info("computing per-channel activation entropy ...")
        channel_entropy = collect_channel_activation_entropy(
            model, tokenizer, prompts, seq_len=args.calib_seq_len, device=device,
            max_prompts=args.max_calib_prompts, bins=args.entropy_bins,
        )
        if "act_entropy" not in signal_names:
            signal_names.append("act_entropy")

    skip = [n for n in in_features if "lm_head" in n] if args.skip_lm_head else []

    # FP16 baseline MUST be measured *before* any base precompute. Sequential GPTQ
    # mutates the decoder weights in place, so measuring after it would record the
    # GPTQ-base PPL under the "fp16" label (the baseline mislabel bug). Measure the
    # unmodified FP16 model here, and record the uniform-base PPL separately below.
    baseline_fp16_ppl = ppl_fn(model, tokenizer)
    LOGGER.info("FP16 baseline ppl = %.4f", baseline_fp16_ppl)

    # optional GPTQ base: precompute error-compensated fake-quant weights once.
    #   gptq       -> from-scratch GPTQ (diagnostics only; shelved, see docs/STATUS.md)
    #   gptq_llmc  -> load a saved LightCompress fake-quant model (the working base)
    gptq_base: Dict[str, Any] = {}
    baseline_base_ppl: Optional[float] = None
    if args.base_quantizer == "gptq":
        from seq_core.gptq import build_gptq_calibration, gptq_quantize_model, gptq_quantize_model_sequential

        # GPTQ needs a full-rank Hessian -> many real tokens (the signal-extraction
        # prompt set is far too small, which produces a singular H and garbage base).
        gptq_prompts = prompts
        if args.gptq_calib_samples > 0:
            LOGGER.info("building GPTQ calibration: %d x %d-token chunks of real text ...",
                        args.gptq_calib_samples, args.calib_seq_len)
            gptq_prompts = build_gptq_calibration(
                tokenizer, n_samples=args.gptq_calib_samples, seq_len=args.calib_seq_len, seed=args.seed,
            )
        LOGGER.info("precomputing GPTQ %d-bit base (%s, group_size=%d) ...",
                    args.base_bits, args.gptq_mode, args.gptq_group_size)
        if args.gptq_mode == "sequential":
            gptq_base = gptq_quantize_model_sequential(
                model, tokenizer, gptq_prompts, bits=args.base_bits, group_size=args.gptq_group_size,
                seq_len=args.calib_seq_len, device=device, max_prompts=args.gptq_calib_samples or 32,
                percdamp=args.gptq_percdamp, skip=skip,
            )
            # sequential GPTQ leaves the model AS the uniform GPTQ base -> this is the
            # k=0 sanity value (uniform base PPL), recorded separately from FP16.
            baseline_base_ppl = ppl_fn(model, tokenizer)
            LOGGER.info("uniform GPTQ base ppl (k=0 gate) = %.4f", baseline_base_ppl)
        else:
            gptq_base = gptq_quantize_model(
                model, tokenizer, gptq_prompts, bits=args.base_bits, group_size=args.gptq_group_size,
                seq_len=args.calib_seq_len, device=device, max_prompts=None,
                percdamp=args.gptq_percdamp, skip=skip, hessian_device=args.gptq_hessian_device,
            )
    elif args.base_quantizer == "gptq_llmc":
        from seq_core.gptq_llmc_base import load_llmc_fake_quant_base

        if not args.gptq_model_path:
            raise ValueError("--base_quantizer gptq_llmc requires --gptq_model_path <saved LightCompress fake-quant dir>")
        LOGGER.info("loading LightCompress fake-quant base from %s ...", args.gptq_model_path)
        gptq_base = load_llmc_fake_quant_base(
            args.gptq_model_path, in_features, skip=skip, device="cpu", dtype=dtype,
            trust_remote_code=bool(args.trust_remote_code),
        )
        LOGGER.info("loaded %d fake-quant layer weights from LightCompress base", len(gptq_base))

    # ---- selection precompute (needs the loaded model, before unload) -------- #
    # residual-aware scores, greedy OMP order, and value-based tier maps all need
    # the FP16 weights (and, for greedy, the input Hessian), so they are computed
    # here — pass 2 reloads a fresh model per config.
    skip_set = set(skip)
    linear_items = [(n, m) for n, m in model.named_modules()
                    if isinstance(m, torch.nn.Linear) and n not in skip_set]

    need_resid = any(("residual_rms" in s or "residual_max" in s) for s in signal_names)
    need_greedy = (args.select == "greedy")
    need_tier_alloc = (args.tier_alloc == "value")

    def _dequant_base_weights() -> Dict[str, Any]:
        """Per-layer HQQ dequantized base Wq (for greedy / tier-alloc distortion)."""
        import copy as _copy
        bw: Dict[str, Any] = {}
        for n, m in linear_items:
            try:
                q = backend.quantize_linear(_copy.deepcopy(m), args.base_bits, device=device,
                                            compute_dtype=dtype, group_size=args.group_size)
                wq = backend.dequantize_weight(q)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("base build: %s failed: %s", n, exc)
                continue
            if wq is not None:
                bw[n] = wq.detach().to(device="cpu", dtype=torch.float32)
        return bw

    # explicit base weights Wq per layer. gptq_* already have it; hqq builds once.
    base_weights: Dict[str, Any] = {}
    if need_greedy or need_tier_alloc:
        if gptq_base:
            base_weights = gptq_base
        elif args.base_quantizer == "hqq":
            LOGGER.info("building HQQ %d-bit base weights for selection ...", args.base_bits)
            base_weights = _dequant_base_weights()

    # residual-aware per-channel scores (residual_rms / residual_max)
    residual_scores: Dict[str, Dict[str, List[float]]] = {}
    if need_resid:
        if args.base_quantizer == "gptq":
            LOGGER.warning("residual signals need FP16 weights, but from-scratch gptq mutated the "
                           "model; skipping residual signals (use hqq or gptq_llmc)")
        else:
            from seq_core.recon_sensitivity import channel_residual_scores
            LOGGER.info("computing residual-aware channel scores ...")
            residual_scores = channel_residual_scores(
                model, tokenizer, prompts, backend, bits=args.base_bits, group_size=args.group_size,
                device=device, compute_dtype=dtype, seq_len=args.calib_seq_len,
                max_prompts=args.max_calib_prompts,
                precomputed_base=(base_weights or gptq_base or None),
            )

    # greedy OMP selection order per layer (interaction-aware; nested across fracs)
    greedy_order: Dict[str, List[int]] = {}
    if need_greedy:
        from seq_core.gptq import collect_gptq_hessians
        from seq_core.greedy_select import greedy_protected_map
        import math as _math
        LOGGER.info("collecting input Hessians for greedy selection ...")
        hessians = collect_gptq_hessians(
            model, tokenizer, prompts, seq_len=args.calib_seq_len, device=device,
            max_prompts=args.max_calib_prompts, hessian_device=args.gptq_hessian_device,
        )
        max_frac = max([f for f in fracs if f > 0] or [0.0])
        weights = {n: m.weight.detach() for n, m in linear_items}
        k_by_layer = {n: int(_math.ceil(max_frac * in_features[n])) for n in weights if n in in_features}
        greedy_order = greedy_protected_map(weights, base_weights, hessians, k_by_layer,
                                            device=device, logger=LOGGER)
        del hessians

    # value-based tier maps per budget (needs FP16 W -> precompute before unload)
    tier_maps: Dict[Any, Dict[str, Dict[int, List[int]]]] = {}
    if need_tier_alloc:
        import math as _math
        from seq_core.channel_protect import _quantize_columns
        from seq_core.channel_utils import greedy_bit_alloc_by_value
        from seq_core.signals import collect_input_stats
        LOGGER.info("collecting activation E[x^2] for value-based tier allocation ...")
        _accs = collect_input_stats(model, tokenizer, prompts, seq_len=args.calib_seq_len,
                                    device=device, max_prompts=args.max_calib_prompts)
        tier_bits = [args.base_bits, 8, 16]
        dist_by_layer: Dict[str, List[List[float]]] = {}
        for n, m in linear_items:
            wq = base_weights.get(n)
            acc = _accs.get(n)
            if wq is None or acc is None:
                continue
            W = m.weight.detach().to(dtype=torch.float32)
            Wq = wq.to(device=W.device, dtype=torch.float32)
            if Wq.shape != W.shape and Wq.t().shape == W.shape:
                Wq = Wq.t().contiguous()
            if Wq.shape != W.shape:
                continue
            asq = acc.act_sq().to(W.device)
            err_base = ((W - Wq) ** 2).sum(dim=0)
            err_8 = ((W - _quantize_columns(W, 8)) ** 2).sum(dim=0)
            d_base = (asq * err_base).detach().cpu().tolist()
            d_8 = (asq * err_8).detach().cpu().tolist()
            dist_by_layer[n] = [[d_base[j], d_8[j], 0.0] for j in range(len(d_base))]
        for budget in fracs:
            tmap: Dict[str, Dict[int, List[int]]] = {}
            for n, D in dist_by_layer.items():
                idx_bits = max(1, _math.ceil(_math.log2(max(2, in_features.get(n, 2)))))
                chosen = greedy_bit_alloc_by_value(D, tier_bits, float(budget), index_bits=idx_bits)
                tiers: Dict[int, List[int]] = {}
                for j, t in enumerate(chosen):
                    if t > 0:
                        tiers.setdefault(int(tier_bits[t]), []).append(j)
                tmap[n] = tiers
            tier_maps[budget] = tmap
        del _accs, dist_by_layer

    unload_model(model, tokenizer)

    from seq_core.channel_utils import combine_scores

    def _base_scores(part: str) -> Dict[str, List[float]]:
        """Per-layer per-channel scores for a single base signal."""
        if part == "random":
            rng = random.Random(args.seed)
            return {n: [rng.random() for _ in range(f)] for n, f in in_features.items()}
        if part == "act_entropy":
            if not channel_entropy:
                LOGGER.warning("act_entropy requested but not computed; skipping")
            return channel_entropy
        if part in ("residual_rms", "residual_max"):
            if not residual_scores:
                LOGGER.warning("residual signal '%s' requested but not computed; skipping", part)
            return {n: v[part] for n, v in residual_scores.items() if part in v}
        sc = _in_channel_scores(signals, part)
        if not sc:
            LOGGER.warning("signal '%s' has no per-channel arrays", part)
        return sc

    # precompute per-signal per-layer channel scores.
    #   'neg_<sig>'      -> protect the LOWEST-scored channels (e.g. low-entropy outliers)
    #   'A*B' / 'A+B'    -> composite: min-max normalize each, then product / sum
    signal_scores: Dict[str, Dict[str, List[float]]] = {}
    for s in signal_names:
        base = s[4:] if s.startswith("neg_") else s
        invert = s.startswith("neg_")
        if "*" in base or "+" in base:
            op = "mul" if "*" in base else "add"
            parts = [p.strip() for p in base.replace("+", "*").split("*") if p.strip()]
            part_maps = {p: _base_scores(p) for p in parts}
            layers = set.intersection(*[set(m.keys()) for m in part_maps.values()]) if all(part_maps.values()) else set()
            sc = {n: combine_scores([part_maps[p][n] for p in parts], op) for n in layers}
        else:
            sc = _base_scores(base)
        if invert:
            sc = {n: [-v for v in arr] for n, arr in sc.items()}
        signal_scores[s] = sc

    # build the config list: value-based tier budgets, single-tier fracs, or tier specs
    import math as _math
    from seq_core.channel_utils import parse_tiers
    if args.tier_alloc == "value":
        configs = [("tier_alloc", f"budget={b}", b) for b in fracs]
    elif args.protect_tiers.strip():
        configs = [("tiers", spec.strip(), parse_tiers(spec)) for spec in args.protect_tiers.split(";") if spec.strip()]
    else:
        configs = [("frac", k, k) for k in fracs]

    results: List[Dict[str, Any]] = []

    def _protect_and_eval(sig_label, kind, label, value, *, scores=None,
                          explicit_protected=None, explicit_tiers=None):
        model, tokenizer = load_model_and_tokenizer(args.model, device, dtype, trust_remote_code=bool(args.trust_remote_code))
        info = apply_channel_protection(
            model, scores if scores is not None else {n: [] for n in in_features},
            value if kind == "frac" else 0.0, backend, args.base_bits,
            device=device, compute_dtype=dtype, group_size=args.group_size,
            skip=skip, protect_bits=args.protect_bits,
            precomputed_base=(gptq_base or None),
            tier_fracs=(value if kind == "tiers" else None),
            explicit_protected=explicit_protected, explicit_tiers=explicit_tiers,
        )
        ppl = ppl_fn(model, tokenizer)
        # Authoritative storage estimate: ChannelProtectedLinear retains the
        # complete low-bit base and stores sparse correction columns on top.
        from seq_core.storage_accounting import account_storage
        import math as _storage_math
        qvals = scales = residual16 = tier8 = indices = bias_values = 0
        for layer in info["per_layer"].values():
            in_f = int(layer["in_features"]); out_f = int(layer["out_features"])
            qvals += in_f * out_f
            scales += _storage_math.ceil(in_f / max(1, int(args.group_size))) * out_f
            tiers = {int(k): int(v) for k, v in layer.get("tier_counts", {}).items()}
            residual16 += tiers.get(16, 0) * out_f
            tier8 += tiers.get(8, 0) * out_f
            indices += sum(tiers.values())
            bias_values += out_f if layer.get("has_bias") else 0
        unquantized = max(0, model_parameter_count - qvals - bias_values)
        storage = account_storage(
            quantized_values=qvals, quantized_bits=args.base_bits,
            scale_values=scales, zero_point_values=scales,
            # ChannelProtectedLinear currently keeps every correction tensor in
            # compute_dtype (FP16), including a logically INT8 target tier.
            fp16_residual_values=residual16 + tier8, int8_values=0,
            channel_index_values=indices,
            bias_values=bias_values, unquantized_parameter_values=unquantized,
            embedding_values=tied_embedding_extra_count,
            parameter_count=model_parameter_count,
        )
        storage["representation"] = "runtime_fake_quant_fp16_corrections"
        storage["logical_int8_tier_values"] = tier8
        row = {
            "signal": sig_label,
            "k_frac": value if kind == "frac" else None,
            "tiers": label if kind in ("tiers", "tier_alloc") else None,
            "effective_bits": info["effective_bits"],
            # comparison axis: weight-only bits/param (comparable to GPTQ-4 = 4.0).
            # Embeddings/lm_head/norms are FP16 in every method and excluded here;
            # the full-model average is kept separately for the deployment-size note.
            "actual_effective_bits": storage["actual_weight_bits_per_param"],
            "actual_model_bits_per_param": storage["actual_model_bits_per_parameter"],
            "storage": storage,
            "ppl": ppl,
            "delta_ppl_vs_fp16": (ppl - baseline_fp16_ppl) if ppl == ppl else None,
            "num_layers": info["num_layers"],
            "errors": len(info["errors"]),
        }
        results.append(row)

        should_save = (
            bool(args.save_model_path)
            and sig_label == args.save_signal
            and kind == "frac"
            and args.save_k_frac is not None
            and abs(float(value) - float(args.save_k_frac)) < 1e-12
        )
        if should_save:
            from seq_core.channel_protect import materialize_channel_protection

            save_path = Path(args.save_model_path)
            save_path.mkdir(parents=True, exist_ok=True)
            replaced = materialize_channel_protection(model)
            model.save_pretrained(save_path, safe_serialization=True)
            tokenizer.save_pretrained(save_path)
            manifest = {
                "format": "dense_fake_quant_evaluation_checkpoint",
                "compact_low_bit_checkpoint": False,
                "source_model": args.model,
                "base_quantizer": args.base_quantizer,
                "base_model_path": args.gptq_model_path or None,
                "signal": sig_label,
                "k_frac": float(value),
                "base_bits": int(args.base_bits),
                "group_size": int(args.group_size),
                "materialized_layers": replaced,
                "measured_ppl_before_save": float(ppl),
                "storage_estimate": storage,
            }
            (save_path / "seq_export_manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            LOGGER.info("saved reloadable dense fake-quant checkpoint to %s", save_path)

        unload_model(model, tokenizer)
        LOGGER.info("%-13s %-16s eff=%.2f ppl=%.4f (Δ%+.4f) errs=%d",
                    sig_label, f"{kind}={label}", info["effective_bits"], ppl,
                    row["delta_ppl_vs_fp16"] or 0.0, len(info["errors"]))

    # ---- pass 2: protect -> quantize -> PPL, by selection mode ------------- #
    if args.select == "greedy":
        # interaction-aware: one greedy order per layer, sliced per frac (signal-agnostic)
        for kind, label, value in configs:
            if kind != "frac":
                continue
            explicit = {n: order[: int(_math.ceil(value * in_features[n]))]
                        for n, order in greedy_order.items()}
            _protect_and_eval("greedy", kind, label, value, explicit_protected=explicit)
    elif args.tier_alloc == "value":
        # value-based bit allocation: per-layer {bits:[idx]} chosen by error-per-byte
        for kind, label, value in configs:
            _protect_and_eval("tier_alloc", kind, label, value, explicit_tiers=tier_maps.get(value, {}))
    else:
        for s in signal_names:
            scores = signal_scores.get(s) or {}
            if not scores:
                continue
            for kind, label, value in configs:
                _protect_and_eval(s, kind, label, value, scores=scores)

    # A precomputed LLMC base is evaluated in pass 2.  Record its k=0 PPL as
    # the base baseline without mutating the FP16 reference used by selectors.
    if baseline_base_ppl is None and args.base_quantizer == "gptq_llmc":
        k0 = next((r for r in results if float(r.get("k_frac") or 0.0) == 0.0
                   and r.get("ppl") is not None), None)
        if k0 is not None:
            baseline_base_ppl = float(k0["ppl"])

    payload = {
        "model": args.model, "backend": backend.name, "base_quantizer": args.base_quantizer,
        "base_bits": args.base_bits, "protect_bits": args.protect_bits, "protect_fracs": fracs,
        "baseline_fp16_ppl": baseline_fp16_ppl, "baseline_base_ppl": baseline_base_ppl,
        "select": args.select, "ppl_mode": args.ppl_mode,
        "skip_lm_head": bool(args.skip_lm_head), "results": results,
    }
    (out_dir / "channel_pareto.json").write_text(json.dumps(payload, indent=2))
    _write_md(out_dir / "channel_pareto.md", payload)
    LOGGER.info("wrote %s", out_dir / "channel_pareto.md")
    return 0


def _cfg_label(r: Dict[str, Any]) -> str:
    return f"[{r['tiers']}]" if r.get("tiers") else f"k={r.get('k_frac')}"


def _write_md(path: Path, payload: Dict[str, Any]) -> None:
    rs = payload["results"]
    signals = sorted({r["signal"] for r in rs})
    # config columns in first-seen order (fracs or tier specs)
    cols: List[str] = []
    for r in rs:
        lab = _cfg_label(r)
        if lab not in cols:
            cols.append(lab)
    base = payload["baseline_fp16_ppl"]

    def cell(s: str, col: str) -> Any:
        return next((r for r in rs if r["signal"] == s and _cfg_label(r) == col), None)

    L = [f"# Per-channel protection — {payload['model']}", ""]
    L.append(f"Backend `{payload['backend']}`, base {payload['base_bits']}-bit, {payload['ppl_mode']} PPL. "
             f"FP16 PPL = **{base:.4f}**.")
    L.append("Rows = signal; columns = protection config; `random` is the control. "
             "**At matched effective bits, signal < random means per-channel importance is real.**")
    L.append("")
    L.append("## PPL by config (effective bits in parentheses)")
    L.append("")
    L.append("| signal | " + " | ".join(cols) + " |")
    L.append("|" + "---|" * (len(cols) + 1))
    for s in signals:
        row = []
        for c in cols:
            r = cell(s, c)
            row.append(f"{r['ppl']:.3f} ({r['effective_bits']:.2f}b)" if r and r["ppl"] == r["ppl"] else "—")
        L.append(f"| `{s}` | " + " | ".join(row) + " |")
    L.append("")
    L.append("## PPL gap vs random (negative = signal beats random)")
    L.append("")
    L.append("| signal | " + " | ".join(cols) + " |")
    L.append("|" + "---|" * (len(cols) + 1))
    for s in signals:
        if s == "random":
            continue
        cells = []
        for c in cols:
            r = cell(s, c)
            rr = cell("random", c)
            if r and rr and r["ppl"] == r["ppl"] and rr["ppl"] == rr["ppl"]:
                cells.append(f"{r['ppl'] - rr['ppl']:+.3f}")
            else:
                cells.append("—")
        L.append(f"| `{s}` | " + " | ".join(cells) + " |")
    L.append("")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
