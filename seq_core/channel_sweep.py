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
    p.add_argument("--signals", default="act_scale,act_max,act_kurt,hessian_diag,magnitude,random")
    p.add_argument("--channel_entropy", action="store_true",
                   help="also compute per-channel activation entropy as signal 'act_entropy' (memory-heavy; try 1B/3B first)")
    p.add_argument("--entropy_bins", type=int, default=32)
    p.add_argument("--skip_lm_head", action="store_true")
    p.add_argument("--base_quantizer", default="hqq", choices=["hqq", "gptq"],
                   help="base quantizer under the protection (gptq = error-compensated; run 1B/3B)")
    p.add_argument("--gptq_group_size", type=int, default=128)
    p.add_argument("--gptq_percdamp", type=float, default=0.01)
    p.add_argument("--gptq_calib_samples", type=int, default=128,
                   help="GPTQ needs many tokens for a full-rank Hessian; build this many "
                        "seq_len chunks of real text (0 = reuse --calibration_prompts, usually too small)")
    p.add_argument("--gptq_hessian_device", default="cpu", choices=["cpu", "cuda"],
                   help="cpu (default) accumulates Hessians off-GPU so 3B/8B don't OOM; cuda is faster for 1B")
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--ppl_mode", default="canonical", choices=["proxy", "canonical"])
    p.add_argument("--ppl_dataset", default="wikitext2")
    p.add_argument("--ppl_seq_len", type=int, default=2048)
    p.add_argument("--ppl_max_examples", type=int, default=64)
    p.add_argument("--out_dir", default="runs/channel")
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

    # optional GPTQ base: precompute error-compensated fake-quant weights once
    gptq_base: Dict[str, Any] = {}
    if args.base_quantizer == "gptq":
        from seq_core.gptq import build_gptq_calibration, gptq_quantize_model

        # GPTQ needs a full-rank Hessian -> many real tokens (the signal-extraction
        # prompt set is far too small, which produces a singular H and garbage base).
        gptq_prompts = prompts
        if args.gptq_calib_samples > 0:
            LOGGER.info("building GPTQ calibration: %d x %d-token chunks of real text ...",
                        args.gptq_calib_samples, args.calib_seq_len)
            gptq_prompts = build_gptq_calibration(
                tokenizer, n_samples=args.gptq_calib_samples, seq_len=args.calib_seq_len, seed=args.seed,
            )
        LOGGER.info("precomputing GPTQ %d-bit base (group_size=%d) ...", args.base_bits, args.gptq_group_size)
        gptq_base = gptq_quantize_model(
            model, tokenizer, gptq_prompts, bits=args.base_bits, group_size=args.gptq_group_size,
            seq_len=args.calib_seq_len, device=device, max_prompts=None,
            percdamp=args.gptq_percdamp, skip=skip, hessian_device=args.gptq_hessian_device,
        )

    baseline_ppl = ppl_fn(model, tokenizer)
    LOGGER.info("FP16 baseline ppl = %.4f", baseline_ppl)
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

    # ---- pass 2: per (signal, k) protect -> quantize -> PPL --------------- #
    results: List[Dict[str, Any]] = []
    for s in signal_names:
        scores = signal_scores.get(s) or {}
        if not scores:
            continue
        for k in fracs:
            model, tokenizer = load_model_and_tokenizer(args.model, device, dtype, trust_remote_code=bool(args.trust_remote_code))
            info = apply_channel_protection(
                model, scores, k, backend, args.base_bits,
                device=device, compute_dtype=dtype, group_size=args.group_size,
                skip=skip, protect_bits=args.protect_bits,
                precomputed_base=(gptq_base or None),
            )
            ppl = ppl_fn(model, tokenizer)
            unload_model(model, tokenizer)
            row = {
                "signal": s, "k_frac": k,
                "effective_bits": info["effective_bits"],
                "ppl": ppl,
                "delta_ppl_vs_fp16": (ppl - baseline_ppl) if ppl == ppl else None,
                "num_layers": info["num_layers"],
                "errors": len(info["errors"]),
            }
            results.append(row)
            LOGGER.info("%-13s k=%.3f eff=%.2f ppl=%.4f (Δ%+.4f) errs=%d",
                        s, k, info["effective_bits"], ppl, row["delta_ppl_vs_fp16"] or 0.0, len(info["errors"]))

    payload = {
        "model": args.model, "backend": backend.name, "base_quantizer": args.base_quantizer,
        "base_bits": args.base_bits, "protect_bits": args.protect_bits, "protect_fracs": fracs,
        "baseline_fp16_ppl": baseline_ppl, "ppl_mode": args.ppl_mode,
        "skip_lm_head": bool(args.skip_lm_head), "results": results,
    }
    (out_dir / "channel_pareto.json").write_text(json.dumps(payload, indent=2))
    _write_md(out_dir / "channel_pareto.md", payload)
    LOGGER.info("wrote %s", out_dir / "channel_pareto.md")
    return 0


def _write_md(path: Path, payload: Dict[str, Any]) -> None:
    rs = payload["results"]
    signals = sorted({r["signal"] for r in rs})
    fracs = payload["protect_fracs"]
    base = payload["baseline_fp16_ppl"]
    L = [f"# Per-channel protection — {payload['model']}", ""]
    L.append(f"Backend `{payload['backend']}`, base {payload['base_bits']}-bit, protected columns at "
             f"{payload['protect_bits']}-bit, {payload['ppl_mode']} PPL. FP16 PPL = **{base:.4f}**.")
    L.append("Rows = signal used to pick protected channels; `random` is the control. "
             "**At each k (same effective bits), signal < random means per-channel importance is real.**")
    L.append("")
    L.append("## PPL by protection fraction k (effective bits in parentheses)")
    L.append("")
    L.append("| signal | " + " | ".join(f"k={k}" for k in fracs) + " |")
    L.append("|" + "---|" * (len(fracs) + 1))
    for s in signals:
        cells = []
        for k in fracs:
            r = next((r for r in rs if r["signal"] == s and r["k_frac"] == k), None)
            cells.append(f"{r['ppl']:.3f} ({r['effective_bits']:.2f}b)" if r and r["ppl"] == r["ppl"] else "—")
        L.append(f"| `{s}` | " + " | ".join(cells) + " |")
    L.append("")
    # gap vs random per k
    L.append("## PPL gap vs random at each k (negative = signal beats random)")
    L.append("")
    L.append("| signal | " + " | ".join(f"k={k}" for k in fracs) + " |")
    L.append("|" + "---|" * (len(fracs) + 1))
    for s in signals:
        if s == "random":
            continue
        cells = []
        for k in fracs:
            r = next((r for r in rs if r["signal"] == s and r["k_frac"] == k), None)
            rr = next((r for r in rs if r["signal"] == "random" and r["k_frac"] == k), None)
            if r and rr and r["ppl"] == r["ppl"] and rr["ppl"] == rr["ppl"]:
                cells.append(f"{r['ppl'] - rr['ppl']:+.3f}")
            else:
                cells.append("—")
        L.append(f"| `{s}` | " + " | ".join(cells) + " |")
    L.append("")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
