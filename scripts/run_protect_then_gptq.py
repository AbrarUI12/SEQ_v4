#!/usr/bin/env python3
"""F3 causal test: post-hoc restoration vs select-before-GPTQ, same protected set.

Three configurations built with the SAME sequential GPTQ implementation, the same
calibration data, the same budget and the same evaluator. Only *when* the protected
channels S are made exact differs:

  base : sequential GPTQ over all columns                      (reference)
  A    : base, then overwrite S with the ORIGINAL FP16 columns (post-hoc restoration)
  B    : S held exact THROUGHOUT compensation                  (select-before-GPTQ, OWQ-style)

S is chosen once, by residual-driven greedy inside the arm-A pass, and reused verbatim for
arm B. If A collapses while B stays near base on the same S, the catastrophe is caused by
discarding already-computed GPTQ compensation rather than by protecting those columns per se.

Uses the SEQUENTIAL path deliberately: the one-shot GPTQ path miscalibrates (upstream layers
change the activations), and an earlier version of this experiment built on it produced a
degenerate base (PPL > 3000) that made all arms uninformative.

Memory: only the protected columns of the original weights are retained (a few % of one
weight matrix per layer), and each arm is evaluated before the next is built.
Weight-only; lm_head excluded, matching the paper's scope.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from seq_core.gptq import build_gptq_calibration, gptq_quantize_model_sequential
from seq_core.pipeline import load_model_and_tokenizer, resolve_device, resolve_dtype, unload_model
from seq_core.proglog import banner, quiet_http_logs
from seq_core.sensitivity import make_ppl_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("protect_then_gptq")
quiet_http_logs()


def _skip_names(model) -> list:
    """lm_head (and any tied output head) is FP16 in every method — never quantize it."""
    return [n for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and "lm_head" in n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base_bits", type=int, default=4)
    ap.add_argument("--group_size", type=int, default=128)
    ap.add_argument("--protect_frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n_calib", type=int, default=128)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype("float16", device)
    ppl_fn = make_ppl_fn(dataset_name="wikitext2", split="test", seq_len=args.seq_len,
                         device=device, dtype=dtype, mode="canonical", full_corpus=True)

    def quantize(model, tokenizer, **kw):
        return gptq_quantize_model_sequential(
            model, tokenizer, build_gptq_calibration(
                tokenizer, n_samples=args.n_calib, seq_len=args.seq_len, seed=args.seed),
            bits=args.base_bits, group_size=args.group_size, seq_len=args.seq_len,
            device=str(device), max_prompts=args.n_calib,
            skip=_skip_names(model), **kw)

    # ---- FP16 reference -------------------------------------------------- #
    banner(LOGGER, 1, 5, f"FP16 reference perplexity ({args.model})")
    model, tok = load_model_and_tokenizer(args.model, device, dtype)
    ppl_fp16 = float(ppl_fn(model, tok))
    LOGGER.info("FP16 reference ppl = %.4f", ppl_fp16)

    # ---- base + arm A: quantize everything, selecting S in-pass ----------- #
    banner(LOGGER, 2, 5, "sequential GPTQ base (selecting protected set S in-pass)")
    selected: dict = {}
    protected_orig: dict = {}
    quantize(model, tok, select_k_frac=args.protect_frac,
             selected_out=selected, protected_orig_out=protected_orig)
    banner(LOGGER, 3, 5, "perplexity of the unprotected GPTQ base")
    ppl_base = float(ppl_fn(model, tok))
    n_sel = sum(len(v) for v in selected.values())
    LOGGER.info("sequential GPTQ base ppl = %.4f (selected %d channels over %d layers)",
                ppl_base, n_sel, len(selected))

    banner(LOGGER, 4, 5, "arm A: post-hoc restore S to original FP16")
    linears = dict(model.named_modules())
    with torch.no_grad():
        for name, cols in selected.items():
            mod = linears.get(name)
            if mod is None or not cols or name not in protected_orig:
                continue
            w = protected_orig[name].to(mod.weight.device, mod.weight.dtype)
            mod.weight.data[:, cols] = w
    ppl_A = float(ppl_fn(model, tok))
    LOGGER.info("arm A (post-hoc restore) ppl = %.4f", ppl_A)
    unload_model(model, tok)

    # ---- arm B: same S, held exact throughout compensation ---------------- #
    banner(LOGGER, 5, 5, "arm B: select-before-GPTQ (S exact throughout compensation)")
    model, tok = load_model_and_tokenizer(args.model, device, dtype)
    quantize(model, tok, protect_cols_by_layer=selected)
    ppl_B = float(ppl_fn(model, tok))
    LOGGER.info("arm B (select-before-GPTQ) ppl = %.4f", ppl_B)
    unload_model(model, tok)

    verdict = ("A_collapses_B_healthy" if (ppl_A > 2.0 * ppl_base and ppl_B < 1.5 * ppl_base)
               else "both_healthy" if (ppl_A < 1.5 * ppl_base and ppl_B < 1.5 * ppl_base)
               else "both_degraded" if (ppl_A > 2.0 * ppl_base and ppl_B > 2.0 * ppl_base)
               else "mixed")
    result = {
        "model": args.model, "base_bits": args.base_bits, "group_size": args.group_size,
        "protect_frac": args.protect_frac, "seed": args.seed, "n_calib": args.n_calib,
        "gptq_path": "sequential", "skip_lm_head": True,
        "layers_selected": len(selected), "channels_selected": n_sel,
        "ppl_fp16": ppl_fp16,
        "ppl_base_gptq": ppl_base,
        "ppl_A_posthoc_restore": ppl_A,
        "ppl_B_protect_before_gptq": ppl_B,
        "delta_A_vs_base": ppl_A - ppl_base,
        "delta_B_vs_base": ppl_B - ppl_base,
        "verdict": verdict,
        "reading": ("A_collapses_B_healthy => post-hoc restoration breaks GPTQ compensation "
                    "(causal); both_degraded => the selected columns are harmful regardless of "
                    "ordering; both_healthy => no catastrophe reproduced in this configuration"),
        "selected_channels_per_layer": {k: len(v) for k, v in selected.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    LOGGER.info("fp16=%.4f base=%.4f A=%.4f B=%.4f -> %s | wrote %s",
                ppl_fp16, ppl_base, ppl_A, ppl_B, verdict, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
