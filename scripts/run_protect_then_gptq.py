#!/usr/bin/env python3
"""F3 causal control: post-hoc restore-after-GPTQ vs protect-before-GPTQ (OWQ-style).

Same protected set S (residual-driven greedy) and the SAME internal GPTQ on both arms,
so the only difference is *when* S is protected relative to error compensation:

  base  : GPTQ every column, no protection                      (reference)
  A     : GPTQ every column, then overwrite S with FP16         (post-hoc restore)
  B     : keep S exact THROUGHOUT compensation (protect_cols)   (select-before-GPTQ / OWQ)

If A collapses while B stays healthy on the *same* S, the F3 catastrophe is caused by
breaking already-computed GPTQ compensation (a causal claim), not by restoring genuinely
useful columns. Also reports the overlap of S with the highest quantization-error columns.

Weight-only, lm_head excluded (matches the paper scope). One GPTQ implementation
(seq_core.gptq) drives both arms. Memory: keeps a few CPU fp16 copies of the linear
weights; fine for 1B/3B, heavier for 7B.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from seq_core.pipeline import load_model_and_tokenizer, resolve_device, resolve_dtype, unload_model
from seq_core.sensitivity import make_ppl_fn
from seq_core.gptq import build_gptq_calibration, collect_gptq_hessians, gptq_quantize_weight
from seq_core.greedy_select import greedy_select_channels

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("protect_then_gptq")


def _linears(model, skip_substr=("lm_head",)):
    import torch.nn as nn
    return {n: m for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and not any(s in n for s in skip_substr)}


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

    def ppl_of(model, tokenizer) -> float:
        fn = make_ppl_fn(dataset_name="wikitext2", split="test", seq_len=args.seq_len,
                         device=device, dtype=dtype, mode="canonical", full_corpus=True)
        return float(fn(model, tokenizer))

    # ---- 1. Hessians on the FP16 model; snapshot FP16 linear weights (CPU) ---- #
    model, tok = load_model_and_tokenizer(args.model, device, dtype)
    calib = build_gptq_calibration(tok, n_samples=args.n_calib, seq_len=args.seq_len, seed=args.seed)
    LOGGER.info("collecting GPTQ Hessians over %d calibration samples ...", len(calib))
    hess = collect_gptq_hessians(model, tok, calib, seq_len=args.seq_len, device=str(device))
    layers = _linears(model)
    W0 = {n: layers[n].weight.detach().float().cpu().clone() for n in layers}
    unload_model(model, tok)

    # ---- 2. per-layer: full GPTQ, select S on ΔW, build arm-A/arm-B weights ---- #
    base_w, armA, armB, stats = {}, {}, {}, {}
    for n, W in W0.items():
        if n not in hess:
            continue
        H = hess[n][0]
        Wq_full = gptq_quantize_weight(W, H, args.base_bits, group_size=args.group_size,
                                       clone_hessian=True)
        dW = W - Wq_full
        in_f = int(W.shape[1])
        k = int(math.ceil(args.protect_frac * in_f))
        S = greedy_select_channels(dW, H, k, exact_k=True)   # identical set for both arms
        Wa = Wq_full.clone()
        if S:
            Wa[:, S] = W[:, S].to(Wa.dtype)
        Wb = gptq_quantize_weight(W, H, args.base_bits, group_size=args.group_size,
                                  clone_hessian=True, protect_cols=S)
        base_w[n] = Wq_full.half()
        armA[n] = Wa.half()
        armB[n] = Wb.half()
        col_energy = (dW * dW).sum(dim=0)
        top = torch.argsort(col_energy, descending=True)[:max(1, k)].tolist()
        overlap = len(set(S) & set(top)) / max(1, k)
        stats[n] = {"k": k, "in_features": in_f,
                    "overlap_S_with_top_error_cols": round(overlap, 4)}
        del H, Wq_full, dW, Wa, Wb
        hess[n] = None

    # ---- 3. evaluate each arm by copying weights into a fresh model ---------- #
    def eval_arm(weights) -> float:
        m, t = load_model_and_tokenizer(args.model, device, dtype)
        lays = _linears(m)
        with torch.no_grad():
            for n, w in weights.items():
                if n in lays:
                    lays[n].weight.data.copy_(w.to(lays[n].weight.device, lays[n].weight.dtype))
        try:
            return ppl_of(m, t)
        finally:
            unload_model(m, t)

    LOGGER.info("evaluating FP16 sanity / base / arm-A (post-hoc) / arm-B (pre-GPTQ) ...")
    ppl_fp16 = eval_arm(W0)
    ppl_base = eval_arm(base_w)
    ppl_A = eval_arm(armA)
    ppl_B = eval_arm(armB)

    mean_overlap = round(sum(s["overlap_S_with_top_error_cols"] for s in stats.values())
                         / max(1, len(stats)), 4)
    result = {
        "model": args.model, "base_bits": args.base_bits, "group_size": args.group_size,
        "protect_frac": args.protect_frac, "seed": args.seed, "n_calib": args.n_calib,
        "skip_lm_head": True, "layers_quantized": len(stats),
        "ppl_fp16_sanity": ppl_fp16,
        "ppl_base_gptq": ppl_base,
        "ppl_A_posthoc_restore": ppl_A,
        "ppl_B_protect_before_gptq": ppl_B,
        "delta_A_vs_base": ppl_A - ppl_base,
        "delta_B_vs_base": ppl_B - ppl_base,
        "mean_overlap_S_with_top_error_cols": mean_overlap,
        "interpretation": ("A>>B (post-hoc collapses, pre-GPTQ healthy) => F3 is caused by "
                           "breaking GPTQ compensation, not by protecting useful columns"),
        "per_layer": stats,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    LOGGER.info("fp16=%.4f base=%.4f A(post-hoc)=%.4f B(pre-GPTQ)=%.4f | overlap=%.3f -> %s",
                ppl_fp16, ppl_base, ppl_A, ppl_B, mean_overlap, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
