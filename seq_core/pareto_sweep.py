#!/usr/bin/env python3
"""RQ2/RQ3 downstream Pareto: does a better signal give a better PPL–bits curve?

This is the *non-circular* test. The reconstruction study shows `hessian_diag`
predicts the local objective almost perfectly and `entropy` points the wrong
way — but a local proxy is not proof of the best end-to-end model. Here we:

  1. extract every signal once from the FP16 model,
  2. for each signal × effective-bit budget, allocate discrete bits with the
     greedy allocator (higher signal -> more bits, the signal's *native* claim
     of importance — exactly how SEQ originally used entropy),
  3. actually quantize the whole model with the chosen backend (HQQ = arbitrary
     bits), and
  4. measure real PPL.

A `random` control at each budget tests whether the signal beats chance. The
output is a PPL-vs-effective-bits table/curve per signal (the paper's headline
figure) at 5–7 effective bits.

Run on a GPU box. Example::

    python -m seq_core.pareto_sweep \
        --model meta-llama/Llama-3.2-1B --backend hqq \
        --signals hessian_diag,salience,magnitude,entropy,random \
        --budgets 4,5,6,7 --levels 3,4,8 \
        --ppl_mode canonical \
        --calibration_prompts calibration_prompts.json \
        --out_dir runs/pareto
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger("pareto_sweep")


def _load_prompts(path: Optional[str]) -> List[str]:
    if not path:
        return []
    from seq_core.pipeline import load_prompts

    return load_prompts(Path(path))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SEQ signal-driven Pareto sweep (RQ2/RQ3)")
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--calibration_prompts", default="calibration_prompts.json")
    p.add_argument("--calib_seq_len", type=int, default=2048)
    p.add_argument("--max_calib_prompts", type=int, default=64)

    p.add_argument("--backend", default="hqq")
    p.add_argument("--group_size", type=int, default=64)
    p.add_argument("--signals", default="hessian_diag,salience,magnitude,entropy,random")
    p.add_argument("--budgets", default="4,5,6,7", help="target effective-bit budgets")
    p.add_argument("--levels", default="3,4,8", help="discrete bit levels available")
    p.add_argument("--min_lm_head_bits", type=int, default=0, help="0 = pure signal (no floor)")
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--ppl_mode", default="proxy", choices=["proxy", "canonical"])
    p.add_argument("--ppl_dataset", default="wikitext2")
    p.add_argument("--ppl_seq_len", type=int, default=2048)
    p.add_argument("--ppl_max_examples", type=int, default=64, help="proxy mode only")
    p.add_argument("--out_dir", default="runs/pareto")
    return p.parse_args()


def _random_scores(names: List[str], seed: int) -> Dict[str, float]:
    rng = random.Random(seed)
    return {n: rng.random() for n in names}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    import torch  # noqa: F401

    from seq_core.pipeline import (
        load_model_and_tokenizer,
        resolve_device,
        resolve_dtype,
        unload_model,
    )
    from seq_core.signals import extract_all_signals, module_scalar_table
    from seq_core.stats_utils import greedy_bit_allocation
    from seq_core.quantizers import (
        apply_bit_map,
        effective_bits_from_map,
        get_backend,
        verify_bit_map,
    )
    from seq_core.sensitivity import make_ppl_fn

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = _load_prompts(args.calibration_prompts)
    levels = [int(x) for x in str(args.levels).split(",") if x.strip()]
    budgets = [float(x) for x in str(args.budgets).split(",") if x.strip()]
    signal_names = [s.strip() for s in str(args.signals).split(",") if s.strip()]

    backend = get_backend(args.backend)
    if not backend.is_available():
        raise RuntimeError(f"backend '{args.backend}' not available")

    ppl_fn = make_ppl_fn(
        dataset_name=args.ppl_dataset,
        split="test" if args.ppl_mode == "canonical" else "validation",
        seq_len=args.ppl_seq_len,
        device=device,
        dtype=dtype,
        mode=args.ppl_mode,
        max_examples=None if args.ppl_mode == "canonical" else args.ppl_max_examples,
        full_corpus=(args.ppl_mode == "canonical"),
        seed=args.seed,
    )

    # ---- pass 1: signals + baseline PPL from the FP16 model ---------------- #
    model, tokenizer = load_model_and_tokenizer(args.model, device, dtype, trust_remote_code=bool(args.trust_remote_code))
    signals = extract_all_signals(
        model, tokenizer=tokenizer, prompts=prompts, seq_len=args.calib_seq_len,
        device=device, max_prompts=args.max_calib_prompts, include_activation=bool(prompts),
    )
    scalar = module_scalar_table(signals, granularity="module")
    param_counts: Dict[str, int] = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            param_counts[name] = int(module.weight.numel() + (module.bias.numel() if module.bias is not None else 0))
    linear_names = list(param_counts.keys())
    baseline_ppl = ppl_fn(model, tokenizer)
    LOGGER.info("FP16 baseline ppl = %.4f", baseline_ppl)
    (out_dir / "signals_module.json").write_text(json.dumps(scalar, indent=2))
    unload_model(model, tokenizer)

    protected = {}
    if args.min_lm_head_bits:
        for n in linear_names:
            if "lm_head" in n:
                protected[n] = args.min_lm_head_bits

    # ---- pass 2: per (signal, budget) allocate -> quantize -> PPL ---------- #
    results: List[Dict[str, Any]] = []
    for signal_name in signal_names:
        if signal_name == "random":
            scores = _random_scores(linear_names, args.seed)
        elif signal_name in scalar:
            scores = {n: v for n, v in scalar[signal_name].items() if n in param_counts}
        else:
            LOGGER.warning("signal '%s' not found; skipping", signal_name)
            continue
        for budget in budgets:
            bit_map = greedy_bit_allocation(scores, param_counts, levels=levels, target_bits=budget, protected=protected or None)
            eff = effective_bits_from_map(bit_map, param_counts)
            model, tokenizer = load_model_and_tokenizer(args.model, device, dtype, trust_remote_code=bool(args.trust_remote_code))
            info = apply_bit_map(model, bit_map, backend, device=device, fp16_dtype=dtype, compute_dtype=dtype, group_size=args.group_size)
            vinfo = verify_bit_map(model, bit_map)
            ppl = ppl_fn(model, tokenizer)
            unload_model(model, tokenizer)
            row = {
                "signal": signal_name,
                "target_bits": budget,
                "effective_bits": eff.get("effective_bits"),
                "ppl": ppl,
                "delta_ppl_vs_fp16": ppl - baseline_ppl if ppl == ppl else None,
                "params_by_bits": eff.get("params_by_bits"),
                "quant_errors": len(info.get("errors", [])),
                "verify_mismatches": vinfo.get("num_mismatches"),
            }
            results.append(row)
            LOGGER.info("%-14s budget=%.1f  eff=%.3f  ppl=%.4f (Δ%+.4f)",
                        signal_name, budget, row["effective_bits"] or -1, ppl, row["delta_ppl_vs_fp16"] or 0.0)

    payload = {
        "model": args.model,
        "backend": backend.name,
        "levels": levels,
        "budgets": budgets,
        "baseline_fp16_ppl": baseline_ppl,
        "ppl_mode": args.ppl_mode,
        "min_lm_head_bits": args.min_lm_head_bits,
        "results": results,
    }
    (out_dir / "pareto.json").write_text(json.dumps(payload, indent=2))
    _write_markdown(out_dir / "pareto.md", payload)
    LOGGER.info("wrote %s", out_dir / "pareto.md")
    return 0


def _write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    results = payload["results"]
    signals = sorted({r["signal"] for r in results})
    budgets = payload["budgets"]
    base = payload["baseline_fp16_ppl"]
    L = [f"# Pareto: PPL vs effective bits — {payload['model']}", ""]
    L.append(f"Backend `{payload['backend']}`, levels {payload['levels']}, "
             f"{payload['ppl_mode']} PPL. FP16 baseline PPL = **{base:.4f}**.")
    L.append("Each signal is used in its native high→more-bits direction (how SEQ used entropy). "
             "Lower PPL at equal effective bits is better; `random` is the chance control.")
    L.append("")
    L.append("## PPL at each target budget (effective bits in parentheses)")
    L.append("")
    L.append("| signal | " + " | ".join(f"~{b} bits" for b in budgets) + " |")
    L.append("|" + "---|" * (len(budgets) + 1))
    for s in signals:
        cells = []
        for b in budgets:
            row = next((r for r in results if r["signal"] == s and r["target_bits"] == b), None)
            if row and row["ppl"] == row["ppl"]:
                cells.append(f"{row['ppl']:.3f} ({row['effective_bits']:.2f})")
            else:
                cells.append("—")
        L.append(f"| `{s}` | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## ΔPPL vs FP16 (lower = closer to lossless)")
    L.append("")
    L.append("| signal | " + " | ".join(f"~{b} bits" for b in budgets) + " |")
    L.append("|" + "---|" * (len(budgets) + 1))
    for s in signals:
        cells = []
        for b in budgets:
            row = next((r for r in results if r["signal"] == s and r["target_bits"] == b), None)
            d = row.get("delta_ppl_vs_fp16") if row else None
            cells.append(f"{d:+.3f}" if d is not None else "—")
        L.append(f"| `{s}` | " + " | ".join(cells) + " |")
    L.append("")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
