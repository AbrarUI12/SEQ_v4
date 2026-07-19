#!/usr/bin/env python3
"""RQ1/RQ2 driver: measure how well each signal predicts quantization sensitivity.

Pipeline:
  1. load model + calibration prompts
  2. extract all candidate signals (module-level scalars)          [signals.py]
  3. measure ground-truth sensitivity via one-hot degrade          [sensitivity.py]
  4. rank-correlate every signal against the ground truth          [stats_utils.py]
  5. allocate bits by the best signal at a target budget and report effective bits
  6. write JSON + Markdown report

Run on a GPU box (needs torch + a loadable model). Example::

    python -m seq_core.signal_study \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --backend hqq --sensitivity_bits 3 \
        --sensitivity_ppl_mode proxy --sensitivity_max_examples 32 \
        --calibration_prompts calibration_prompts.json \
        --target_bits 6.0 --out_dir results
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger("signal_study")


def _load_prompts(path: Optional[str]) -> List[str]:
    if not path:
        return []
    from seq_core.pipeline import load_prompts

    return load_prompts(Path(path))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SEQ signal-quality study (RQ1/RQ2)")
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--calibration_prompts", default="calibration_prompts.json")
    p.add_argument("--calib_seq_len", type=int, default=2048)
    p.add_argument("--max_calib_prompts", type=int, default=64)

    p.add_argument("--backend", default="hqq", help="hqq|bnb (bit substrate for the degrade test)")
    p.add_argument("--group_size", type=int, default=64)
    p.add_argument("--sensitivity_bits", type=int, default=3)
    p.add_argument("--skip_sensitivity", action="store_true", help="only extract + save signals")
    p.add_argument("--max_modules", type=int, default=0, help="0 = all Linear modules")
    p.add_argument(
        "--ground_truth",
        default="ppl_degrade",
        choices=["ppl_degrade", "recon"],
        help="ppl_degrade: one-hot ΔPPL (noisy, global). recon: local reconstruction error "
        "(deterministic, per-channel-capable, recommended after run 1).",
    )

    p.add_argument("--sensitivity_ppl_mode", default="proxy", choices=["proxy", "canonical"])
    p.add_argument("--sensitivity_ppl_dataset", default="wikitext2")
    p.add_argument("--sensitivity_ppl_seq_len", type=int, default=512)
    p.add_argument("--sensitivity_max_examples", type=int, default=32)

    p.add_argument("--target_bits", type=float, default=6.0)
    p.add_argument("--alloc_levels", default="3,4,8", help="comma-separated bit levels for allocation")
    p.add_argument(
        "--out_dir",
        default=str(Path(__file__).resolve().parents[1] / "results"),
        help="directory for signal-study reports (defaults to this checkout's results directory)",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    import torch  # noqa: F401

    from seq_core.pipeline import load_model_and_tokenizer, resolve_device, resolve_dtype
    from seq_core.signals import extract_all_signals, module_scalar_table
    from seq_core.stats_utils import greedy_bit_allocation
    from seq_core.quantizers import effective_bits_from_map, get_backend
    from seq_core import sensitivity as sens

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = _load_prompts(args.calibration_prompts)
    LOGGER.info("loaded %d calibration prompts", len(prompts))

    model, tokenizer = load_model_and_tokenizer(
        args.model, device, dtype, trust_remote_code=bool(args.trust_remote_code)
    )

    # ---- 2. signals -------------------------------------------------------- #
    LOGGER.info("extracting signals ...")
    signals = extract_all_signals(
        model,
        tokenizer=tokenizer,
        prompts=prompts,
        seq_len=args.calib_seq_len,
        device=device,
        max_prompts=args.max_calib_prompts,
        include_activation=bool(prompts),
        return_channels=False,
    )
    scalar_table = module_scalar_table(signals, granularity="module")
    (out_dir / "signals_module.json").write_text(json.dumps(scalar_table, indent=2))
    LOGGER.info("signals: %s", ", ".join(sorted(scalar_table.keys())))

    # module -> param count (for allocation + effective bits)
    param_counts: Dict[str, int] = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            n = module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
            param_counts[name] = int(n)

    report: Dict[str, Any] = {
        "model": args.model,
        "device": device,
        "signals": sorted(scalar_table.keys()),
        "num_modules": len(param_counts),
    }

    if not args.skip_sensitivity:
        # ---- 3. ground-truth sensitivity ---------------------------------- #
        names = sens.linear_module_names(model)
        if args.max_modules and args.max_modules > 0:
            names = names[: args.max_modules]
        backend = get_backend(args.backend)
        if not backend.is_available():
            raise RuntimeError(f"backend '{args.backend}' not available in this environment")

        if args.ground_truth == "recon":
            from seq_core import recon_sensitivity as recon

            LOGGER.info("measuring reconstruction sensitivity (%d modules @ %d-bit) ...",
                        len(names), args.sensitivity_bits)
            gt_result = recon.reconstruction_sensitivity(
                model, tokenizer, prompts, backend,
                bits=args.sensitivity_bits, group_size=args.group_size,
                device=device, compute_dtype=dtype,
                seq_len=args.calib_seq_len, max_prompts=args.max_calib_prompts,
                module_names=names, return_channels=False,
            )
            (out_dir / "sensitivity_recon.json").write_text(json.dumps(gt_result, indent=2))
            gt = recon.ground_truth_scores(gt_result, key="module")
            ranked = recon.correlate_signals(scalar_table, gt)
        else:
            ppl_fn = sens.make_ppl_fn(
                dataset_name=args.sensitivity_ppl_dataset,
                split="test" if args.sensitivity_ppl_mode == "canonical" else "validation",
                seq_len=args.sensitivity_ppl_seq_len,
                device=device,
                dtype=dtype,
                mode=args.sensitivity_ppl_mode,
                max_examples=None if args.sensitivity_ppl_mode == "canonical" else args.sensitivity_max_examples,
                full_corpus=(args.sensitivity_ppl_mode == "canonical"),
            )
            LOGGER.info("measuring one-hot degrade sensitivity (%d modules @ %d-bit) ...",
                        len(names), args.sensitivity_bits)
            gt_result = sens.one_hot_degrade(
                model, tokenizer, ppl_fn, backend,
                bits=args.sensitivity_bits, module_names=names,
                device=device, compute_dtype=dtype, group_size=args.group_size,
            )
            (out_dir / "sensitivity_degrade.json").write_text(json.dumps(gt_result, indent=2))
            gt = sens.ground_truth_scores(gt_result, key="delta_ppl")
            report["baseline_ppl"] = gt_result.get("baseline_ppl")
            ranked = sens.correlate_signals(scalar_table, gt)

        # ---- 4. correlate ------------------------------------------------- #
        report["ground_truth"] = args.ground_truth
        report["sensitivity_bits"] = args.sensitivity_bits
        report["signal_ranking"] = ranked
        (out_dir / "signal_ranking.json").write_text(json.dumps(ranked, indent=2))

        # ---- 5. downstream allocation by best signal ---------------------- #
        best_row = next((r for r in ranked if r["spearman"] is not None), None)
        best = best_row["signal"] if best_row else None
        if best:
            levels = [int(x) for x in str(args.alloc_levels).split(",") if x.strip()]
            # Orient the signal so that "higher == more sensitive == more bits":
            # a signal inversely related to sensitivity (e.g. weight entropy may
            # be) has negative Spearman and must be negated before allocation.
            sign = -1.0 if best_row["spearman"] < 0 else 1.0
            oriented = {k: sign * v for k, v in scalar_table[best].items()}
            alloc = greedy_bit_allocation(
                oriented, param_counts, levels=levels, target_bits=args.target_bits
            )
            eff = effective_bits_from_map(alloc, param_counts)
            report["best_signal"] = best
            report["allocation_target_bits"] = args.target_bits
            report["allocation_effective_bits"] = eff.get("effective_bits")
            report["allocation_params_by_bits"] = eff.get("params_by_bits")
            (out_dir / "allocation_best_signal.json").write_text(
                json.dumps({"signal": best, "bit_map": alloc, "effective_bits": eff}, indent=2)
            )

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    _write_markdown(out_dir / "report.md", report)
    LOGGER.info("wrote report to %s", out_dir)
    return 0


def _write_markdown(path: Path, report: Dict[str, Any]) -> None:
    lines = [f"# Signal-quality study — {report.get('model')}", ""]
    lines.append(f"- modules: {report.get('num_modules')}")
    if "baseline_ppl" in report:
        lines.append(f"- baseline FP16 ppl: {report.get('baseline_ppl')}")
        lines.append(f"- degrade bits: {report.get('sensitivity_bits')}")
    lines.append("")
    ranking = report.get("signal_ranking")
    if ranking:
        lines.append("## RQ1/RQ2 — signal vs. measured sensitivity")
        lines.append("")
        lines.append("| rank | signal | Spearman ρ | Kendall τ | Pearson | n |")
        lines.append("|---|---|---|---|---|---|")
        for i, r in enumerate(ranking, 1):
            def fmt(v: Any) -> str:
                return "—" if v is None else f"{v:.3f}"
            lines.append(
                f"| {i} | `{r['signal']}` | {fmt(r.get('spearman'))} | "
                f"{fmt(r.get('kendall_tau'))} | {fmt(r.get('pearson'))} | {r.get('n')} |"
            )
        lines.append("")
    if report.get("best_signal"):
        lines.append("## Downstream allocation")
        lines.append("")
        lines.append(f"- best signal: `{report['best_signal']}`")
        lines.append(f"- target effective bits: {report.get('allocation_target_bits')}")
        lines.append(f"- achieved effective bits: {report.get('allocation_effective_bits')}")
        lines.append(f"- params by bits: {report.get('allocation_params_by_bits')}")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
