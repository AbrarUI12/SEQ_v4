#!/usr/bin/env python3
"""Per-channel true-sensitivity audit — the honest ground truth for protection.

Runs 4–5 showed protecting channels by activation *outlier magnitude* (act_max)
beats AWQ's mean magnitude. But is act_max's ordering actually optimal, or does
it miss channels? This measures it directly:

  1. quantize the whole model to ``base_bits`` (no protection) → baseline PPL,
  2. for sampled layers, rank input channels by a signal and split into buckets,
  3. protect one bucket at a time (restore those columns to FP16) and measure the
     ΔPPL it recovers — the *true* protection value of that rank band,
  4. correlate every candidate signal's per-bucket value against measured ΔPPL.

If the value is monotonic in the ranking signal, the signal is well-ordered; if
another signal correlates better with measured value, there is headroom for a
signal that beats the outlier-magnitude heuristic. Uses proxy PPL by default
(many evals). Run on 1B/3B (the FP16 correction doubles a layer's memory).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger("channel_audit")


def _load_prompts(path: Optional[str]) -> List[str]:
    if not path:
        return []
    from seq_core.pipeline import load_prompts

    return load_prompts(Path(path))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-channel true-sensitivity audit")
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--calibration_prompts", default="calibration_prompts.json")
    p.add_argument("--calib_seq_len", type=int, default=2048)
    p.add_argument("--max_calib_prompts", type=int, default=64)

    p.add_argument("--backend", default="hqq")
    p.add_argument("--group_size", type=int, default=64)
    p.add_argument("--base_bits", type=int, default=3)
    p.add_argument("--rank_by", default="act_max", help="signal used to bucket channels")
    p.add_argument("--num_buckets", type=int, default=8)
    p.add_argument("--audit_layers", type=int, default=6, help="how many layers to sample (evenly by depth)")
    p.add_argument("--layers", default="", help="explicit comma-separated layer names (overrides sampling)")
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--ppl_mode", default="proxy", choices=["proxy", "canonical"])
    p.add_argument("--ppl_seq_len", type=int, default=512)
    p.add_argument("--ppl_max_examples", type=int, default=128)
    p.add_argument("--out_dir", default="runs/audit")
    return p.parse_args()


def _channel_arr(signals: Dict[str, Any], layer: str, signal: str) -> Optional[List[float]]:
    by = signals.get(layer, {}).get(signal)
    if isinstance(by, dict) and isinstance(by.get("in_channel"), list):
        return by["in_channel"]
    return None


def _bucket_mean(arr: Optional[List[float]], idx: List[int]) -> Optional[float]:
    if not arr or not idx:
        return None
    vals = [arr[i] for i in idx if 0 <= i < len(arr) and math.isfinite(arr[i])]
    return sum(vals) / len(vals) if vals else None


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    import torch  # noqa: F401

    from seq_core.pipeline import load_model_and_tokenizer, resolve_device, resolve_dtype, unload_model
    from seq_core.signals import extract_all_signals, ACT_SIGNALS
    from seq_core.channel_protect import apply_channel_protection
    from seq_core.channel_utils import bucket_by_rank
    from seq_core.quantizers import get_backend
    from seq_core.sensitivity import make_ppl_fn
    from seq_core.stats_utils import spearman

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = _load_prompts(args.calibration_prompts)
    backend = get_backend(args.backend)
    if not backend.is_available():
        raise RuntimeError(f"backend '{args.backend}' not available")

    ppl_fn = make_ppl_fn(
        dataset_name="wikitext2", split="test" if args.ppl_mode == "canonical" else "validation",
        seq_len=args.ppl_seq_len, device=device, dtype=dtype, mode=args.ppl_mode,
        max_examples=None if args.ppl_mode == "canonical" else args.ppl_max_examples,
        full_corpus=(args.ppl_mode == "canonical"), seed=args.seed,
    )

    # signals (per channel) from the FP16 model
    model, tokenizer = load_model_and_tokenizer(args.model, device, dtype, trust_remote_code=bool(args.trust_remote_code))
    signals = extract_all_signals(
        model, tokenizer=tokenizer, prompts=prompts, seq_len=args.calib_seq_len,
        device=device, max_prompts=args.max_calib_prompts, include_activation=bool(prompts),
        return_channels=True,
    )
    linear_names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    unload_model(model, tokenizer)

    if args.layers.strip():
        audit_layers = [s.strip() for s in args.layers.split(",") if s.strip()]
    else:
        rankable = [n for n in linear_names if _channel_arr(signals, n, args.rank_by) is not None]
        step = max(1, len(rankable) // max(1, args.audit_layers))
        audit_layers = rankable[::step][: args.audit_layers]
    LOGGER.info("auditing %d layers: %s", len(audit_layers), audit_layers)

    empty_scores = {n: [] for n in linear_names}

    def quantized_ppl(explicit: Optional[Dict[str, List[int]]]) -> float:
        model, tokenizer = load_model_and_tokenizer(args.model, device, dtype, trust_remote_code=bool(args.trust_remote_code))
        apply_channel_protection(
            model, empty_scores, 0.0, backend, args.base_bits,
            device=device, compute_dtype=dtype, group_size=args.group_size,
            explicit_protected=explicit,
        )
        ppl = ppl_fn(model, tokenizer)
        unload_model(model, tokenizer)
        return ppl

    baseline_ppl = quantized_ppl(None)
    LOGGER.info("all-%dbit baseline ppl = %.4f", args.base_bits, baseline_ppl)
    if not math.isfinite(baseline_ppl):
        raise RuntimeError(
            f"baseline PPL is non-finite at base_bits={args.base_bits} — the audit needs a "
            "finite base. Try a higher --base_bits (e.g. 4) or check the dataset/backend."
        )

    cand_signals = [s for s in ACT_SIGNALS if not s.endswith("_pp")]
    records: List[Dict[str, Any]] = []
    for L in audit_layers:
        arr = _channel_arr(signals, L, args.rank_by)
        if not arr:
            continue
        buckets = bucket_by_rank(arr, args.num_buckets)
        for b_idx, bucket in enumerate(buckets):
            ppl = quantized_ppl({L: bucket})
            delta = baseline_ppl - ppl  # >0 == protecting this bucket helped
            rec = {
                "layer": L, "bucket_rank": b_idx, "bucket_size": len(bucket),
                "ppl": ppl, "protect_value": delta,
                "signal_means": {s: _bucket_mean(_channel_arr(signals, L, s), bucket) for s in cand_signals},
            }
            records.append(rec)
            LOGGER.info("%s bucket %d/%d value=%+.4f", L, b_idx, len(buckets), delta)

    # aggregate: protection value by bucket rank (is rank_by monotonic?)
    by_rank: Dict[int, List[float]] = {}
    for r in records:
        by_rank.setdefault(r["bucket_rank"], []).append(r["protect_value"])
    rank_curve = {k: sum(v) / len(v) for k, v in sorted(by_rank.items())}

    # which signal best predicts measured protection value (across all buckets)?
    signal_corr: Dict[str, Any] = {}
    values = [r["protect_value"] for r in records]
    for s in cand_signals:
        xs, ys = [], []
        for r in records:
            m = r["signal_means"].get(s)
            if m is not None and math.isfinite(m) and math.isfinite(r["protect_value"]):
                xs.append(m)
                ys.append(r["protect_value"])
        signal_corr[s] = {"spearman": spearman(xs, ys), "n": len(xs)}

    payload = {
        "model": args.model, "backend": backend.name, "base_bits": args.base_bits,
        "rank_by": args.rank_by, "num_buckets": args.num_buckets, "ppl_mode": args.ppl_mode,
        "baseline_ppl": baseline_ppl, "audit_layers": audit_layers,
        "protect_value_by_rank": rank_curve,
        "signal_vs_measured_value": signal_corr,
        "records": records,
    }
    (out_dir / "audit.json").write_text(json.dumps(payload, indent=2))
    _write_md(out_dir / "audit.md", payload)
    LOGGER.info("wrote %s", out_dir / "audit.md")
    return 0


def _write_md(path: Path, payload: Dict[str, Any]) -> None:
    L = [f"# Per-channel sensitivity audit — {payload['model']}", ""]
    L.append(f"Base {payload['base_bits']}-bit ({payload['backend']}), channels ranked by "
             f"`{payload['rank_by']}`, {payload['num_buckets']} buckets, {payload['ppl_mode']} PPL. "
             f"All-{payload['base_bits']}bit PPL = {payload['baseline_ppl']:.4f}.")
    L.append("")
    L.append("## Protection value by rank bucket (0 = highest `%s`)" % payload["rank_by"])
    L.append("")
    L.append("| bucket rank | mean ΔPPL recovered |")
    L.append("|---|---|")
    for k, v in payload["protect_value_by_rank"].items():
        L.append(f"| {k} | {v:+.4f} |")
    L.append("")
    L.append("> Monotonic decreasing ⇒ the ranking signal orders channels correctly. "
             "A non-monotonic / flat curve ⇒ the signal mis-ranks some channels (headroom).")
    L.append("")
    L.append("## Which signal best predicts measured protection value?")
    L.append("")
    L.append("| signal | Spearman ρ vs ΔPPL | n |")
    L.append("|---|---|---|")
    rows = sorted(payload["signal_vs_measured_value"].items(),
                  key=lambda kv: (kv[1]["spearman"] if kv[1]["spearman"] is not None else -9), reverse=True)
    for s, c in rows:
        rho = "—" if c["spearman"] is None else f"{c['spearman']:.3f}"
        L.append(f"| `{s}` | {rho} | {c['n']} |")
    L.append("")
    L.append("> The signal with the highest ρ is the best channel-importance predictor. "
             "If it beats `act_max`, it is a candidate to beat the outlier-magnitude heuristic.")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
