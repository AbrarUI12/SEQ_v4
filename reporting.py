#!/usr/bin/env python3
import json
import os
from typing import Any, Dict, List, Optional, Tuple


def _render_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        values = [str(row.get(col, "")) for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _top_bottom(table: Dict[str, Dict[str, Any]], key: str, k: int = 8) -> Dict[str, List[Dict[str, Any]]]:
    items = [(name, row.get(key)) for name, row in table.items() if row.get(key) is not None]
    items = sorted(items, key=lambda x: x[1])
    low = items[:k]
    high = items[-k:][::-1]
    return {
        "low": [{"module": n, key: v} for n, v in low],
        "high": [{"module": n, key: v} for n, v in high],
    }


def read_research_summary(path: str, max_lines: int = 10) -> List[str]:
    try:
        with open(path, "r") as f:
            lines = [line.strip() for line in f if line.strip().startswith("-")]
        return lines[:max_lines]
    except Exception:
        return ["- research_scouting.md not available"]


def _default_eval_summary() -> Dict[str, Any]:
    return {
        "tail_risk": {"exact_span_match_rate": None, "repetition_rate": None, "truncation_rate": None},
        "json_stress": {"success_rate": None},
        "temperature_sweep": {"num_temps": None},
        "long_context": {"num_lengths": None},
        "perplexity": {"loss": None, "ppl": None},
        "size": {"quant_model_dir_bytes": None, "fp16_weight_est_bytes": None},
        "latency": {"latency_sec": None, "tokens_per_sec": None},
        "memory": {"peak_memory_bytes": None},
        "warnings": [],
    }


def read_eval_summary(path: str) -> Tuple[Dict[str, Any], List[str]]:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        warnings = data.get("warnings", [])
        return data, warnings
    except Exception as exc:
        summary = _default_eval_summary()
        return summary, [f"eval_summary_missing: {path} ({exc})"]


def _default_bench_summary() -> Dict[str, Any]:
    return {
        "disk_size_GB": None,
        "disk_size_bytes": None,
        "ppl": {"ppl": None},
        "latency": {"prefill_p50_ms": None, "decode_tokens_per_sec_mean": None},
        "memory": {"peak_allocated_gb": None},
        "effective_bits_per_param": None,
        "size": {"method": None, "total_GB": None, "estimated_GB": None},
        "notes": [],
    }


def read_bench_summary(path: str) -> Tuple[Dict[str, Any], List[str]]:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data, []
    except Exception as exc:
        return _default_bench_summary(), [f"bench_summary_missing: {path} ({exc})"]


def _safe_delta(baseline: Optional[float], quant: Optional[float]) -> Optional[float]:
    if baseline is None or quant is None:
        return None
    try:
        return float(quant) - float(baseline)
    except Exception:
        return None


def _latency_p50(eval_summary: Dict[str, Any]) -> Optional[float]:
    latency = eval_summary.get("latency", {})
    if not isinstance(latency, dict):
        return None
    for key in ("decode_sec", "prefill_sec", "latency_sec"):
        block = latency.get(key)
        if isinstance(block, dict) and block.get("p50") is not None:
            return block.get("p50")
    return None


def _bytes_to_gb(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value) / (1024 ** 3)
    except Exception:
        return None


def build_report(
    metadata: Dict[str, Any],
    thresholds: Dict[str, Any],
    protections: Dict[str, Any],
    weight_table: Dict[str, Dict[str, Any]],
    act_table: Dict[str, Dict[str, Any]],
    precision_table: List[Dict[str, Any]],
    effective_bits: Dict[str, Any],
    eval_baseline: Dict[str, Any],
    eval_quant: Dict[str, Any],
    bench_baseline: Dict[str, Any],
    bench_quant: Dict[str, Any],
    ablation_rows: List[Dict[str, Any]],
    warnings: List[str],
    research_summary_lines: List[str],
) -> str:
    lines = []
    lines.append("# Entropy-Guided Mixed-Precision Report")
    lines.append("")

    lines.append("## Run Metadata")
    lines.append("")
    for key in [
        "run_id",
        "experiment_name",
        "model_name",
        "device",
        "dtype",
        "run_dir",
    ]:
        lines.append(f"- {key}: {metadata.get(key)}")
    lines.append("")

    lines.append("## Thresholds and Protections")
    lines.append("")
    lines.append(f"- weight_high_pct: {thresholds.get('weight_high_pct')}")
    lines.append(f"- act_high_pct: {thresholds.get('act_high_pct')}")
    lines.append(f"- weight_cutoff: {thresholds.get('weight_cutoff')}")
    lines.append(f"- act_cutoff: {thresholds.get('act_cutoff')}")
    lines.append(f"- protections: {json.dumps(protections, sort_keys=True)}")
    lines.append("")

    lines.append("## Tier Distribution")
    lines.append("")
    lines.append(f"- params_by_tier: {effective_bits.get('params_by_tier')}")
    lines.append(f"- effective_bits_per_param: {effective_bits.get('effective_bits')}")
    lines.append("")

    lines.append("## Entropy Extremes")
    lines.append("")
    weight_ext = _top_bottom(weight_table, "entropy_bits")
    act_ext = _top_bottom(act_table, "entropy_bits")
    lines.append("### Weight Entropy (Lowest)")
    lines.append("")
    lines.append(_render_table(weight_ext["low"], ["module", "entropy_bits"]))
    lines.append("")
    lines.append("### Weight Entropy (Highest)")
    lines.append("")
    lines.append(_render_table(weight_ext["high"], ["module", "entropy_bits"]))
    lines.append("")
    lines.append("### Activation Entropy (Lowest)")
    lines.append("")
    lines.append(_render_table(act_ext["low"], ["module", "entropy_bits"]))
    lines.append("")
    lines.append("### Activation Entropy (Highest)")
    lines.append("")
    lines.append(_render_table(act_ext["high"], ["module", "entropy_bits"]))
    lines.append("")

    lines.append("## Evaluation Summary (Baseline vs Quant)")
    lines.append("")
    eval_rows = [
        {
            "metric": "tail_exact_match",
            "baseline": eval_baseline.get("tail_risk", {}).get("exact_span_match_rate"),
            "quant": eval_quant.get("tail_risk", {}).get("exact_span_match_rate"),
        },
        {
            "metric": "tail_repetition_rate",
            "baseline": eval_baseline.get("tail_risk", {}).get("repetition_rate"),
            "quant": eval_quant.get("tail_risk", {}).get("repetition_rate"),
        },
        {
            "metric": "json_success_rate",
            "baseline": eval_baseline.get("json_stress", {}).get("success_rate"),
            "quant": eval_quant.get("json_stress", {}).get("success_rate"),
        },
        {
            "metric": "latency_p50_sec",
            "baseline": _latency_p50(eval_baseline),
            "quant": _latency_p50(eval_quant),
        },
        {
            "metric": "tokens_per_sec_mean",
            "baseline": eval_baseline.get("latency", {}).get("tokens_per_sec", {}).get("mean")
            if isinstance(eval_baseline.get("latency", {}).get("tokens_per_sec"), dict)
            else None,
            "quant": eval_quant.get("latency", {}).get("tokens_per_sec", {}).get("mean")
            if isinstance(eval_quant.get("latency", {}).get("tokens_per_sec"), dict)
            else None,
        },
        {
            "metric": "peak_memory_bytes",
            "baseline": eval_baseline.get("memory", {}).get("peak_memory_bytes"),
            "quant": eval_quant.get("memory", {}).get("peak_memory_bytes"),
        },
    ]
    for row in eval_rows:
        row["delta"] = _safe_delta(row.get("baseline"), row.get("quant"))
        if row["baseline"] is None:
            row["baseline"] = "NA"
        if row["quant"] is None:
            row["quant"] = "NA"
        if row["delta"] is None:
            row["delta"] = "NA"
    lines.append(_render_table(eval_rows, ["metric", "baseline", "quant", "delta"]))
    lines.append("")

    lines.append("## Standard Quantization Metrics")
    lines.append("")
    size_base = bench_baseline.get("disk_size_GB")
    size_quant = bench_quant.get("disk_size_GB")
    if size_base is None:
        size_base = _bytes_to_gb(eval_baseline.get("size", {}).get("fp16_weight_est_bytes"))
    if size_quant is None:
        size_quant = _bytes_to_gb(eval_quant.get("size", {}).get("quant_model_dir_bytes"))
    ppl_base = bench_baseline.get("ppl", {}).get("ppl")
    ppl_quant = bench_quant.get("ppl", {}).get("ppl")
    if ppl_base is None:
        ppl_base = eval_baseline.get("perplexity", {}).get("ppl")
    if ppl_quant is None:
        ppl_quant = eval_quant.get("perplexity", {}).get("ppl")
    tps_base = bench_baseline.get("latency", {}).get("decode_tokens_per_sec_mean")
    tps_quant = bench_quant.get("latency", {}).get("decode_tokens_per_sec_mean")
    prefill_base = bench_baseline.get("latency", {}).get("prefill_p50_ms")
    prefill_quant = bench_quant.get("latency", {}).get("prefill_p50_ms")
    vram_base = bench_baseline.get("memory", {}).get("peak_allocated_gb")
    vram_quant = bench_quant.get("memory", {}).get("peak_allocated_gb")
    resident_vram_base = bench_baseline.get("memory", {}).get("resident_mem_after_load_gb")
    resident_vram_quant = bench_quant.get("memory", {}).get("resident_mem_after_load_gb")
    extra_vram_base = bench_baseline.get("memory", {}).get("extra_peak_over_resident_gb")
    extra_vram_quant = bench_quant.get("memory", {}).get("extra_peak_over_resident_gb")
    eff_base = bench_baseline.get("effective_bits_per_param")
    eff_quant = bench_quant.get("effective_bits_per_param")
    std_rows = [
        {"metric": "disk_size_GB", "baseline": size_base, "quant": size_quant},
        {"metric": "ppl", "baseline": ppl_base, "quant": ppl_quant},
        {"metric": "decode_tokens_per_sec", "baseline": tps_base, "quant": tps_quant},
        {"metric": "prefill_p50_ms", "baseline": prefill_base, "quant": prefill_quant},
        {"metric": "peak_vram_GB", "baseline": vram_base, "quant": vram_quant},
        {"metric": "resident_vram_before_measure_GB", "baseline": resident_vram_base, "quant": resident_vram_quant},
        {"metric": "extra_peak_over_resident_GB", "baseline": extra_vram_base, "quant": extra_vram_quant},
        {"metric": "effective_bits_per_param", "baseline": eff_base, "quant": eff_quant},
    ]
    for row in std_rows:
        row["delta"] = _safe_delta(row.get("baseline"), row.get("quant"))
        if row["baseline"] is None:
            row["baseline"] = "NA"
        if row["quant"] is None:
            row["quant"] = "NA"
        if row["delta"] is None:
            row["delta"] = "NA"
    lines.append(_render_table(std_rows, ["metric", "baseline", "quant", "delta"]))
    if bench_baseline.get("size", {}).get("method") == "estimate_from_num_params":
        lines.append("")
        lines.append("- Note: baseline disk size is estimated from FP16 params (no checkpoint saved).")
    if bench_quant.get("size", {}).get("method") == "actual_dir_size":
        lines.append("- Note: quantized size reflects serialization overhead (bnb artifacts may not shrink).")
    lines.append("")

    lines.append("## Ablation Comparisons (Latest Runs)")
    lines.append("")
    if ablation_rows:
        lines.append(_render_table(ablation_rows, ["experiment", "run_id", "effective_bits", "tail_exact_match", "json_success_rate"]))
    else:
        lines.append("No ablation comparisons available.")
    lines.append("")

    lines.append("## Research Scouting Summary")
    lines.append("")
    for line in research_summary_lines:
        lines.append(line)
    lines.append("")

    lines.append("## Observed Failure Modes and Limitations")
    lines.append("")
    lines.append("- Histogram entropy can be sensitive to clipping and bin count.")
    lines.append("- Calibration shift can reorder entropy rankings.")
    lines.append("- Tail-risk behaviors can be noisy for short prompts.")
    lines.append("- Long-context needle retrieval is sensitive to prompt construction.")
    if warnings:
        lines.append("")
        lines.append("## Warnings")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")

    return "\n".join(lines)


def build_ablation_rows(
    index_rows: List[Dict[str, Any]],
    experiment_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    rows = []
    latest = {row.get("experiment_name"): row for row in index_rows}
    if experiment_names is None:
        experiment_names = list(latest.keys())
    for name in experiment_names:
        row = latest.get(name)
        if not row:
            rows.append(
                {
                    "experiment": name,
                    "run_id": "n/a",
                    "effective_bits": "n/a",
                    "tail_exact_match": "n/a",
                    "json_success_rate": "n/a",
                }
            )
            continue
        rows.append(
            {
                "experiment": name,
                "run_id": row.get("run_id"),
                "effective_bits": row.get("effective_bits"),
                "tail_exact_match": row.get("tail_exact_match"),
                "json_success_rate": row.get("json_success_rate"),
            }
        )
    return rows


def build_allreport(index_rows: List[Dict[str, Any]], runs_root: Optional[str] = None) -> str:
    lines = []
    lines.append("# Suite Allreport")
    lines.append("")
    rows = []
    for row in index_rows:
        run_dir = row.get("run_dir")
        if (not run_dir) and runs_root and row.get("run_id"):
            candidate = os.path.join(runs_root, row.get("run_id"))
            if os.path.isdir(candidate):
                run_dir = candidate
        if not run_dir:
            continue
        base_bench_path = os.path.join(run_dir, "bench_baseline", "bench_summary.json")
        quant_bench_path = os.path.join(run_dir, "bench_quant", "bench_summary.json")
        base_bench, _ = read_bench_summary(base_bench_path)
        quant_bench, _ = read_bench_summary(quant_bench_path)
        base_eval, _ = read_eval_summary(os.path.join(run_dir, "eval_baseline", "eval_summary.json"))
        quant_eval, _ = read_eval_summary(os.path.join(run_dir, "eval_quant", "eval_summary.json"))
        experiment_name = row.get("experiment_name") or row.get("run_name") or "NA"
        base_disk = base_bench.get("disk_size_GB")
        quant_disk = quant_bench.get("disk_size_GB")
        if base_disk is None:
            base_disk = _bytes_to_gb(base_eval.get("size", {}).get("fp16_weight_est_bytes"))
        if quant_disk is None:
            quant_disk = _bytes_to_gb(quant_eval.get("size", {}).get("quant_model_dir_bytes"))
        base_ppl = base_bench.get("ppl", {}).get("ppl")
        quant_ppl = quant_bench.get("ppl", {}).get("ppl")
        if base_ppl is None:
            base_ppl = base_eval.get("perplexity", {}).get("ppl")
        if quant_ppl is None:
            quant_ppl = quant_eval.get("perplexity", {}).get("ppl")
        base_prefill = base_bench.get("latency", {}).get("prefill_p50_ms")
        quant_prefill = quant_bench.get("latency", {}).get("prefill_p50_ms")
        base_tps = base_bench.get("latency", {}).get("decode_tokens_per_sec_mean")
        quant_tps = quant_bench.get("latency", {}).get("decode_tokens_per_sec_mean")
        base_vram = base_bench.get("memory", {}).get("peak_allocated_gb")
        quant_vram = quant_bench.get("memory", {}).get("peak_allocated_gb")
        base_eff = base_bench.get("effective_bits_per_param")
        quant_eff = quant_bench.get("effective_bits_per_param")
        rows.append(
            {
                "experiment": experiment_name,
                "run_id": row.get("run_id"),
                "disk_gb_b": base_disk,
                "disk_gb_q": quant_disk,
                "disk_gb_d": _safe_delta(base_disk, quant_disk),
                "ppl_b": base_ppl,
                "ppl_q": quant_ppl,
                "ppl_d": _safe_delta(base_ppl, quant_ppl),
                "prefill_ms_b": base_prefill,
                "prefill_ms_q": quant_prefill,
                "prefill_ms_d": _safe_delta(base_prefill, quant_prefill),
                "tps_b": base_tps,
                "tps_q": quant_tps,
                "tps_d": _safe_delta(base_tps, quant_tps),
                "vram_gb_b": base_vram,
                "vram_gb_q": quant_vram,
                "vram_gb_d": _safe_delta(base_vram, quant_vram),
                "eff_bits_b": base_eff,
                "eff_bits_q": quant_eff,
                "eff_bits_d": _safe_delta(base_eff, quant_eff),
            }
        )
        for key, value in rows[-1].items():
            if value is None:
                rows[-1][key] = "NA"

    if rows:
        lines.append(
            _render_table(
                rows,
                [
                    "experiment",
                    "run_id",
                    "disk_gb_b",
                    "disk_gb_q",
                    "disk_gb_d",
                    "ppl_b",
                    "ppl_q",
                    "ppl_d",
                    "prefill_ms_b",
                    "prefill_ms_q",
                    "prefill_ms_d",
                    "tps_b",
                    "tps_q",
                    "tps_d",
                    "vram_gb_b",
                    "vram_gb_q",
                    "vram_gb_d",
                    "eff_bits_b",
                    "eff_bits_q",
                    "eff_bits_d",
                ],
            )
        )
    else:
        lines.append("No runs available.")

    return "\n".join(lines)
