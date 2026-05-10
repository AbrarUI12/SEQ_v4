#!/usr/bin/env python3
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .entropy_metrics import parse_block_index

LOGGER = logging.getLogger(__name__)


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _latency_p50(eval_summary: Dict[str, Any]) -> Optional[float]:
    latency = eval_summary.get("latency", {})
    if not isinstance(latency, dict):
        return None
    for key in ("decode_sec", "prefill_sec", "latency_sec"):
        block = latency.get(key)
        if isinstance(block, dict) and block.get("p50") is not None:
            return block.get("p50")
    return None


def _extract_metrics(eval_summary: Dict[str, Any]) -> Dict[str, Optional[float]]:
    latency = eval_summary.get("latency", {})
    tps = latency.get("tokens_per_sec") if isinstance(latency, dict) else None
    memory = eval_summary.get("memory", {})
    peak_mem = None
    resident_mem = None
    extra_peak = None
    if isinstance(memory, dict):
        peak_mem = memory.get("peak_allocated_bytes")
        if peak_mem is None:
            peak_mem = memory.get("peak_memory_bytes")
        resident_mem = memory.get("resident_mem_after_load_bytes")
        if resident_mem is None:
            resident_mem = memory.get("resident_allocated_bytes")
        extra_peak = memory.get("extra_peak_over_resident_bytes")
    return {
        "ppl": eval_summary.get("perplexity", {}).get("ppl"),
        "tail_exact_match": eval_summary.get("tail_risk", {}).get("exact_span_match_rate"),
        "json_success_rate": eval_summary.get("json_stress", {}).get("success_rate"),
        "latency_p50_sec": _latency_p50(eval_summary),
        "tokens_per_sec_mean": tps.get("mean") if isinstance(tps, dict) else None,
        "peak_memory_bytes": peak_mem,
        "resident_mem_after_load_bytes": resident_mem,
        "extra_peak_over_resident_bytes": extra_peak,
    }


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception:
        return {}


def _read_entropy_json(path: Path) -> Dict[str, Dict[str, Any]]:
    return _load_json(path)


def _read_entropy_values(path: Path) -> List[float]:
    if not path.exists():
        return []
    values = []
    try:
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "entropy_bits" not in row:
                    continue
                try:
                    values.append(float(row["entropy_bits"]))
                except Exception:
                    continue
    except Exception:
        return []
    return values


def _plot_bar_comparison(
    out_path: Path,
    labels: List[str],
    baseline_vals: List[float],
    quant_vals: List[float],
    title: str,
) -> None:
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width / 2, baseline_vals, width, label="baseline")
    ax.bar(x + width / 2, quant_vals, width, label="quant")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_simple_bars(
    out_path: Path,
    labels: List[str],
    values: List[float],
    title: str,
    ylabel: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, values, color="#4C78A8")
    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_entropy_heatmap(
    out_path: Path,
    entropy_table: Dict[str, Dict[str, Any]],
    cutoff: Optional[float],
    title: str,
) -> bool:
    entries: List[Tuple[int, str, float]] = []
    for name, row in entropy_table.items():
        value = row.get("entropy_bits")
        if value is None:
            continue
        block = parse_block_index(name)
        if block is None:
            continue
        group = _classify_group(name)
        entries.append((block, group, float(value)))

    if not entries:
        return False

    groups = ["attention_qkv", "attention_o_proj", "mlp_up", "mlp_gate", "mlp_down", "other_linear"]
    blocks = sorted({b for b, _, _ in entries})
    matrix = np.full((len(groups), len(blocks)), np.nan, dtype=np.float32)
    counts = np.zeros_like(matrix, dtype=np.int32)
    for block, group, value in entries:
        if block not in blocks:
            continue
        if group not in groups:
            group = "other_linear"
        r = groups.index(group)
        c = blocks.index(block)
        if np.isnan(matrix[r, c]):
            matrix[r, c] = 0.0
        matrix[r, c] += value
        counts[r, c] += 1
    with np.errstate(invalid="ignore"):
        matrix = np.where(counts > 0, matrix / np.maximum(counts, 1), np.nan)

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(blocks)))
    ax.set_xticklabels(blocks, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(groups)))
    ax.set_yticklabels(groups, fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    if cutoff is not None:
        all_vals = [val for _, _, val in entries]
        high_count = sum(1 for val in all_vals if val >= cutoff)
        fig.text(
            0.01,
            0.01,
            f"cutoff={cutoff:.4f}, high_modules={high_count}",
            ha="left",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def _plot_entropy_1d_heatmap(
    out_path: Path,
    entropy_table: Dict[str, Dict[str, Any]],
    cutoff: Optional[float],
    title: str,
) -> bool:
    items = []
    for name, row in entropy_table.items():
        value = row.get("entropy_bits")
        if value is None:
            continue
        items.append((name, float(value)))
    if not items:
        return False
    items = sorted(items, key=lambda x: x[0])
    values = np.array([v for _, v in items], dtype=np.float32)[None, :]

    fig, ax = plt.subplots(figsize=(12, 2))
    im = ax.imshow(values, aspect="auto", interpolation="nearest")
    ax.set_yticks([0])
    ax.set_yticklabels(["entropy"])
    ax.set_xticks([])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    if cutoff is not None:
        high_count = sum(1 for _, v in items if v >= cutoff)
        fig.text(
            0.01,
            0.01,
            f"cutoff={cutoff:.4f}, high_modules={high_count}",
            ha="left",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def _classify_group(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ["qkv", "q_proj", "k_proj", "v_proj", "c_attn", "query", "key", "value"]):
        return "attention_qkv"
    if "o_proj" in lowered or "out_proj" in lowered or ("c_proj" in lowered and "attn" in lowered):
        return "attention_o_proj"
    if "gate_proj" in lowered or "mlp_gate" in lowered or "gate" in lowered and "mlp" in lowered:
        return "mlp_gate"
    if "up_proj" in lowered or "fc1" in lowered or "w1" in lowered:
        return "mlp_up"
    if "down_proj" in lowered or "fc2" in lowered or "w2" in lowered:
        return "mlp_down"
    return "other_linear"


def _long_context_exact_match(long_context_path: Path) -> Optional[float]:
    data = _load_json(long_context_path)
    results = data.get("results", [])
    candidates: List[Tuple[int, Optional[bool]]] = []
    for entry in results:
        success = entry.get("success")
        if success is None:
            success = entry.get("exact_match") is not None
        if not success:
            continue
        length = entry.get("context_length_tokens") or entry.get("context_length") or 0
        candidates.append((int(length), entry.get("exact_match")))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def plot_run_baseline_vs_quant(
    run_dir: Path,
    eval_baseline: Dict[str, Any],
    eval_quant: Dict[str, Any],
    effective_bits: Dict[str, Any],
    thresholds: Dict[str, Any],
) -> List[Path]:
    run_dir = Path(run_dir)
    fig_dir = run_dir / "report" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    created: List[Path] = []

    base_metrics = _extract_metrics(eval_baseline)
    quant_metrics = _extract_metrics(eval_quant)

    # PPL comparison
    ppl_base = base_metrics.get("ppl")
    ppl_quant = quant_metrics.get("ppl")
    if ppl_base is None or ppl_quant is None:
        LOGGER.warning("Skipping ppl_comparison: missing ppl values")
    else:
        path = fig_dir / "ppl_comparison.png"
        _plot_bar_comparison(
            path,
            ["ppl"],
            [_safe_float(ppl_base)],
            [_safe_float(ppl_quant)],
            "Perplexity (Baseline vs Quant)",
        )
        created.append(path)

    # Quality comparison
    tail_base = base_metrics.get("tail_exact_match")
    tail_quant = quant_metrics.get("tail_exact_match")
    json_base = base_metrics.get("json_success_rate")
    json_quant = quant_metrics.get("json_success_rate")
    needle_base = _long_context_exact_match(run_dir / "eval_baseline" / "long_context.json")
    needle_quant = _long_context_exact_match(run_dir / "eval_quant" / "long_context.json")
    quality_vals = [tail_base, json_base, needle_base, tail_quant, json_quant, needle_quant]
    if any(val is None for val in quality_vals):
        LOGGER.warning("Skipping quality_comparison: missing quality metrics")
    else:
        path = fig_dir / "quality_comparison.png"
        _plot_bar_comparison(
            path,
            ["tail_exact_match", "json_success_rate", "needle_exact_match"],
            [_safe_float(tail_base), _safe_float(json_base), _safe_float(needle_base)],
            [_safe_float(tail_quant), _safe_float(json_quant), _safe_float(needle_quant)],
            "Quality Metrics (Baseline vs Quant)",
        )
        created.append(path)

    # Latency/throughput
    tps_base = base_metrics.get("tokens_per_sec_mean")
    tps_quant = quant_metrics.get("tokens_per_sec_mean")
    p50_base = base_metrics.get("latency_p50_sec")
    p50_quant = quant_metrics.get("latency_p50_sec")
    latency_vals = [tps_base, tps_quant, p50_base, p50_quant]
    if any(val is None for val in latency_vals):
        LOGGER.warning("Skipping latency_throughput: missing latency metrics")
    else:
        path = fig_dir / "latency_throughput.png"
        _plot_bar_comparison(
            path,
            ["tokens_per_sec", "latency_p50_sec"],
            [_safe_float(tps_base), _safe_float(p50_base)],
            [_safe_float(tps_quant), _safe_float(p50_quant)],
            "Latency/Throughput (Baseline vs Quant)",
        )
        created.append(path)

    # Memory footprint
    peak_base = base_metrics.get("peak_memory_bytes")
    peak_quant = quant_metrics.get("peak_memory_bytes")
    total_params = effective_bits.get("total_params")
    effective_bits_val = effective_bits.get("effective_bits")
    fp16_bytes = eval_baseline.get("size", {}).get("fp16_weight_est_bytes")
    quant_dir_bytes = eval_quant.get("size", {}).get("quant_model_dir_bytes")
    if fp16_bytes is None and total_params is not None:
        fp16_bytes = int(total_params) * 2
    quant_theoretical = None
    if total_params is not None and effective_bits_val is not None:
        quant_theoretical = float(total_params) * float(effective_bits_val) / 8.0
    mem_vals = [peak_base, peak_quant, fp16_bytes, quant_theoretical, quant_dir_bytes]
    if any(val is None for val in mem_vals):
        LOGGER.warning("Skipping memory_footprint: missing size/memory metrics")
    else:
        peak_base_gb = float(peak_base) / (1024 ** 3)
        peak_quant_gb = float(peak_quant) / (1024 ** 3)
        fp16_gb = float(fp16_bytes) / (1024 ** 3)
        quant_theoretical_gb = float(quant_theoretical) / (1024 ** 3)
        quant_dir_gb = float(quant_dir_bytes) / (1024 ** 3)
        path = fig_dir / "memory_footprint.png"
        _plot_simple_bars(
            path,
            ["peak_vram_baseline", "peak_vram_quant", "fp16_theoretical", "quant_theoretical", "quant_disk"],
            [peak_base_gb, peak_quant_gb, fp16_gb, quant_theoretical_gb, quant_dir_gb],
            "Memory/Footprint (GB)",
            ylabel="GB",
        )
        created.append(path)

    # Tier distribution
    percent = effective_bits.get("percent_params", {})
    tiers = ["int4", "int8", "fp16"]
    values = [float(percent.get(tier, 0.0)) * 100.0 for tier in tiers]
    path = fig_dir / "tier_distribution.png"
    _plot_simple_bars(path, tiers, values, "Tier Distribution (%)", ylabel="Percent")
    created.append(path)

    # Entropy heatmaps
    weight_table = _read_entropy_json(run_dir / "entropy" / "weight_entropy.json")
    act_table = _read_entropy_json(run_dir / "entropy" / "activation_entropy.json")
    weight_cutoff = thresholds.get("weight_cutoff")
    act_cutoff = thresholds.get("act_cutoff")
    weight_heat = fig_dir / "weight_entropy_heatmap.png"
    if _plot_entropy_heatmap(weight_heat, weight_table, weight_cutoff, "Weight Entropy Heatmap"):
        created.append(weight_heat)
    else:
        fallback = fig_dir / "weight_entropy_heatmap.png"
        if _plot_entropy_1d_heatmap(fallback, weight_table, weight_cutoff, "Weight Entropy Heatmap"):
            created.append(fallback)
        else:
            LOGGER.warning("Skipping weight entropy heatmap: missing data")

    act_heat = fig_dir / "activation_entropy_heatmap.png"
    if _plot_entropy_heatmap(act_heat, act_table, act_cutoff, "Activation Entropy Heatmap"):
        created.append(act_heat)
    else:
        fallback = fig_dir / "activation_entropy_heatmap.png"
        if _plot_entropy_1d_heatmap(fallback, act_table, act_cutoff, "Activation Entropy Heatmap"):
            created.append(fallback)
        else:
            LOGGER.warning("Skipping activation entropy heatmap: missing data")

    return created


def _load_run_summaries(run_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    run_dir = Path(run_dir)
    eval_base = _load_json(run_dir / "eval_baseline" / "eval_summary.json")
    eval_quant = _load_json(run_dir / "eval_quant" / "eval_summary.json")
    effective_bits = _load_json(run_dir / "quant" / "effective_bits.json")
    config = _load_json(run_dir / "config.json")
    return eval_base, eval_quant, effective_bits, config


def plot_compare_two_models(
    run_dir_a: Path,
    run_dir_b: Path,
    out_dir: Path,
    label_a: Optional[str] = None,
    label_b: Optional[str] = None,
) -> List[Path]:
    run_dir_a = Path(run_dir_a)
    run_dir_b = Path(run_dir_b)
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    base_a, quant_a, bits_a, cfg_a = _load_run_summaries(run_dir_a)
    base_b, quant_b, bits_b, cfg_b = _load_run_summaries(run_dir_b)

    if not label_a:
        label_a = cfg_a.get("model_name") or run_dir_a.name
    if not label_b:
        label_b = cfg_b.get("model_name") or run_dir_b.name

    created: List[Path] = []

    # compare_ppl
    ppl_a_base = _extract_metrics(base_a).get("ppl")
    ppl_a_quant = _extract_metrics(quant_a).get("ppl")
    ppl_b_base = _extract_metrics(base_b).get("ppl")
    ppl_b_quant = _extract_metrics(quant_b).get("ppl")
    if None not in (ppl_a_base, ppl_a_quant, ppl_b_base, ppl_b_quant):
        path = fig_dir / "compare_ppl.png"
        labels = [f"{label_a}_base", f"{label_a}_quant", f"{label_b}_base", f"{label_b}_quant"]
        values = [
            _safe_float(ppl_a_base),
            _safe_float(ppl_a_quant),
            _safe_float(ppl_b_base),
            _safe_float(ppl_b_quant),
        ]
        _plot_simple_bars(path, labels, values, "Perplexity Comparison")
        created.append(path)

    # compare_quality
    tail_a_base = _extract_metrics(base_a).get("tail_exact_match")
    tail_a_quant = _extract_metrics(quant_a).get("tail_exact_match")
    tail_b_base = _extract_metrics(base_b).get("tail_exact_match")
    tail_b_quant = _extract_metrics(quant_b).get("tail_exact_match")
    json_a_base = _extract_metrics(base_a).get("json_success_rate")
    json_a_quant = _extract_metrics(quant_a).get("json_success_rate")
    json_b_base = _extract_metrics(base_b).get("json_success_rate")
    json_b_quant = _extract_metrics(quant_b).get("json_success_rate")
    needle_a = _long_context_exact_match(run_dir_a / "eval_quant" / "long_context.json")
    needle_b = _long_context_exact_match(run_dir_b / "eval_quant" / "long_context.json")
    if None not in (
        tail_a_quant,
        tail_b_quant,
        json_a_quant,
        json_b_quant,
        needle_a,
        needle_b,
    ):
        path = fig_dir / "compare_quality.png"
        labels = ["tail_exact_match", "json_success_rate", "needle_exact_match"]
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width / 2, [_safe_float(tail_a_quant), _safe_float(json_a_quant), _safe_float(needle_a)], width, label=label_a)
        ax.bar(x + width / 2, [_safe_float(tail_b_quant), _safe_float(json_b_quant), _safe_float(needle_b)], width, label=label_b)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_title("Quality Comparison (Quant)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        created.append(path)

    # compare_speed
    tps_a = _extract_metrics(quant_a).get("tokens_per_sec_mean")
    tps_b = _extract_metrics(quant_b).get("tokens_per_sec_mean")
    p50_a = _latency_p50(quant_a)
    p50_b = _latency_p50(quant_b)
    if None not in (tps_a, tps_b, p50_a, p50_b):
        path = fig_dir / "compare_speed.png"
        labels = ["tokens_per_sec", "latency_p50_sec"]
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width / 2, [_safe_float(tps_a), _safe_float(p50_a)], width, label=label_a)
        ax.bar(x + width / 2, [_safe_float(tps_b), _safe_float(p50_b)], width, label=label_b)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_title("Speed Comparison (Quant)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        created.append(path)

    # compare_memory
    peak_a = _extract_metrics(quant_a).get("peak_memory_bytes")
    peak_b = _extract_metrics(quant_b).get("peak_memory_bytes")
    fp16_a = base_a.get("size", {}).get("fp16_weight_est_bytes")
    fp16_b = base_b.get("size", {}).get("fp16_weight_est_bytes")
    qdir_a = quant_a.get("size", {}).get("quant_model_dir_bytes")
    qdir_b = quant_b.get("size", {}).get("quant_model_dir_bytes")
    if None not in (peak_a, peak_b, fp16_a, fp16_b, qdir_a, qdir_b):
        path = fig_dir / "compare_memory.png"
        labels = [
            f"{label_a}_peak_vram",
            f"{label_b}_peak_vram",
            f"{label_a}_fp16_bytes",
            f"{label_b}_fp16_bytes",
            f"{label_a}_quant_disk",
            f"{label_b}_quant_disk",
        ]
        values = [
            float(peak_a) / (1024 ** 3),
            float(peak_b) / (1024 ** 3),
            float(fp16_a) / (1024 ** 3),
            float(fp16_b) / (1024 ** 3),
            float(qdir_a) / (1024 ** 3),
            float(qdir_b) / (1024 ** 3),
        ]
        _plot_simple_bars(path, labels, values, "Memory/Footprint Comparison (GB)", ylabel="GB")
        created.append(path)

    # compare_entropy_distributions
    weight_a = _read_entropy_values(run_dir_a / "entropy" / "weight_entropy.csv")
    weight_b = _read_entropy_values(run_dir_b / "entropy" / "weight_entropy.csv")
    act_a = _read_entropy_values(run_dir_a / "entropy" / "activation_entropy.csv")
    act_b = _read_entropy_values(run_dir_b / "entropy" / "activation_entropy.csv")
    if weight_a and weight_b and act_a and act_b:
        path = fig_dir / "compare_entropy_distributions.png"
        fig, ax = plt.subplots(figsize=(8, 4))
        bins = 60
        ax.hist(weight_a, bins=bins, alpha=0.4, label=f"{label_a} weight")
        ax.hist(weight_b, bins=bins, alpha=0.4, label=f"{label_b} weight")
        ax.hist(act_a, bins=bins, alpha=0.4, label=f"{label_a} act")
        ax.hist(act_b, bins=bins, alpha=0.4, label=f"{label_b} act")
        ax.set_title("Entropy Distributions")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        created.append(path)

    return created
