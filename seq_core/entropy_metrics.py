#!/usr/bin/env python3
import json
import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)

# Locked constants for degeneracy detection
MIN_SAMPLES = 4096
MAX_NAN_FRAC = 0.01
MAX_BOOTSTRAP_MULTINOMIAL_N = np.iinfo(np.int32).max - 1


@dataclass
class RunningStats:
    count: int = 0
    total: int = 0
    mean: float = 0.0
    m2: float = 0.0
    nonfinite_count: int = 0
    cast_count: int = 0
    max_abs: float = 0.0

    def update(self, values: torch.Tensor) -> None:
        flat = values.reshape(-1)
        self.total += int(flat.numel())
        if flat.numel() == 0:
            return
        mask = torch.isfinite(flat)
        if mask.sum().item() == 0:
            self.nonfinite_count += int(flat.numel())
            return
        finite = flat[mask]
        finite_count = int(finite.numel())
        self.nonfinite_count += int(flat.numel() - finite_count)
        if finite_count == 0:
            return

        # Promote to higher precision before squaring to avoid overflow.
        finite = finite.to(dtype=torch.float64)
        self.cast_count += int(finite.numel())
        try:
            max_abs = float(finite.abs().max().item())
            if max_abs > self.max_abs:
                self.max_abs = max_abs
        except Exception:
            pass

        batch_count = int(finite.numel())
        batch_mean = float(finite.mean().item())
        diff = finite - batch_mean
        batch_m2 = float((diff * diff).sum().item())

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total_count
        self.m2 = self.m2 + batch_m2 + delta * delta * self.count * batch_count / total_count
        self.count = total_count

    @property
    def variance(self) -> float:
        if self.count <= 1:
            return 0.0
        return self.m2 / self.count

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)


def parse_block_index(name: str) -> Optional[int]:
    patterns = [
        r"(?:layers|layer|h|blocks|block)\.(\d+)",
        r"(?:encoder|decoder)\.(\d+)",
    ]
    for pat in patterns:
        match = re.search(pat, name)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                continue
    return None


def _shannon_entropy_from_counts(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    probs = counts[counts > 0] / total
    if probs.size == 0:
        return 0.0
    return float(-(probs * np.log2(probs)).sum())


def _bootstrap_entropy_from_counts(counts: np.ndarray, iters: int) -> float:
    if iters <= 0:
        return 0.0
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if total <= 0:
        return 0.0

    sample_total = total
    if total > MAX_BOOTSTRAP_MULTINOMIAL_N:
        # On Windows, numpy multinomial is limited by C long. Downscale
        # histogram counts deterministically while preserving bin proportions.
        probs64 = counts.astype(np.float64) / float(total)
        scaled = probs64 * float(MAX_BOOTSTRAP_MULTINOMIAL_N)
        sample_counts = np.floor(scaled).astype(np.int64)
        remainder = int(MAX_BOOTSTRAP_MULTINOMIAL_N - int(sample_counts.sum()))
        if remainder > 0:
            frac = scaled - sample_counts
            add_idx = np.argsort(-frac)[:remainder]
            sample_counts[add_idx] += 1
        counts = sample_counts
        sample_total = int(counts.sum())
        LOGGER.warning(
            "Downscaled bootstrap multinomial n from %d to %d for platform compatibility",
            total,
            sample_total,
        )

    probs = counts / float(sample_total)
    entropies = []
    for _ in range(iters):
        sample = np.random.multinomial(sample_total, probs)
        entropies.append(_shannon_entropy_from_counts(sample))
    return float(np.std(np.array(entropies, dtype=np.float64)))


def compute_weight_entropy(
    model: torch.nn.Module,
    bins: int = 256,
    clip: float = 6.0,
    eps: float = 1e-5,
    exclude_embeddings: bool = True,
    include_linear: bool = True,
    degeneracy_mode: str = "rms",
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    module_map = dict(model.named_modules())
    for name, module in module_map.items():
        if exclude_embeddings and isinstance(module, torch.nn.Embedding):
            continue
        if include_linear and not isinstance(module, torch.nn.Linear):
            continue
        if not hasattr(module, "weight"):
            continue

        weight = module.weight.detach()
        total = int(weight.numel())
        if total == 0:
            results[name] = {
                "entropy_bits": None,
                "mean": None,
                "std": None,
                "num_params": 0,
                "shape": list(weight.shape),
                "flags": {
                    "missing": False,
                    "degenerate": True,
                    "nan_frac": 1.0,
                    "sample_count": 0,
                    "unreliable": True,
                },
            }
            continue

        flat = weight.reshape(-1).to(dtype=torch.float32)
        mask = torch.isfinite(flat)
        finite = flat[mask]
        sample_count = int(finite.numel())
        nan_frac = 1.0 - (float(sample_count) / float(total))

        if sample_count == 0:
            mean = None
            std = None
            variance = 0.0
        else:
            mean = float(finite.mean().item())
            std = float(finite.std(unbiased=False).item())
            variance = std * std

        if degeneracy_mode == "old":
            degenerate = variance < eps
        else:
            floor2 = 1e-16
            tiny_factor = 1e-5
            if sample_count == 0:
                rms2 = 0.0
            else:
                rms2 = float((finite * finite).mean().item())
            scale2 = max(rms2, floor2)
            degenerate = variance < (scale2 * tiny_factor)
        unreliable = sample_count < MIN_SAMPLES or degenerate or nan_frac > MAX_NAN_FRAC

        entropy_bits = None
        if sample_count > 0:
            norm = (finite - mean) / (std + eps)
            norm = torch.clamp(norm, -clip, clip)
            counts = torch.histc(norm, bins=bins, min=-clip, max=clip)
            entropy_bits = _shannon_entropy_from_counts(counts.cpu().numpy())

        results[name] = {
            "entropy_bits": entropy_bits,
            "mean": mean,
            "std": std,
            "num_params": int(weight.numel()),
            "shape": list(weight.shape),
            "flags": {
                "missing": False,
                "degenerate": bool(degenerate),
                "nan_frac": float(nan_frac),
                "sample_count": int(sample_count),
                "unreliable": bool(unreliable),
            },
        }
        if unreliable:
            LOGGER.warning("Weight entropy unreliable for %s", name)
    return results


def _per_token_entropy(
    tokens: torch.Tensor,
    mean: float,
    std: float,
    bins: int,
    clip: float,
    eps: float,
    bin_edges: torch.Tensor,
) -> torch.Tensor:
    values = tokens.to(dtype=torch.float32)
    mask = torch.isfinite(values)
    values = torch.where(mask, values, torch.zeros_like(values))
    values = (values - mean) / (std + eps)
    values = torch.clamp(values, -clip, clip)

    idx = torch.bucketize(values, bin_edges, right=False) - 1
    idx = torch.clamp(idx, 0, bins - 1)

    num_tokens, width = idx.shape
    counts = torch.zeros((num_tokens, bins), device=idx.device, dtype=torch.int32)
    ones = torch.ones_like(idx, dtype=torch.int32)
    counts.scatter_add_(1, idx, ones)

    probs = counts.to(dtype=torch.float32)
    denom = probs.sum(dim=1, keepdim=True).clamp_min(1.0)
    probs = probs / denom
    ent = -(probs * torch.log2(probs.clamp_min(1e-12))).sum(dim=1)
    return ent


def collect_activation_entropy(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    seq_len: int,
    device: str,
    dtype: torch.dtype,
    bins: int = 256,
    bootstrap_iters: int = 20,
    clip: float = 6.0,
    eps: float = 1e-5,
    summary_path: Optional[str] = None,
    degeneracy_mode: str = "rms",
) -> Dict[str, Dict[str, Any]]:
    model.eval()
    stats: Dict[str, RunningStats] = {}
    linear_names = []

    module_map = dict(model.named_modules())
    for name, module in module_map.items():
        if isinstance(module, torch.nn.Linear):
            stats[name] = RunningStats()
            linear_names.append(name)

    def make_hook_pass1(name: str):
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            stats[name].update(output.detach())
        return hook

    hooks = []
    for name in linear_names:
        module = module_map[name]
        hooks.append(module.register_forward_hook(make_hook_pass1(name)))

    with torch.no_grad():
        for prompt in prompts:
            if not prompt.strip():
                continue
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=seq_len,
            )
            inputs = inputs.to(next(model.parameters()).device)
            _ = model(**inputs)

    for hook in hooks:
        hook.remove()

    stats_summary: Dict[str, Dict[str, Any]] = {}
    nonfinite_modules = []
    for name, running in stats.items():
        total = running.total
        finite = running.count
        nan_frac = 1.0 if total == 0 else 1.0 - (float(finite) / float(total))
        variance = running.variance
        mean = running.mean
        std = running.std
        nonfinite_stats = (
            finite == 0
            or not math.isfinite(mean)
            or not math.isfinite(std)
            or not math.isfinite(variance)
        )
        if nonfinite_stats:
            nonfinite_modules.append(name)
        if degeneracy_mode == "old":
            degenerate = True if nonfinite_stats else (variance < eps)
        else:
            floor2 = 1e-16
            tiny_factor = 1e-5
            if nonfinite_stats:
                degenerate = True
            else:
                rms2 = variance + (mean * mean)
                scale2 = max(rms2, floor2)
                degenerate = variance < (scale2 * tiny_factor)
        unreliable = (
            finite < MIN_SAMPLES
            or degenerate
            or nan_frac > MAX_NAN_FRAC
            or nonfinite_stats
        )
        stats_summary[name] = {
            "mean": mean,
            "std": std,
            "variance": variance,
            "sample_count": finite,
            "total_count": total,
            "nan_frac": nan_frac,
            "degenerate": degenerate,
            "unreliable": unreliable,
            "nonfinite_stats": nonfinite_stats,
            "nonfinite_count": int(running.nonfinite_count),
            "cast_count": int(running.cast_count),
            "max_abs": float(running.max_abs),
        }
        if unreliable:
            LOGGER.warning("Activation stats unreliable for %s", name)

    hist_counts: Dict[str, np.ndarray] = {}
    per_token_entropies: Dict[str, List[float]] = {name: [] for name in linear_names}
    warnings_by_module: Dict[str, List[str]] = {}
    bin_edges = torch.linspace(-clip, clip, bins + 1, device=device, dtype=torch.float32)

    def make_hook_pass2(name: str):
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            summary = stats_summary[name]
            std = summary.get("std")
            mean = summary.get("mean")
            if summary["unreliable"]:
                return
            if std is None or mean is None or not math.isfinite(std) or not math.isfinite(mean) or std < eps:
                summary["unreliable"] = True
                summary["nonfinite_stats"] = True
                warnings_by_module.setdefault(name, []).append("nonfinite_or_small_std")
                return
            mean = summary["mean"]
            std = summary["std"]
            values = output.detach().to(dtype=torch.float32)
            flat = values.reshape(-1)
            mask = torch.isfinite(flat)
            flat = flat[mask]
            if flat.numel() == 0:
                return
            norm = (flat - mean) / (std + eps)
            norm = torch.clamp(norm, -clip, clip)
            idx = torch.bucketize(norm, bin_edges, right=False) - 1
            idx = torch.clamp(idx, 0, bins - 1)
            counts = torch.bincount(idx, minlength=bins).cpu().numpy()
            if name not in hist_counts:
                hist_counts[name] = counts
            else:
                hist_counts[name] += counts

            if values.dim() == 1:
                tokens = values.unsqueeze(0)
            elif values.dim() == 2:
                tokens = values
            else:
                tokens = values.reshape(-1, values.shape[-1])
            ent = _per_token_entropy(tokens, mean, std, bins, clip, eps, bin_edges)
            per_token_entropies[name].extend(ent.detach().cpu().tolist())
        return hook

    hooks = []
    for name in linear_names:
        module = module_map[name]
        hooks.append(module.register_forward_hook(make_hook_pass2(name)))

    with torch.no_grad():
        for prompt in prompts:
            if not prompt.strip():
                continue
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=seq_len,
            )
            inputs = inputs.to(next(model.parameters()).device)
            _ = model(**inputs)

    for hook in hooks:
        hook.remove()

    results: Dict[str, Dict[str, Any]] = {}
    for name in linear_names:
        summary = stats_summary[name]
        unreliable = summary["unreliable"]
        entropy_bits = None
        entropy_std = None
        tail_p95 = None
        if not unreliable and name in hist_counts:
            counts = hist_counts[name]
            entropy_bits = _shannon_entropy_from_counts(counts)
            entropy_std = _bootstrap_entropy_from_counts(counts, bootstrap_iters)
        if per_token_entropies.get(name):
            tail_p95 = float(np.percentile(np.array(per_token_entropies[name]), 95))

        results[name] = {
            "entropy_bits": entropy_bits,
            "entropy_std": entropy_std,
            "tail_entropy_p95": tail_p95,
            "mean": summary["mean"],
            "std": summary["std"],
            "sample_count": int(summary["sample_count"]),
            "nonfinite_count": summary.get("nonfinite_count"),
            "cast_count": summary.get("cast_count"),
            "max_abs": summary.get("max_abs"),
            "flags": {
                "missing": False,
                "degenerate": bool(summary["degenerate"]),
                "nan_frac": float(summary["nan_frac"]),
                "sample_count": int(summary["sample_count"]),
                "unreliable": bool(summary["unreliable"]),
                "nonfinite_stats": bool(summary.get("nonfinite_stats")),
            },
            "warnings": warnings_by_module.get(name, []),
        }
        if summary["unreliable"]:
            LOGGER.warning("Activation entropy unreliable for %s", name)

    nonfinite_count = len(nonfinite_modules)
    forced_unreliable = sum(1 for name in nonfinite_modules if stats_summary[name]["unreliable"])
    if nonfinite_count:
        LOGGER.warning(
            "Activation stats non-finite for %d modules; forced unreliable: %d",
            nonfinite_count,
            forced_unreliable,
        )
    if summary_path:
        try:
            payload = {
                "num_modules": len(linear_names),
                "nonfinite_stats_modules": nonfinite_count,
                "forced_unreliable_nonfinite": forced_unreliable,
                "nonfinite_modules": nonfinite_modules,
            }
            with open(summary_path, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as exc:
            LOGGER.warning("Failed to write activation entropy summary: %s", exc)

    return results


def compute_percentile_thresholds(
    weight_table: Dict[str, Dict[str, Any]],
    act_table: Dict[str, Dict[str, Any]],
    weight_high_pct: float,
    act_high_pct: float,
) -> Dict[str, Any]:
    weight_vals = []
    act_vals = []
    for name, row in weight_table.items():
        if row.get("entropy_bits") is None:
            continue
        if row.get("flags", {}).get("unreliable"):
            continue
        weight_vals.append(row["entropy_bits"])
    for name, row in act_table.items():
        if row.get("entropy_bits") is None:
            continue
        if row.get("flags", {}).get("unreliable"):
            continue
        act_vals.append(row["entropy_bits"])

    weight_cut = float(np.percentile(np.array(weight_vals), weight_high_pct * 100.0)) if weight_vals else None
    act_cut = float(np.percentile(np.array(act_vals), act_high_pct * 100.0)) if act_vals else None

    weight_high = {}
    act_high = {}
    for name in set(weight_table.keys()) | set(act_table.keys()):
        w_ent = weight_table.get(name, {}).get("entropy_bits")
        a_ent = act_table.get(name, {}).get("entropy_bits")
        weight_high[name] = bool(weight_cut is not None and w_ent is not None and w_ent >= weight_cut)
        act_high[name] = bool(act_cut is not None and a_ent is not None and a_ent >= act_cut)

    return {
        "weight_high_pct": weight_high_pct,
        "act_high_pct": act_high_pct,
        "weight_cutoff": weight_cut,
        "act_cutoff": act_cut,
        "weight_high": weight_high,
        "act_high": act_high,
    }


def compute_weight_magnitude(
    model: torch.nn.Module,
    exclude_embeddings: bool = True,
    include_linear: bool = True,
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for name, module in model.named_modules():
        if exclude_embeddings and isinstance(module, torch.nn.Embedding):
            continue
        if include_linear and not isinstance(module, torch.nn.Linear):
            continue
        if not hasattr(module, "weight"):
            continue
        weight = module.weight.detach().to(dtype=torch.float32)
        mean_abs = float(weight.abs().mean().item()) if weight.numel() > 0 else 0.0
        results[name] = {
            "mean_abs": mean_abs,
            "num_params": int(weight.numel()),
            "shape": list(weight.shape),
        }
    return results


def save_table_json(path: str, table: Dict[str, Dict[str, Any]]) -> None:
    with open(path, "w") as f:
        json.dump(table, f, indent=2)


def save_table_csv(path: str, table: Dict[str, Dict[str, Any]], columns: List[str]) -> None:
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for name, row in table.items():
            out = {"module_name": name}
            for col in columns:
                if col == "module_name":
                    continue
                out[col] = row.get(col)
            writer.writerow(out)
