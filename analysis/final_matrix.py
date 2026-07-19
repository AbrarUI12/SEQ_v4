#!/usr/bin/env python3
"""Manifest-driven validation and early-gate reporting for final SEQ sweeps."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


T_CRIT_95 = {2: 4.3026527297}


def load_matrix(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "models", "bases", "fractions", "scalar_selectors", "set_selectors",
        "gate_selectors", "deterministic_seed", "random_seeds", "uniform_hqq_bits",
        "value_tier", "matched_bits_tolerance", "gate_required_wins_per_stratum",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"matrix is missing keys: {', '.join(missing)}")
    if len(set(data["random_seeds"])) < 3:
        raise ValueError("matrix requires at least three distinct random seeds")
    return data


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _number(value: Any) -> float | None:
    return float(value) if _finite(value) else None


def _same_number(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    a, b = _number(left), _number(right)
    return a is not None and b is not None and abs(a - b) <= tolerance


def _tier_budget(row: dict[str, Any]) -> float | None:
    label = row.get("tiers")
    if not isinstance(label, str) or not label.startswith("budget="):
        return None
    return _number(label.split("=", 1)[1])


def collect_observations(roots: Iterable[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    observations: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    files: set[Path] = set()
    for root in roots:
        if root.is_file() and root.name == "channel_pareto.json":
            files.add(root)
        elif root.exists():
            files.update(root.glob("**/channel_pareto.json"))
    for path in sorted(files):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            parse_errors.append({"path": str(path), "message": f"invalid JSON: {exc}"})
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            parse_errors.append({"path": str(path), "message": "missing results array"})
            continue
        payload_seed = payload.get("seed")
        for row_index, row in enumerate(payload["results"]):
            if not isinstance(row, dict):
                parse_errors.append({"path": str(path), "row": row_index, "message": "result is not an object"})
                continue
            storage = row.get("storage") if isinstance(row.get("storage"), dict) else {}
            bits = storage.get("actual_weight_bits_per_param", row.get("actual_effective_bits"))
            observations.append({
                "model": payload.get("model"),
                "base": payload.get("base_quantizer", "hqq"),
                "base_bits": payload.get("base_bits"),
                "selector": row.get("signal"),
                "fraction": _number(row.get("k_frac")),
                "budget": _tier_budget(row),
                "seed": row.get("seed", payload_seed),
                "bits": _number(bits),
                "ppl": _number(row.get("ppl")),
                "storage": storage,
                "errors": row.get("errors", 0),
                "path": str(path),
                "row": row_index,
            })
    return observations, parse_errors


def nominal_cell_count(matrix: dict[str, Any]) -> int:
    selectors = len(matrix["scalar_selectors"]) + len(matrix["set_selectors"])
    core = len(matrix["models"]) * len(matrix["bases"]) * selectors * len(matrix["fractions"])
    uniform = len(matrix["models"]) * len(matrix["uniform_hqq_bits"])
    value = len(matrix["value_tier"]["models"]) * len(matrix["value_tier"]["budgets"])
    return core + uniform + value


def _cell(**values: Any) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def validate_matrix(matrix: dict[str, Any], roots: Iterable[Path]) -> dict[str, Any]:
    observations, parse_errors = collect_observations(roots)
    tolerance = float(matrix["matched_bits_tolerance"])
    models = set(matrix["models"])
    base_specs = {(item["name"], int(item["base_bits"])) for item in matrix["bases"]}
    selectors = set(matrix["scalar_selectors"]) | set(matrix["set_selectors"])
    deterministic = selectors - {"random"}
    fractions = [float(value) for value in matrix["fractions"]]
    nonzero = [value for value in fractions if value != 0.0]
    det_seed = int(matrix["deterministic_seed"])
    random_seeds = {int(value) for value in matrix["random_seeds"]}

    invalid: list[dict[str, Any]] = []
    for obs in observations:
        messages: list[str] = []
        if obs["model"] not in models:
            messages.append("unexpected model")
        if obs["ppl"] is None:
            messages.append("non-finite PPL")
        if obs["bits"] is None or not 0 < float(obs["bits"]) <= 16:
            messages.append("missing/invalid actual weight bits")
        if not obs["storage"]:
            messages.append("missing storage breakdown")
        if obs["errors"]:
            messages.append("sweep errors present")
        if messages:
            invalid.append({**_cell(model=obs["model"], base=obs["base"], selector=obs["selector"],
                                    fraction=obs["fraction"], budget=obs["budget"], seed=obs["seed"]),
                            "path": obs["path"], "message": "; ".join(messages)})

    def matching(*, model: str, base: str, bits: int, selector: str | None = None,
                 fraction: float | None = None, budget: float | None = None,
                 seed: int | None = None) -> list[dict[str, Any]]:
        found = []
        for obs in observations:
            if obs["model"] != model or obs["base"] != base or int(obs["base_bits"] or -1) != bits:
                continue
            if selector is not None and obs["selector"] != selector:
                continue
            if fraction is not None and not _same_number(obs["fraction"], fraction):
                continue
            if budget is not None and not _same_number(obs["budget"], budget):
                continue
            if seed is not None and obs["seed"] != seed:
                continue
            found.append(obs)
        return found

    missing: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    inconsistent_anchors: list[dict[str, Any]] = []
    for model in matrix["models"]:
        for base, bits in sorted(base_specs):
            anchors = matching(model=model, base=base, bits=bits, fraction=0.0)
            if not anchors:
                missing.append(_cell(model=model, base=base, base_bits=bits, fraction=0.0,
                                     message="missing shared zero-budget anchor"))
            elif len(anchors) > 1:
                ref = anchors[0]
                if any(abs(float(item["bits"]) - float(ref["bits"])) > tolerance or
                       abs(float(item["ppl"]) - float(ref["ppl"])) > 1e-8
                       for item in anchors[1:] if item["bits"] is not None and item["ppl"] is not None):
                    inconsistent_anchors.append(_cell(model=model, base=base, base_bits=bits,
                                                       message="zero-budget anchors disagree"))
            for selector in sorted(deterministic):
                for fraction in nonzero:
                    found = matching(model=model, base=base, bits=bits, selector=selector,
                                     fraction=fraction, seed=det_seed)
                    cell = _cell(model=model, base=base, base_bits=bits, selector=selector,
                                 fraction=fraction, seed=det_seed)
                    if not found:
                        missing.append({**cell, "message": "missing deterministic cell"})
                    elif len(found) > 1:
                        duplicates.append({**cell, "paths": [item["path"] for item in found]})
            if "random" in selectors:
                for fraction in nonzero:
                    for seed in sorted(random_seeds):
                        found = matching(model=model, base=base, bits=bits, selector="random",
                                         fraction=fraction, seed=seed)
                        cell = _cell(model=model, base=base, base_bits=bits, selector="random",
                                     fraction=fraction, seed=seed)
                        if not found:
                            missing.append({**cell, "message": "missing random seed replicate"})
                        elif len(found) > 1:
                            duplicates.append({**cell, "paths": [item["path"] for item in found]})

    for model in matrix["models"]:
        for bits in matrix["uniform_hqq_bits"]:
            if not matching(model=model, base="hqq", bits=int(bits), fraction=0.0):
                missing.append(_cell(model=model, base="hqq", base_bits=int(bits), family="uniform",
                                     fraction=0.0, message="missing uniform HQQ cell"))
    for model in matrix["value_tier"]["models"]:
        for budget in matrix["value_tier"]["budgets"]:
            found = matching(model=model, base="hqq", bits=4, selector="tier_alloc",
                             budget=float(budget), seed=det_seed)
            cell = _cell(model=model, base="hqq", base_bits=4, family="value_tier",
                         selector="tier_alloc", budget=float(budget), seed=det_seed)
            if not found:
                missing.append({**cell, "message": "missing value-tier cell"})
            elif len(found) > 1:
                duplicates.append({**cell, "paths": [item["path"] for item in found]})

    fatal = bool(parse_errors or invalid or missing or duplicates or inconsistent_anchors)
    return {
        "status": "FAIL" if fatal else "PASS",
        "nominal_expanded_cells": nominal_cell_count(matrix),
        "observations": len(observations),
        "files": len({item["path"] for item in observations}),
        "parse_errors": parse_errors,
        "invalid": invalid,
        "missing": missing,
        "duplicates": duplicates,
        "inconsistent_anchors": inconsistent_anchors,
    }


def _sample_ci(values: list[float]) -> dict[str, Any]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return {"mean": mean, "stddev": None, "ci_low": None, "ci_high": None, "n": len(values)}
    stddev = statistics.stdev(values)
    critical = T_CRIT_95.get(len(values) - 1, 1.9599639845)
    half = critical * stddev / math.sqrt(len(values))
    return {"mean": mean, "stddev": stddev, "ci_low": mean - half, "ci_high": mean + half,
            "n": len(values)}


def build_gate_summary(matrix: dict[str, Any], roots: Iterable[Path]) -> dict[str, Any]:
    observations, parse_errors = collect_observations(roots)
    tolerance = float(matrix["matched_bits_tolerance"])
    seeds = [int(value) for value in matrix["random_seeds"]]
    fractions = [float(value) for value in matrix["fractions"] if float(value) != 0.0]
    required_wins = int(matrix["gate_required_wins_per_stratum"])
    errors = list(parse_errors)
    cells: list[dict[str, Any]] = []
    strata: list[dict[str, Any]] = []

    def select(model: str, base: str, bits: int, selector: str, fraction: float,
               seed: int | None = None) -> list[dict[str, Any]]:
        return [obs for obs in observations
                if obs["model"] == model and obs["base"] == base and int(obs["base_bits"] or -1) == bits
                and obs["selector"] == selector and _same_number(obs["fraction"], fraction)
                and (seed is None or obs["seed"] == seed)]

    for model in matrix["models"]:
        for spec in matrix["bases"]:
            base, bits = spec["name"], int(spec["base_bits"])
            counts = {"greedy_vs_greedy_indep": 0, "greedy_vs_residual_max": 0,
                      "greedy_vs_random_ci": 0}
            for fraction in fractions:
                deterministic: dict[str, dict[str, Any]] = {}
                for selector in ("greedy", "greedy_indep", "residual_max"):
                    found = select(model, base, bits, selector, fraction, int(matrix["deterministic_seed"]))
                    if len(found) != 1:
                        errors.append(_cell(model=model, base=base, selector=selector, fraction=fraction,
                                            message=f"expected one gate cell, found {len(found)}"))
                    elif found[0]["ppl"] is None or found[0]["bits"] is None:
                        errors.append(_cell(model=model, base=base, selector=selector, fraction=fraction,
                                            message="gate cell has invalid PPL/bits"))
                    else:
                        deterministic[selector] = found[0]
                random_rows: list[dict[str, Any]] = []
                for seed in seeds:
                    found = select(model, base, bits, "random", fraction, seed)
                    if len(found) != 1:
                        errors.append(_cell(model=model, base=base, selector="random", fraction=fraction,
                                            seed=seed, message=f"expected one random replicate, found {len(found)}"))
                    elif found[0]["ppl"] is not None and found[0]["bits"] is not None:
                        random_rows.append(found[0])
                if len(deterministic) != 3 or len(random_rows) != len(seeds):
                    continue
                greedy = deterministic["greedy"]
                all_rows = list(deterministic.values()) + random_rows
                if any(abs(float(item["bits"]) - float(greedy["bits"])) > tolerance for item in all_rows[1:]):
                    errors.append(_cell(model=model, base=base, fraction=fraction,
                                        message="actual bits do not match within tolerance"))
                    continue
                random_ci = _sample_ci([float(item["ppl"]) for item in random_rows])
                wins = {
                    "greedy_vs_greedy_indep": float(greedy["ppl"]) < float(deterministic["greedy_indep"]["ppl"]),
                    "greedy_vs_residual_max": float(greedy["ppl"]) < float(deterministic["residual_max"]["ppl"]),
                    "greedy_vs_random_ci": float(greedy["ppl"]) < float(random_ci["ci_low"]),
                }
                for key, won in wins.items():
                    counts[key] += int(won)
                cells.append({
                    "model": model, "base": base, "base_bits": bits, "fraction": fraction,
                    "actual_weight_bits": greedy["bits"], "greedy_ppl": greedy["ppl"],
                    "greedy_indep_ppl": deterministic["greedy_indep"]["ppl"],
                    "residual_max_ppl": deterministic["residual_max"]["ppl"],
                    "delta_vs_greedy_indep": float(greedy["ppl"]) - float(deterministic["greedy_indep"]["ppl"]),
                    "delta_vs_residual_max": float(greedy["ppl"]) - float(deterministic["residual_max"]["ppl"]),
                    "random": {**random_ci, "seeds": seeds,
                               "values": [float(item["ppl"]) for item in random_rows]},
                    "wins": wins,
                })
            passes = all(value >= required_wins for value in counts.values())
            strata.append({"model": model, "base": base, "base_bits": bits,
                           "required_wins": required_wins, "win_counts": counts, "passes": passes})

    status = "FAIL" if errors else "PASS"
    framing = "method" if status == "PASS" and strata and all(item["passes"] for item in strata) else "audit"
    return {"status": status, "framing": framing, "criteria": {
        "nonzero_budgets_per_stratum": len(fractions), "required_wins_per_comparator": required_wins,
        "random_rule": "greedy PPL must be below the lower bound of the 95% Student-t CI",
        "matched_bits_tolerance": tolerance,
    }, "strata": strata, "cells": cells, "errors": errors}


def _matrix_markdown(report: dict[str, Any]) -> str:
    lines = ["# Final matrix validation", "", f"Status: **{report['status']}**", "",
             f"Nominal expanded cells (diagnostic only): {report['nominal_expanded_cells']}",
             f"Observed rows: {report['observations']}", ""]
    for section in ("parse_errors", "invalid", "missing", "duplicates", "inconsistent_anchors"):
        lines.extend([f"## {section.replace('_', ' ').title()}", ""])
        values = report[section]
        if not values:
            lines.extend(["None.", ""])
        else:
            lines.extend([f"- `{json.dumps(value, sort_keys=True)}`" for value in values])
            lines.append("")
    return "\n".join(lines)


def _gate_markdown(report: dict[str, Any]) -> str:
    lines = ["# Early selector gate", "", f"Status: **{report['status']}**",
             f"Framing: **{report['framing']}**", "",
             "| model | base | fraction | bits | greedy | greedy_indep | residual_max | random mean [95% CI] |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["cells"]:
        rnd = row["random"]
        lines.append(f"| {row['model']} | {row['base']} | {row['fraction']:.2f} | "
                     f"{row['actual_weight_bits']:.6f} | {row['greedy_ppl']:.4f} | "
                     f"{row['greedy_indep_ppl']:.4f} | {row['residual_max_ppl']:.4f} | "
                     f"{rnd['mean']:.4f} [{rnd['ci_low']:.4f}, {rnd['ci_high']:.4f}] |")
    lines.extend(["", "## Strata", ""])
    for row in report["strata"]:
        lines.append(f"- **{'PASS' if row['passes'] else 'FAIL'}** {row['model']} / {row['base']}: "
                     f"{json.dumps(row['win_counts'], sort_keys=True)}")
    if report["errors"]:
        lines.extend(["", "## Errors", ""] + [f"- `{json.dumps(item, sort_keys=True)}`" for item in report["errors"]])
    return "\n".join(lines) + "\n"


def _write_report(report: dict[str, Any], json_path: Path, markdown_path: Path, markdown: str) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate", "gate"])
    parser.add_argument("--matrix", type=Path, default=Path("configs/final_comparison_matrix.json"))
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    try:
        matrix = load_matrix(args.matrix)
        if args.command == "validate":
            report = validate_matrix(matrix, args.roots)
            markdown = _matrix_markdown(report)
        else:
            report = build_gate_summary(matrix, args.roots)
            markdown = _gate_markdown(report)
        _write_report(report, args.json, args.markdown, markdown)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "PASS" else 1
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
