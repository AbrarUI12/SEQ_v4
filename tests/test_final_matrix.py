from __future__ import annotations

import json
import sys
from pathlib import Path

from analysis.final_matrix import build_gate_summary, load_matrix, nominal_cell_count, validate_matrix
from analysis import build_comparison


ROOT = Path(__file__).resolve().parents[1]


def _write_sweep(
    root: Path, model: str, base: str, bits: int, selector: str, fractions: list[float],
    seed: int, *, ppl: float,
) -> Path:
    slug = model.rsplit("/", 1)[-1]
    path = root / base / slug / f"b{bits}" / selector / f"seed-{seed}" / "channel_pareto.json"
    rows = []
    for fraction in fractions:
        actual = bits + fraction * 12.0
        rows.append({
            "signal": selector, "seed": seed, "k_frac": fraction, "tiers": None,
            "effective_bits": actual, "actual_effective_bits": actual,
            "ppl": ppl + fraction, "errors": 0,
            "storage": {"actual_weight_bits_per_param": actual,
                        "actual_model_bits_per_parameter": actual + 1.0},
        })
    payload = {"model": model, "base_quantizer": base, "base_bits": bits,
               "seed": seed, "results": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_value(root: Path, model: str, seed: int, budgets: list[float]) -> None:
    path = root / "value_tier" / model.rsplit("/", 1)[-1] / "channel_pareto.json"
    rows = [{
        "signal": "tier_alloc", "seed": seed, "k_frac": None, "tiers": f"budget={budget}",
        "effective_bits": 4 + budget, "actual_effective_bits": 4 + budget,
        "ppl": 10 - budget / 10, "errors": 0,
        "storage": {"actual_weight_bits_per_param": 4 + budget,
                    "actual_model_bits_per_parameter": 5 + budget},
    } for budget in budgets]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": model, "base_quantizer": "hqq", "base_bits": 4,
                                "seed": seed, "results": rows}), encoding="utf-8")


def _complete_tree(tmp_path: Path) -> tuple[dict, Path]:
    matrix = load_matrix(ROOT / "configs" / "final_comparison_matrix.json")
    sweep_root = tmp_path / "sweeps"
    nonzero = [float(value) for value in matrix["fractions"] if float(value)]
    seed = int(matrix["deterministic_seed"])
    for model in matrix["models"]:
        for spec in matrix["bases"]:
            base, bits = spec["name"], int(spec["base_bits"])
            for selector in matrix["scalar_selectors"] + matrix["set_selectors"]:
                if selector == "random":
                    for random_seed in matrix["random_seeds"]:
                        _write_sweep(sweep_root, model, base, bits, selector, nonzero,
                                     int(random_seed), ppl=11.0 + int(random_seed) % 3 / 10)
                else:
                    fractions = [0.0, *nonzero] if selector == "residual_max" else nonzero
                    ppl = 9.0 if selector == "greedy" else (10.0 if selector == "greedy_indep" else 10.5)
                    _write_sweep(sweep_root, model, base, bits, selector, fractions, seed, ppl=ppl)
        for bits in matrix["uniform_hqq_bits"]:
            if int(bits) != 4:
                _write_sweep(sweep_root / "uniform", model, "hqq", int(bits), "act_max", [0.0],
                             seed, ppl=10.5)
    for model in matrix["value_tier"]["models"]:
        _write_value(sweep_root, model, seed,
                     [float(value) for value in matrix["value_tier"]["budgets"]])
    return matrix, sweep_root


def test_manifest_validation_accepts_shared_zero_anchors_and_seed_replicates(tmp_path: Path):
    matrix, sweep_root = _complete_tree(tmp_path)
    report = validate_matrix(matrix, [sweep_root])
    assert nominal_cell_count(matrix) == 152
    assert report["status"] == "PASS", report
    assert report["nominal_expanded_cells"] == 152


def test_manifest_validation_names_the_specific_missing_cell(tmp_path: Path):
    matrix, sweep_root = _complete_tree(tmp_path)
    missing_path = sweep_root / "gptq_llmc" / "Llama-3.2-3B" / "b4" / "greedy" / "seed-1234" / "channel_pareto.json"
    missing_path.unlink()
    report = validate_matrix(matrix, [sweep_root])
    assert report["status"] == "FAIL"
    assert any(item.get("model") == "meta-llama/Llama-3.2-3B"
               and item.get("base") == "gptq_llmc"
               and item.get("selector") == "greedy" for item in report["missing"])


def test_gate_summary_uses_three_seed_t_interval_and_selects_method(tmp_path: Path):
    matrix, sweep_root = _complete_tree(tmp_path)
    report = build_gate_summary(matrix, [sweep_root])
    assert report["status"] == "PASS", report
    assert report["framing"] == "method"
    assert len(report["strata"]) == 4
    assert all(row["passes"] for row in report["strata"])
    assert all(row["random"]["n"] == 3 for row in report["cells"])
    assert all(row["random"]["ci_low"] < row["random"]["mean"] < row["random"]["ci_high"]
               for row in report["cells"])


def test_comparison_end_to_end_aggregates_random_replicates(tmp_path: Path, monkeypatch):
    matrix, sweep_root = _complete_tree(tmp_path)
    baselines = tmp_path / "baselines.json"
    baselines.write_text(json.dumps({model: [{"method": "FP16", "bits": 16.0, "ppl": 8.0}]
                                     for model in matrix["models"]}), encoding="utf-8")
    out = tmp_path / "comparison.md"
    csv_path = tmp_path / "comparison.csv"
    json_path = tmp_path / "comparison.json"
    raw_path = tmp_path / "random_replicates.json"
    monkeypatch.setattr(sys, "argv", [
        "build_comparison.py", "--sweeps", str(sweep_root), "--baselines", str(baselines),
        "--signals", ",".join(matrix["scalar_selectors"] + matrix["set_selectors"] + ["tier_alloc"]),
        "--require-sweep-points", "--out", str(out), "--csv", str(csv_path),
        "--json", str(json_path), "--random-replicates", str(raw_path),
    ])
    assert build_comparison.main() == 0
    rows = json.loads(json_path.read_text(encoding="utf-8"))
    random_rows = [row for row in rows if row["method"].startswith("SEQ:random(")]
    assert random_rows and all(row["n_seeds"] == 3 for row in random_rows)
    assert all(row["ppl_ci_low"] < row["ppl"] < row["ppl_ci_high"] for row in random_rows)
    assert len(json.loads(raw_path.read_text(encoding="utf-8"))) == 48
