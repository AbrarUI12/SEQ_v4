#!/usr/bin/env python3
"""Pure-stdlib test for the comparison-table Pareto logic."""
import os
import sys
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.build_comparison import aggregate_random_replicates, pareto_frontier  # noqa: E402
from analysis import build_comparison

FAILS = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    (print("ok  :", msg) if cond else (FAILS.append(msg), print("FAIL:", msg)))


# (bits, ppl); frontier = non-dominated toward (min bits, min ppl)
pts = [(4.0, 11.0), (5.0, 10.3), (6.0, 10.4), (4.0, 12.0), (8.0, 9.8)]
f = set(pareto_frontier(pts))
check(f == {0, 1, 4}, "frontier keeps non-dominated points")
check(2 not in f, "(6,10.4) dominated by (5,10.3)")
check(3 not in f, "(4,12) dominated by (4,11)")

# a strictly dominating point removes all others at higher bits+ppl
pts2 = [(4.0, 10.0), (5.0, 11.0), (6.0, 12.0)]
check(set(pareto_frontier(pts2)) == {0}, "single dominator")

# ties: equal points both non-dominated (no strict improvement)
pts3 = [(4.0, 10.0), (4.0, 10.0)]
check(len(pareto_frontier(pts3)) == 2, "identical points both kept")

check(pareto_frontier([]) == [], "empty -> empty")


def test_require_sweep_points_refuses_baseline_only_without_writing(tmp_path, monkeypatch):
    baselines = tmp_path / "baselines.json"
    baselines.write_text(json.dumps({"model": [{"method": "FP16", "bits": 16, "ppl": 1.0}]}))
    out = tmp_path / "comparison.md"
    csv_path = tmp_path / "comparison.csv"
    json_path = tmp_path / "comparison.json"
    raw_path = tmp_path / "random.json"
    monkeypatch.setattr(sys, "argv", [
        "build_comparison.py", "--sweeps", str(tmp_path / "missing"),
        "--baselines", str(baselines), "--require-sweep-points",
        "--out", str(out), "--csv", str(csv_path), "--json", str(json_path),
        "--random-replicates", str(raw_path),
    ])
    assert build_comparison.main() == 2
    assert not any(path.exists() for path in (out, csv_path, json_path, raw_path))


def test_random_replicates_are_aggregated_with_ci_and_raw_rows():
    rows = {"model": [
        {"method": "SEQ:random(hqq-4b k=0.1)", "bits": 5.0, "nominal_bits": 5.2,
         "model_bits": 6.0, "ppl": ppl, "seed": seed, "source": "sweep",
         "accounting_status": "test"}
        for seed, ppl in ((1234, 10.0), (2345, 11.0), (3456, 12.0))
    ]}
    aggregated, raw = aggregate_random_replicates(rows)
    row = aggregated["model"][0]
    assert row["ppl"] == 11.0
    assert row["n_seeds"] == 3
    assert row["ppl_ci_low"] < row["ppl"] < row["ppl_ci_high"]
    assert {item["seed"] for item in raw} == {1234, 2345, 3456}

print("\n%d checks, %d failures" % (CHECKS, len(FAILS)))
if __name__ == "__main__":
    sys.exit(1 if FAILS else 0)
