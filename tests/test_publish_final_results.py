from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts import publish_final_results


def _stage(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    stage = tmp_path / "stage"
    files = {
        stage / "docs" / "COMPARISON.md": "new comparison\n",
        stage / "results" / "final_comparison.csv": "model,ppl\n",
        stage / "results" / "final_comparison.json": "[]\n",
        stage / "results" / "final_random_replicates.json": "[]\n",
        stage / "figures" / "ppl_vs_actual_bits_model.pdf": "pdf",
    }
    for path, text in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    validation = tmp_path / "validation.json"
    gate = tmp_path / "gate.json"
    validation.write_text(json.dumps({"status": "PASS", "nominal_expanded_cells": 152,
                                      "observations": 160}), encoding="utf-8")
    gate.write_text(json.dumps({"status": "PASS", "framing": "audit"}), encoding="utf-8")
    return stage, tmp_path / "repo", validation, gate


def test_publish_replaces_targets_and_writes_success_manifest_last(tmp_path: Path, monkeypatch):
    stage, repo, validation, gate = _stage(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "publish_final_results.py", "--stage-root", str(stage), "--repo-root", str(repo),
        "--validation", str(validation), "--gate-summary", str(gate),
    ])
    assert publish_final_results.main() == 0
    assert (repo / "docs" / "COMPARISON.md").read_text() == "new comparison\n"
    manifest = json.loads((repo / "results" / "final_publication.json").read_text())
    assert manifest["status"] == "PASS"
    assert manifest["gate_framing"] == "audit"
    assert len(manifest["files"]) == 5


def test_publish_refuses_failed_validation_without_touching_target(tmp_path: Path, monkeypatch):
    stage, repo, validation, gate = _stage(tmp_path)
    target = repo / "docs" / "COMPARISON.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("old\n", encoding="utf-8")
    validation.write_text(json.dumps({"status": "FAIL"}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "publish_final_results.py", "--stage-root", str(stage), "--repo-root", str(repo),
        "--validation", str(validation), "--gate-summary", str(gate),
    ])
    try:
        publish_final_results.main()
    except SystemExit:
        pass
    else:
        raise AssertionError("failed validation should stop publication")
    assert target.read_text(encoding="utf-8") == "old\n"
    assert not (repo / "results" / "final_publication.json").exists()
