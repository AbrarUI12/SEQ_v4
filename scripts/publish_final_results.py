#!/usr/bin/env python3
"""Publish a validated staged comparison with per-file atomic replacement."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def _load_pass(path: Path, label: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise SystemExit(f"{label} is not PASS: {path}")
    return payload


def _atomic_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.publish-", suffix=".tmp", dir=target.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        shutil.copyfile(source, temp)
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    return hashlib.sha256(target.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--gate-summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("results/final_publication.json"))
    args = parser.parse_args()
    stage = args.stage_root.resolve()
    root = args.repo_root.resolve()
    validation = _load_pass(args.validation, "matrix validation")
    gate = _load_pass(args.gate_summary, "gate summary")
    mappings = [
        (stage / "docs" / "COMPARISON.md", root / "docs" / "COMPARISON.md"),
        (stage / "results" / "final_comparison.csv", root / "results" / "final_comparison.csv"),
        (stage / "results" / "final_comparison.json", root / "results" / "final_comparison.json"),
        (stage / "results" / "final_random_replicates.json", root / "results" / "final_random_replicates.json"),
    ]
    figures = sorted((stage / "figures").glob("*")) if (stage / "figures").exists() else []
    mappings.extend((path, root / "figures" / "final_corrected" / path.name) for path in figures if path.is_file())
    missing = [str(source) for source, _ in mappings if not source.is_file()]
    if not any(path.is_file() for path in figures):
        missing.append(str(stage / "figures" / "ppl_vs_actual_bits_*.pdf"))
    if missing:
        raise SystemExit("staged publication files are missing:\n" + "\n".join(missing))
    published = []
    for source, target in mappings:
        digest = _atomic_copy(source, target)
        published.append({"source": str(source), "target": str(target), "sha256": digest})
    manifest = {
        "status": "PASS",
        "published_at": datetime.now(timezone.utc).isoformat(),
        "gate_framing": gate.get("framing"),
        "matrix_nominal_expanded_cells": validation.get("nominal_expanded_cells"),
        "matrix_observations": validation.get("observations"),
        "files": published,
    }
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp_manifest, manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
