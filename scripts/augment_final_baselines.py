#!/usr/bin/env python3
"""Add measured FP16 and uniform-HQQ rows to an LLMC baseline index."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _baseline_row(payload: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    storage = result.get("storage") if isinstance(result.get("storage"), dict) else None
    return {
        "method": f"HQQ-{int(payload['base_bits'])} uniform",
        "bits": float(result.get("actual_effective_bits", result["effective_bits"])),
        "nominal_bits": float(result["effective_bits"]),
        "ppl": float(result["ppl"]),
        "source": "HQQ",
        "storage": storage,
        "model_bits": result.get("actual_model_bits_per_param"),
        "accounting_status": "recomputed_from_storage_breakdown" if storage else "declared_external",
    }


def _add_payload(data: dict[str, list[dict[str, Any]]], payload: dict[str, Any]) -> None:
    model = payload["model"]
    result = next((row for row in payload.get("results", [])
                   if float(row.get("k_frac") or 0.0) == 0.0), None)
    if result is None:
        return
    row = _baseline_row(payload, result)
    rows = data.setdefault(model, [])
    if not any(item.get("method") == row["method"] for item in rows):
        rows.append(row)
    fp16 = payload.get("baseline_fp16_ppl")
    if fp16 is not None and not any(item.get("method") == "FP16" for item in rows):
        rows.append({"method": "FP16", "bits": 16.0, "nominal_bits": 16.0,
                     "ppl": float(fp16), "source": "FP16"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--uniform-root", type=Path, required=True)
    parser.add_argument("--anchor-root", type=Path,
                        help="additional sweep root used to supply a shared HQQ-4 k=0 anchor")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.index.read_text(encoding="utf-8"))
    files = list(args.uniform_root.glob("**/channel_pareto.json"))
    if args.anchor_root:
        files.extend(args.anchor_root.glob("**/channel_pareto.json"))
    for path in sorted(set(files)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("base_quantizer", "hqq") != "hqq":
            continue
        # Anchor roots can contain many HQQ-4 selectors; the first valid k=0
        # supplies the one shared uniform row and later candidates are ignored.
        _add_payload(data, payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
