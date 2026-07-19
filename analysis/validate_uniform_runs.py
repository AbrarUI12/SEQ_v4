#!/usr/bin/env python3
"""Validate uniform HQQ run directories without loading model weights."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path


def _finite(v):
    try: return math.isfinite(float(v))
    except (TypeError, ValueError): return False


def validate(root: Path):
    rows = []; errors = []
    for p in sorted(root.glob("**/*.json")):
        if p.name in {"model_index.json", "validation.json"}: continue
        try: data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc: errors.append(f"{p}: invalid JSON: {exc}"); continue
        if not isinstance(data, dict): continue
        candidates = data.get("results") if isinstance(data.get("results"), list) else [data]
        for i, row in enumerate(candidates):
            if not isinstance(row, dict): continue
            if not any(k in row for k in ("ppl", "uniform_hqq_ppl")): continue
            ppl = row.get("uniform_hqq_ppl", row.get("ppl"))
            local_errors = []
            if not _finite(ppl): local_errors.append("non-finite PPL")
            if row.get("num_quantized_modules", row.get("quantized_modules", data.get("num_layers", 1))) == 0:
                local_errors.append("zero quantized modules")
            bits = row.get("actual_effective_bits", row.get("effective_bits"))
            if bits is not None and (not _finite(bits) or not 0 < float(bits) <= 16): local_errors.append("invalid effective bits")
            errors.extend(f"{p}: {e}" for e in local_errors)
            rows.append({"path": str(p), "row": i, "ppl": ppl, "effective_bits": bits,
                         "status": "FAIL" if local_errors else "PASS"})
    report = {"root": str(root), "files_checked": len(rows), "errors": errors, "status": "FAIL" if errors else "PASS", "rows": rows}
    return report


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root", type=Path, default=Path("runs/hqq_uniform"), nargs="?"); ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(); report = validate(args.root)
    text = json.dumps(report, indent=2) + "\n"
    if args.output: args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(text, encoding="utf-8")
    print(text, end=""); return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__": raise SystemExit(main())
