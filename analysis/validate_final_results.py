#!/usr/bin/env python3
"""Recursively validate SEQ result JSONs and write a human-readable report."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path


def _finite(x):
    try: return math.isfinite(float(x))
    except (TypeError, ValueError): return False


def validate(root: Path):
    checks = []; configs = set(); files = list(root.glob("**/*.json")) if root.exists() else []
    for path in files:
        try: payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            checks.append({"status":"FAIL","path":str(path),"message":f"invalid JSON: {exc}"}); continue
        if not isinstance(payload, dict): continue
        if "results" in payload and isinstance(payload["results"], list):
            for row in payload["results"]:
                cfg = row.get("configuration_id") or f"{payload.get('model')}|{row.get('signal')}|{row.get('k_frac')}|{row.get('tiers')}"
                status = "PASS"; messages = []
                if cfg in configs: status = "FAIL"; messages.append("duplicate configuration")
                configs.add(cfg)
                if not _finite(row.get("ppl")): status = "FAIL"; messages.append("non-finite PPL")
                bits = row.get("actual_effective_bits", row.get("effective_bits"))
                if bits is None: status = "FAIL"; messages.append("missing actual bits")
                elif not _finite(bits) or not 0 < float(bits) <= 16: status = "FAIL"; messages.append("invalid bits")
                if row.get("verification_errors") or row.get("errors"): messages.append("verification/errors present"); status = "FAIL"
                checks.append({"status":status,"path":str(path),"configuration_id":cfg,"message":"; ".join(messages) or "valid"})
        elif any(k in payload for k in ("ppl", "fp16_ppl", "gptq_ppl")):
            if not _finite(payload.get("ppl", payload.get("fp16_ppl", payload.get("gptq_ppl")))):
                checks.append({"status":"FAIL","path":str(path),"message":"non-finite PPL"})
    fatal = any(c["status"] == "FAIL" for c in checks)
    return {"root":str(root),"files_checked":len(files),"configurations":len(configs),"status":"FAIL" if fatal else ("MISSING" if not checks else "PASS"),"checks":checks}


def _markdown(report):
    lines = ["# Final results validation", "", f"Status: **{report['status']}**", "", "| status | path | configuration | message |", "|---|---|---|---|"]
    for c in report["checks"]: lines.append(f"| {c['status']} | `{c['path']}` | `{c.get('configuration_id','')}` | {c['message']} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root", type=Path, default=Path("runs/final"), nargs="?"); ap.add_argument("--json", type=Path, default=Path("results/final_validation.json")); ap.add_argument("--markdown", type=Path, default=Path("docs/VALIDATION_REPORT.md")); args = ap.parse_args()
    report = validate(args.root); args.json.parent.mkdir(parents=True, exist_ok=True); args.json.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8"); args.markdown.parent.mkdir(parents=True, exist_ok=True); args.markdown.write_text(_markdown(report), encoding="utf-8"); print(json.dumps(report, indent=2)); return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__": raise SystemExit(main())
