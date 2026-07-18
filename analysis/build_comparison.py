#!/usr/bin/env python3
"""Assemble the make-or-break comparison table: SEQ vs baselines at ACTUAL bits.

Pure-stdlib. Merges:
  - SEQ / random / act_scale points from channel_sweep JSONs (channel_pareto.json),
  - external baselines (AWQ, GPTQ, uniform HQQ, FP16) from a small JSON you fill
    with numbers measured on the SAME model + evaluator (e.g. via LightCompress),
into one per-model table sorted by actual bits, with the Pareto frontier marked
and the decisive verdict: is SEQ on the frontier / does it beat the baselines at
matched bits?

Bit accounting: SEQ's effective_bits already includes the FP16 residual columns
(16·k). We add a small index overhead for the protected-channel table. Base
group scales/zero-points are common to every method (SEQ, AWQ, GPTQ, uniform
HQQ), so they are excluded from the compared axis. Provide baseline bits on the
same convention (weight+residual bits), or pass measured checkpoint bits.

Baselines file (baselines.json)::

  {"meta-llama/Llama-3.2-1B": [
     {"method": "FP16",          "bits": 16.0, "ppl": 9.757},
     {"method": "HQQ-4 uniform", "bits": 4.0,  "ppl": 11.19},
     {"method": "AWQ-4 g128",    "bits": 4.0,  "ppl": 10.1},
     {"method": "GPTQ-4 g128",   "bits": 4.0,  "ppl": 10.0}
  ]}

Usage::

  python analysis/build_comparison.py \
      --sweeps runs/seq9_tiered runs/channel5 runs/gptq_check2 \
      --baselines baselines.json --signals act_max,random,act_scale \
      --out docs/COMPARISON.md
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple


def pareto_frontier(points: List[Tuple[float, float]]) -> List[int]:
    """Indices of points on the lower-left Pareto frontier (min bits, min ppl)."""
    keep = []
    for i, (b, p) in enumerate(points):
        dominated = any(
            j != i and q[0] <= b and q[1] <= p and (q[0] < b or q[1] < p)
            for j, q in enumerate(points)
        )
        if not dominated:
            keep.append(i)
    return keep


def _cfg_label(r: Dict[str, Any]) -> str:
    if r.get("tiers"):
        return f"[{r['tiers']}]"
    return f"k={r.get('k_frac')}"


def load_sweep_points(
    sweep_dirs: List[str],
    signals: List[str],
    index_overhead: float,
) -> Dict[str, List[Dict[str, Any]]]:
    """Extract SEQ/control points per model from channel_sweep JSONs."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    files: List[str] = []
    for d in sweep_dirs:
        files += glob.glob(os.path.join(d, "**", "channel_pareto.json"), recursive=True)
        if os.path.isfile(os.path.join(d, "channel_pareto.json")):
            files.append(os.path.join(d, "channel_pareto.json"))
    for f in sorted(set(files)):
        try:
            p = json.load(open(f))
        except Exception:  # noqa: BLE001
            continue
        model = p.get("model", f)
        base = p.get("base_quantizer", "hqq")
        bb = p.get("base_bits")
        for r in p.get("results", []):
            if r.get("signal") not in signals:
                continue
            if r.get("ppl") is None or r["ppl"] != r["ppl"]:
                continue
            k = r.get("k_frac") or 0.0
            eff = r.get("effective_bits", bb)
            # Prefer authoritative byte accounting from new runs. Historical
            # rows retain the documented approximation for backward compatibility.
            actual = r.get("actual_effective_bits")
            if actual is None:
                actual = eff + (index_overhead if (k and k > 0) or r.get("tiers") else 0.0)
            name = f"SEQ:{r['signal']}({base}-{bb}b {_cfg_label(r)})"
            out.setdefault(model, []).append(
                {"method": name, "bits": round(actual, 3), "nominal_bits": round(eff, 3),
                 "ppl": round(r["ppl"], 4), "source": "sweep"}
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", nargs="*", default=["runs"])
    ap.add_argument("--baselines", default="", help="JSON of external method numbers per model")
    ap.add_argument("--signals", default="act_max,random,act_scale")
    ap.add_argument("--index_overhead", type=float, default=0.1,
                    help="extra bits/param for the protected-channel index table")
    ap.add_argument("--out", default="docs/COMPARISON.md")
    ap.add_argument("--csv", default="results/final_comparison.csv")
    ap.add_argument("--json", default="results/final_comparison.json")
    args = ap.parse_args()

    signals = [s.strip() for s in args.signals.split(",") if s.strip()]
    per_model = load_sweep_points(args.sweeps, signals, args.index_overhead)

    if args.baselines and os.path.isfile(args.baselines):
        base = json.load(open(args.baselines))
        for model, rows in base.items():
            for r in rows:
                per_model.setdefault(model, []).append(
                    {"method": r["method"], "bits": round(float(r["bits"]), 3),
                     "nominal_bits": round(float(r.get("nominal_bits", r["bits"])), 3),
                     "ppl": round(float(r["ppl"]), 4), "source": "baseline"}
                )

    if not per_model:
        print("no points found — check --sweeps / --baselines", file=sys.stderr)
        return 1

    all_rows: List[Dict[str, Any]] = []
    L: List[str] = ["# SEQ vs baselines — actual-bits comparison", ""]
    L.append("Points from SEQ sweeps + external baselines, sorted by actual bits. "
             "★ = on the Pareto frontier (no method has both fewer bits and lower PPL). "
             "SEQ bits include the FP16 residual + index table; base group scales are common "
             "to all methods and excluded from this axis.")
    L.append("")
    for model, rows in per_model.items():
        fp16 = next((r["ppl"] for r in rows if "fp16" in r["method"].lower()), None)
        rows = sorted(rows, key=lambda r: (r["bits"], r["ppl"]))
        all_rows.extend({"model": model, **r} for r in rows)
        pts = [(r["bits"], r["ppl"]) for r in rows]
        front = set(pareto_frontier(pts))
        L.append(f"## {model}" + (f"  (FP16 PPL {fp16})" if fp16 else ""))
        L.append("")
        L.append("| method | actual bits | nominal bits | PPL | Δ vs FP16 | frontier |")
        L.append("|---|---|---|---|---|---|")
        for i, r in enumerate(rows):
            d = f"{r['ppl']-fp16:+.3f}" if fp16 else "—"
            star = "★" if i in front else ""
            L.append(f"| {r['method']} | {r['bits']:.2f} | {r['nominal_bits']:.2f} | "
                     f"{r['ppl']:.3f} | {d} | {star} |")
        L.append("")
        # verdict: is any SEQ point on the frontier, and does SEQ beat baselines near its bits?
        seq_front = [rows[i] for i in front if rows[i]["source"] == "sweep" and rows[i]["method"].startswith("SEQ:")]
        base_pts = [r for r in rows if r["source"] == "baseline" and (fp16 is None or r["ppl"] != fp16)]
        verdict = []
        if seq_front:
            verdict.append(f"SEQ is on the Pareto frontier ({len(seq_front)} point(s)).")
        else:
            verdict.append("**No SEQ point is on the frontier** — a baseline dominates it.")
        # closest baseline within ±0.3 bits of the best SEQ point
        best_seq = min((r for r in rows if r["method"].startswith("SEQ:")), key=lambda r: r["ppl"], default=None)
        if best_seq and base_pts:
            near = [r for r in base_pts if abs(r["bits"] - best_seq["bits"]) <= 0.3]
            if near:
                bestnear = min(near, key=lambda r: r["ppl"])
                cmp = "beats" if best_seq["ppl"] < bestnear["ppl"] else "loses to"
                verdict.append(f"At ~{best_seq['bits']:.1f} bits, best SEQ ({best_seq['ppl']:.3f}) "
                               f"**{cmp}** {bestnear['method']} ({bestnear['ppl']:.3f}).")
        L.append("> " + " ".join(verdict))
        L.append("")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write("\n".join(L).rstrip() + "\n")
    for path in (args.csv, args.json):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = ["model", "method", "bits", "nominal_bits", "ppl", "source"]
    with open(args.csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerows({k: row.get(k) for k in fields} for row in all_rows)
    with open(args.json, "w", encoding="utf-8") as handle:
        json.dump(all_rows, handle, indent=2); handle.write("\n")
    print("wrote", args.out)
    for model, rows in per_model.items():
        print(f"  {model}: {len(rows)} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
