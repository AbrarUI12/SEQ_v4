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


# The weight-only comparison axis (comparable to GPTQ-4 = 4.0) counts ONLY the
# quantized linear weights plus their inline overhead. Embeddings/lm_head/norms/
# bias are FP16 in every method and excluded so the axis is not dominated by
# them (a pure 4-bit base would otherwise measure ~7 bits on a 1B model).
_WEIGHT_BYTE_KEYS = (
    "dense_quantized_weight_bytes", "quantization_scale_bytes", "zero_point_bytes",
    "group_metadata_bytes", "fp16_residual_bytes", "int8_tier_bytes", "channel_index_bytes",
)


def _weight_bits_from_storage(storage: Any, base_bits: Any) -> Optional[float]:
    """Weight-only bits/param recomputed from a saved ``storage`` breakdown.

    Robust to the earlier bug where rows stored the full-model average (with FP16
    embeddings) under ``actual_effective_bits``: this ignores that field and
    rebuilds the weight-only number from the byte breakdown, so regenerating the
    table needs only a CPU re-run of this script — no GPU re-sweep.
    """
    if not isinstance(storage, dict):
        return None
    v = storage.get("actual_weight_bits_per_param")
    if isinstance(v, (int, float)) and v > 0:
        return float(v)
    qbytes = storage.get("dense_quantized_weight_bytes")
    if not qbytes or not base_bits:
        return None
    qparams = float(qbytes) * 8 / float(base_bits)  # count of quantized linear weights
    wbytes = sum(float(storage.get(k, 0) or 0) for k in _WEIGHT_BYTE_KEYS)
    return (wbytes * 8 / qparams) if qparams else None


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
            # ONE consistent weight-only axis for every sweep row: recompute from
            # the saved byte breakdown when present (authoritative), else fall back
            # to nominal + index overhead for legacy rows. Never use a stored
            # full-model average as the axis.
            storage = r.get("storage") if isinstance(r.get("storage"), dict) else None
            actual = _weight_bits_from_storage(storage, bb)
            if actual is None:
                # Do not silently put a nominal/full-model number on the
                # weight-only axis.  Legacy rows without a storage breakdown
                # are preserved on disk but excluded from the corrected table.
                print(f"WARNING: excluding sweep row with missing storage breakdown: {f} "
                      f"signal={r.get('signal')} k={r.get('k_frac')}", file=sys.stderr)
                continue
            storage = storage or {}
            model_bits = storage.get("actual_model_bits_per_parameter")
            name = f"SEQ:{r['signal']}({base}-{bb}b {_cfg_label(r)})"
            out.setdefault(model, []).append(
                {"method": name, "bits": round(actual, 3), "nominal_bits": round(eff, 3),
                 "model_bits": round(float(model_bits), 3) if isinstance(model_bits, (int, float)) else None,
                 "ppl": round(r["ppl"], 4), "source": "sweep",
                 "accounting_status": "recomputed_from_storage_breakdown"}
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
                if not isinstance(r.get("storage"), dict) and r.get("method") != "FP16":
                    print(f"WARNING: baseline has no storage breakdown; using declared bits for "
                          f"{model} {r.get('method')}", file=sys.stderr)
                per_model.setdefault(model, []).append(
                    {"method": r["method"], "bits": round(float(r["bits"]), 3),
                     "nominal_bits": round(float(r.get("nominal_bits", r["bits"])), 3),
                     "model_bits": (round(float(r["model_bits"]), 3) if r.get("model_bits") is not None else None),
                     "ppl": round(float(r["ppl"]), 4), "source": "baseline",
                     "accounting_status": r.get("accounting_status", "declared_external")}
                )

    if not per_model:
        print("no points found — check --sweeps / --baselines", file=sys.stderr)
        return 1

    all_rows: List[Dict[str, Any]] = []
    L: List[str] = ["# SEQ vs baselines — actual-bits comparison", ""]
    L.append("Points from SEQ sweeps + external baselines, sorted by **weight-only bits/param** — "
             "quantized linear weights plus their inline overhead (group scales/zeros, FP16/INT8 "
             "protection residual, channel index), divided by the quantized-linear parameter count. "
             "Embeddings, lm_head, norms and biases are FP16 in every method, common to the axis, "
             "and excluded — so this axis is directly comparable to GPTQ-4 = 4.0. "
             "★ = on the Pareto frontier (no method has both fewer bits and lower PPL).")
    L.append("")
    for model, rows in per_model.items():
        fp16 = next((r["ppl"] for r in rows if "fp16" in r["method"].lower()), None)
        rows = sorted(rows, key=lambda r: (r["bits"], r["ppl"]))
        all_rows.extend({"model": model, **r} for r in rows)
        pts = [(r["bits"], r["ppl"]) for r in rows]
        front = set(pareto_frontier(pts))
        L.append(f"## {model}" + (f"  (FP16 PPL {fp16})" if fp16 else ""))
        L.append("")
        L.append("Axis = **weight-only bits/param** (quantized linear weights + inline overhead; "
                 "FP16 embeddings/lm_head/norms excluded, common to all methods — so it is comparable "
                 "to GPTQ-4 = 4.0). *full-model bits* is the deployment average including FP16 "
                 "embeddings, shown for reference only, not the frontier axis.")
        L.append("")
        L.append("| method | weight bits | nominal bits | full-model bits | PPL | Δ vs FP16 | frontier |")
        L.append("|---|---|---|---|---|---|---|")
        for i, r in enumerate(rows):
            d = f"{r['ppl']-fp16:+.3f}" if fp16 else "—"
            star = "★" if i in front else ""
            mb = f"{r['model_bits']:.2f}" if r.get("model_bits") is not None else "—"
            L.append(f"| {r['method']} | {r['bits']:.2f} | {r['nominal_bits']:.2f} | {mb} | "
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
    fields = ["model", "method", "bits", "nominal_bits", "model_bits", "ppl", "source", "accounting_status"]
    with open(args.csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        writer.writerows({k: row.get(k) for k in fields} for row in all_rows)
    with open(args.json, "w", encoding="utf-8") as handle:
        json.dump(all_rows, handle, indent=2); handle.write("\n")
    print("wrote", args.out)
    for model, rows in per_model.items():
        print(f"  {model}: {len(rows)} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
