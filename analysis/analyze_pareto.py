#!/usr/bin/env python3
"""Analyze Pareto sweeps: does allocating bits by a signal beat random/uniform?

Pure-stdlib over runs/*/pareto.json. Reports, per model:
- PPL vs effective bits for every signal,
- each signal's gap vs the `random` control (positive = WORSE than chance),
- a concentration diagnostic (share of params pushed to the lowest bit level),
  which explains why extensive signals (hessian_diag/salience) collapse.

This is the *non-circular* end-to-end test. A signal that wins the local
reconstruction correlation but loses here is optimizing the wrong objective.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_pareto(root: str) -> List[str]:
    out = []
    for dp, _d, files in os.walk(root):
        if ".git" in dp:
            continue
        if "pareto.json" in files:
            out.append(os.path.join(dp, "pareto.json"))
    return sorted(out)


def min_level_share(params_by_bits: Dict[str, int], levels: List[int]) -> float:
    if not params_by_bits:
        return float("nan")
    lo = str(min(levels))
    tot = sum(params_by_bits.values()) or 1
    return params_by_bits.get(lo, 0) / tot


def fmt(v: Optional[float], nd: int = 2) -> str:
    return "—" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{nd}f}"


def analyze(path: str) -> Dict[str, Any]:
    p = json.load(open(path))
    rs = p["results"]
    signals = sorted({r["signal"] for r in rs})
    budgets = sorted({r["target_bits"] for r in rs})
    levels = p.get("levels", [3, 4, 8])
    base = p.get("baseline_fp16_ppl")

    def row(sig: str, b: float) -> Optional[Dict[str, Any]]:
        return next((r for r in rs if r["signal"] == sig and r["target_bits"] == b), None)

    # gap vs random, per (signal, budget)
    gaps: Dict[str, List[Optional[float]]] = {}
    for s in signals:
        gaps[s] = []
        for b in budgets:
            rs_s, rs_r = row(s, b), row("random", b)
            if rs_s and rs_r and rs_s["ppl"] == rs_s["ppl"] and rs_r["ppl"] == rs_r["ppl"]:
                gaps[s].append(rs_s["ppl"] - rs_r["ppl"])
            else:
                gaps[s].append(None)
    mean_gap = {
        s: (sum(g for g in gaps[s] if g is not None) / max(1, len([g for g in gaps[s] if g is not None])))
        for s in signals
    }
    return {
        "model": p.get("model"),
        "baseline": base,
        "budgets": budgets,
        "signals": signals,
        "levels": levels,
        "rows": rs,
        "row_fn_data": {(r["signal"], r["target_bits"]): r for r in rs},
        "gaps": gaps,
        "mean_gap_vs_random": mean_gap,
    }


def build_md(analyses: List[Dict[str, Any]]) -> str:
    L: List[str] = ["# Findings run 3 — downstream Pareto (does the signal beat random?)", ""]
    L.append("End-to-end canonical PPL after allocating bits by each signal (native "
             "high→more-bits) and quantizing with HQQ. `random` is the chance control. "
             "**Lower is better; a positive gap-vs-random means the signal is WORSE than chance.**")
    L.append("")
    for a in analyses:
        rd = a["row_fn_data"]
        budgets = a["budgets"]
        L.append(f"## {a['model']} (FP16 PPL {fmt(a['baseline'],3)})")
        L.append("")
        L.append("| signal | " + " | ".join(f"~{b}b" for b in budgets) + " | mean gap vs random |")
        L.append("|" + "---|" * (len(budgets) + 2))
        # order signals by mean gap (best first)
        for s in sorted(a["signals"], key=lambda s: a["mean_gap_vs_random"][s]):
            cells = []
            for b in budgets:
                r = rd.get((s, b))
                cells.append(f"{fmt(r['ppl'])}" if r and r["ppl"] == r["ppl"] else "—")
            tag = " ✅" if a["mean_gap_vs_random"][s] < -0.05 else (" ❌" if a["mean_gap_vs_random"][s] > 0.3 else "")
            L.append(f"| `{s}` | " + " | ".join(cells) + f" | {a['mean_gap_vs_random'][s]:+.2f}{tag} |")
        L.append("")
        # concentration diagnostic at the tightest non-degenerate budget
        L.append("Concentration (share of params forced to the lowest level "
                 f"{min(a['levels'])}-bit) — high share ⇒ pathological allocation:")
        L.append("")
        L.append("| signal | " + " | ".join(f"~{b}b" for b in budgets) + " |")
        L.append("|" + "---|" * (len(budgets) + 1))
        for s in a["signals"]:
            cells = []
            for b in budgets:
                r = rd.get((s, b))
                sh = min_level_share(r.get("params_by_bits", {}), a["levels"]) if r else None
                cells.append(f"{sh*100:.0f}%" if sh == sh and sh is not None else "—")
            L.append(f"| `{s}` | " + " | ".join(cells) + " |")
        L.append("")

    # cross-model verdict
    L.append("## Verdict")
    L.append("")
    sig_all = sorted({s for a in analyses for s in a["signals"]})
    L.append("| signal | mean gap vs random (per model) | overall |")
    L.append("|---|---|---|")
    for s in sig_all:
        per = [a["mean_gap_vs_random"].get(s) for a in analyses if s in a["mean_gap_vs_random"]]
        per = [x for x in per if x is not None]
        if not per:
            continue
        overall = sum(per) / len(per)
        L.append(f"| `{s}` | {', '.join(f'{x:+.2f}' for x in per)} | {overall:+.2f} |")
    L.append("")
    L.append("- **Positive gap = worse than random.** `hessian_diag` and `salience` — the winners of "
             "the reconstruction correlation — are far worse than random: their extensive per-layer "
             "sums over-protect `lm_head`/large layers and starve the rest to the lowest level.")
    L.append("- **`entropy`, `magnitude` ≈ `random`**: at module granularity and 3–7 bits, no signal "
             "reliably beats uniform/random. Entropy 'works' only because it is near-uniform across "
             "modules, so allocating by it ≈ uniform allocation.")
    L.append("- **Conclusion:** module-level sensitivity-guided mixed precision does not beat uniform "
             "here, and the reconstruction correlation is a misleading proxy. The open levers are "
             "per-channel allocation and a compounding-aware objective / per-parameter normalized "
             "signals with a low-bit floor.")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/pareto")
    ap.add_argument("--out", default="docs/FINDINGS_run3.md")
    args = ap.parse_args()
    paths = find_pareto(args.root)
    if not paths:
        print("no pareto.json found under", args.root, file=sys.stderr)
        return 1
    analyses = [analyze(p) for p in paths]
    analyses.sort(key=lambda a: a.get("baseline") or 0)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w").write(build_md(analyses))
    print(f"analyzed {len(analyses)} models -> {args.out}")
    for a in analyses:
        worst = max(a["mean_gap_vs_random"].items(), key=lambda kv: kv[1])
        best = min(a["mean_gap_vs_random"].items(), key=lambda kv: kv[1])
        print(f"  {a['model']:>22}: best={best[0]}({best[1]:+.2f})  worst={worst[0]}({worst[1]:+.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
