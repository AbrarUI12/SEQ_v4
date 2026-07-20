#!/usr/bin/env python3
"""Assemble the F5 downstream-accuracy table from run_downstream_eval.sh outputs.

Reads ``<root>/<Model>/<point>/{seq_meta.json, lm_eval/**}`` produced by
``scripts/run_downstream_eval.sh`` and emits:

  * a Markdown table (accuracy per task + macro-average, per model),
  * a CSV and a JSON with the same numbers, and
  * paired-bootstrap 95% CIs for the contrasts declared in the config
    (e.g. residual_max-on-GPTQ vs GPTQ-4, greedy-on-GPTQ vs GPTQ-4).

Pure standard library on purpose (no numpy/torch): it runs on the CPU/analysis
box and is unit-testable anywhere, matching seq_core/stats_utils.py.

Paired bootstrap: when lm-eval was run with ``--log_samples`` the per-example
0/1 correctness is available, so two systems are compared on the *same* examples
(a paired design, tighter and correct). If sample logs are missing we fall back
to an unpaired normal-approximation CI from the harness's own stderr and label it.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

# preferred primary metric per task (acc_norm where the task reports it)
_ACC_NORM_TASKS = {"hellaswag", "arc_challenge", "arc_easy", "piqa"}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _find_results_json(lm_dir: str) -> Optional[str]:
    """Newest lm-eval results file (a JSON with a top-level 'results' dict)."""
    cands = sorted(glob.glob(os.path.join(lm_dir, "**", "*.json"), recursive=True),
                   key=os.path.getmtime, reverse=True)
    for path in cands:
        if os.path.basename(path).startswith("samples_"):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                obj = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(obj, dict) and isinstance(obj.get("results"), dict):
            return path
    return None


def _sample_files(lm_dir: str) -> Dict[str, str]:
    """Map task -> newest samples_<task>_*.jsonl path (per-example logs)."""
    out: Dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(lm_dir, "**", "samples_*.jsonl"), recursive=True),
                       key=os.path.getmtime):
        base = os.path.basename(path)
        task = base[len("samples_"):].rsplit("_", 1)[0]  # strip 'samples_' and trailing _<ts>
        out[task] = path  # newest wins (sorted ascending by mtime)
    return out


def _primary_metric_base(task: str, metrics: Dict[str, Any]) -> Optional[str]:
    order = (("acc_norm", "acc") if task in _ACC_NORM_TASKS else ("acc", "acc_norm"))
    for base in order:
        if f"{base},none" in metrics:
            return base
    for key in metrics:  # last resort: any ',none' metric that isn't a stderr
        if key.endswith(",none") and "stderr" not in key:
            return key[: -len(",none")]
    return None


def load_points(root: str) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    for model_dir in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(model_dir):
            continue
        for point_dir in sorted(glob.glob(os.path.join(model_dir, "*"))):
            meta_path = os.path.join(point_dir, "seq_meta.json")
            if not os.path.isfile(meta_path):
                continue
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            lm_dir = os.path.join(point_dir, "lm_eval")
            res_path = _find_results_json(lm_dir) if os.path.isdir(lm_dir) else None
            results: Dict[str, Dict[str, Any]] = {}
            if res_path:
                with open(res_path, encoding="utf-8") as fh:
                    raw = json.load(fh)["results"]
                for task, metrics in raw.items():
                    base = _primary_metric_base(task, metrics)
                    if base is None:
                        continue
                    results[task] = {
                        "metric": base,
                        "value": _as_float(metrics.get(f"{base},none")),
                        "stderr": _as_float(metrics.get(f"{base}_stderr,none")),
                    }
            points.append({
                "model": meta.get("model"),
                "point": meta.get("point") or os.path.basename(point_dir),
                "meta": meta,
                "results": results,
                "samples": _sample_files(lm_dir) if os.path.isdir(lm_dir) else {},
            })
    return points


def _as_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _per_example_correct(samples_path: str, metric_base: str) -> Dict[Any, float]:
    """doc_id -> 0/1 correctness for the given metric from an lm-eval samples log."""
    out: Dict[Any, float] = {}
    with open(samples_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc_id = row.get("doc_id")
            val = row.get(metric_base)
            if val is None and isinstance(row.get("metrics"), dict):
                val = row["metrics"].get(metric_base)
            f = _as_float(val)
            if doc_id is not None and f is not None:
                out[doc_id] = f
    return out


# --------------------------------------------------------------------------- #
# Paired bootstrap
# --------------------------------------------------------------------------- #
def paired_bootstrap_diff(
    a: Sequence[float], b: Sequence[float], n_boot: int = 2000, seed: int = 1234,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """95% CI for mean(a) - mean(b) over paired samples via case resampling."""
    n = len(a)
    if n == 0 or n != len(b):
        return {"n": n, "diff": None, "lo": None, "hi": None}
    point = _mean(a) - _mean(b)
    rng = random.Random(seed)
    diffs: List[float] = []
    for _ in range(n_boot):
        sa = sb = 0.0
        for _ in range(n):
            i = rng.randrange(n)
            sa += a[i]; sb += b[i]
        diffs.append((sa - sb) / n)
    diffs.sort()
    lo = diffs[max(0, int((alpha / 2) * n_boot))]
    hi = diffs[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return {"n": n, "diff": point, "lo": lo, "hi": hi}


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def contrast_cis(points_by_key: Dict[Tuple[str, str], Dict[str, Any]],
                 model: str, a_point: str, b_point: str,
                 seed: int = 1234) -> Dict[str, Any]:
    """Per-task and macro-average paired-bootstrap CIs for point a vs b on a model."""
    a = points_by_key.get((model, a_point))
    b = points_by_key.get((model, b_point))
    if not a or not b:
        return {"available": False, "reason": "missing point"}
    tasks = sorted(set(a["results"]) & set(b["results"]))
    per_task: Dict[str, Any] = {}
    macro_terms: List[List[float]] = []  # aligned per-task [ca_i - cb_i] not used; store paired arrays
    paired_available = True
    for task in tasks:
        base = a["results"][task]["metric"]
        sa, sb = a["samples"].get(task), b["samples"].get(task)
        if sa and sb:
            ca_map = _per_example_correct(sa, base)
            cb_map = _per_example_correct(sb, b["results"][task]["metric"])
            shared = sorted(set(ca_map) & set(cb_map), key=str)
            ca = [ca_map[d] for d in shared]
            cb = [cb_map[d] for d in shared]
            ci = paired_bootstrap_diff(ca, cb, seed=seed)
            ci["paired"] = True
            per_task[task] = ci
            macro_terms.append([x - y for x, y in zip(ca, cb)])
        else:  # unpaired fallback from harness stderr
            paired_available = False
            va, vb = a["results"][task]["value"], b["results"][task]["value"]
            se_a = a["results"][task]["stderr"] or 0.0
            se_b = b["results"][task]["stderr"] or 0.0
            diff = (va - vb) if (va is not None and vb is not None) else None
            se = math.sqrt(se_a ** 2 + se_b ** 2)
            per_task[task] = {
                "paired": False, "diff": diff,
                "lo": (diff - 1.96 * se) if diff is not None else None,
                "hi": (diff + 1.96 * se) if diff is not None else None,
            }
    macro = None
    if macro_terms and all(len(t) for t in macro_terms):
        # bootstrap the macro-average of per-task diffs, resampling within each task
        rng = random.Random(seed + 1)
        boots: List[float] = []
        for _ in range(2000):
            acc = 0.0
            for terms in macro_terms:
                m = len(terms)
                acc += sum(terms[rng.randrange(m)] for _ in range(m)) / m
            boots.append(acc / len(macro_terms))
        boots.sort()
        macro = {
            "diff": sum(_mean(t) for t in macro_terms) / len(macro_terms),
            "lo": boots[int(0.025 * len(boots))], "hi": boots[int(0.975 * len(boots))],
            "paired": True,
        }
    return {"available": True, "paired": paired_available,
            "per_task": per_task, "macro_avg": macro}


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _fmt(v: Optional[float], pct: bool = True) -> str:
    if v is None:
        return "—"
    return f"{100 * v:.2f}" if pct else f"{v:.4f}"


def _macro_avg(results: Dict[str, Dict[str, Any]]) -> Optional[float]:
    vals = [r["value"] for r in results.values() if r["value"] is not None]
    return sum(vals) / len(vals) if vals else None


def render_markdown(points: List[Dict[str, Any]], config: Dict[str, Any]) -> str:
    order = list(config.get("point_defs", {}).keys()) if config else []
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    all_tasks: List[str] = []
    for p in points:
        by_model.setdefault(p["model"], []).append(p)
        for t in p["results"]:
            if t not in all_tasks:
                all_tasks.append(t)
    tasks = sorted(all_tasks)

    lines = ["# F5 — Downstream task accuracy at the paper operating points",
             "",
             "Zero-shot accuracy (acc_norm where the task reports it, else acc), WikiText-2-"
             "matched checkpoints. Higher is better. `avg` is the macro-average across tasks.",
             ""]
    pkey = {(p["model"], p["point"]): p for p in points}
    for model in sorted(by_model):
        lines += [f"## {model}", "",
                  "| point | bits | " + " | ".join(tasks) + " | avg | note |",
                  "|" + "---|" * (len(tasks) + 4)]
        pts = sorted(by_model[model], key=lambda p: order.index(p["point"]) if p["point"] in order else 99)
        for p in pts:
            bits = p["meta"].get("nominal_bits")
            cells = [_fmt(p["results"].get(t, {}).get("value")) for t in tasks]
            row = [p["point"], (f"{bits:g}" if isinstance(bits, (int, float)) else "—"),
                   *cells, _fmt(_macro_avg(p["results"])), str(p["meta"].get("note") or "")]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        # paired contrasts
        contrasts = (config or {}).get("paired_contrasts", [])
        if contrasts:
            lines += ["### Paired contrasts (Δ accuracy points, 95% CI)", ""]
            for c in contrasts:
                ci = contrast_cis(pkey, model, c["a"], c["b"])
                if not ci.get("available"):
                    continue
                macro = ci.get("macro_avg")
                tag = "paired bootstrap" if ci.get("paired") else "UNPAIRED approx (no sample logs)"
                if macro and macro["diff"] is not None:
                    lines.append(
                        f"- **{c['a']} − {c['b']}** ({c.get('claim','')}): "
                        f"macro Δ = {100*macro['diff']:+.2f} "
                        f"[{100*macro['lo']:+.2f}, {100*macro['hi']:+.2f}] pts — {tag}.")
                else:
                    lines.append(f"- **{c['a']} − {c['b']}** ({c.get('claim','')}): {tag}; see JSON.")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_csv(points: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "point", "nominal_bits", "task", "metric", "value", "stderr"])
        for p in points:
            for task, r in sorted(p["results"].items()):
                w.writerow([p["model"], p["point"], p["meta"].get("nominal_bits"),
                            task, r["metric"], r["value"], r["stderr"]])


def build_json(points: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    pkey = {(p["model"], p["point"]): p for p in points}
    models = sorted({p["model"] for p in points})
    out: Dict[str, Any] = {"points": [], "contrasts": {}}
    for p in points:
        out["points"].append({
            "model": p["model"], "point": p["point"],
            "nominal_bits": p["meta"].get("nominal_bits"),
            "results": p["results"], "macro_avg": _macro_avg(p["results"]),
        })
    for model in models:
        out["contrasts"][model] = {}
        for c in (config or {}).get("paired_contrasts", []):
            out["contrasts"][model][c["name"]] = contrast_cis(pkey, model, c["a"], c["b"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="runs/final/downstream",
                    help="root written by scripts/run_downstream_eval.sh")
    ap.add_argument("--config", default="configs/downstream_operating_points.json")
    ap.add_argument("--out", default="docs/DOWNSTREAM.md")
    ap.add_argument("--csv", default="results/downstream.csv")
    ap.add_argument("--json", default="results/downstream.json")
    args = ap.parse_args()

    config: Dict[str, Any] = {}
    if os.path.isfile(args.config):
        with open(args.config, encoding="utf-8") as fh:
            config = json.load(fh)

    points = load_points(args.root)
    if not points:
        print(f"no operating points with seq_meta.json found under {args.root}")
        return 1

    for path in (args.out, args.csv, args.json):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(points, config))
    write_csv(points, args.csv)
    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(build_json(points, config), fh, indent=2)

    n_eval = sum(1 for p in points if p["results"])
    print(f"wrote {args.out}, {args.csv}, {args.json} "
          f"({len(points)} points, {n_eval} with lm-eval results)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
