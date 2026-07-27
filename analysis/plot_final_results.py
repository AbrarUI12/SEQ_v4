#!/usr/bin/env python3
"""Generate publication figures for the SEQ findings paper.

One invocation writes every figure whose source data is present:

  Fig 1  dose-response      PPL vs protection fraction, HQQ|GPTQ panels, per selector
                            (random shown with a seed band); log-y. Source: sweep JSON.
  Fig 2  Pareto frontier    PPL vs weight-only bits/param scatter. Source: comparison
                            CSV (--input) or sweep channel_pareto.json (--root).
  Fig 3  downstream forest  paired-contrast macro-Δ accuracy ± 95% CI, both models,
                            zero line. Source: results/downstream.json.
  Fig 4  F3 forensic        per model, FP16 vs greedy runtime PPL vs materialized PPL
                            bars (log-y). Source: results/f3check_*/channel_pareto.json.

Missing sources are skipped with a note, so this runs before and after the GPU jobs land.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

# selector -> (display label, colour) for the dose-response panels
SELECTOR_STYLE = {
    "greedy":        ("greedy (OMP)",        "#d1495b"),
    "greedy_indep":  ("greedy-indep",        "#edae49"),
    "residual_max":  ("residual_max (safe)", "#00798c"),
    "random":        ("random (control)",    "#8d99ae"),
}
DOSE_SELECTORS = ["greedy", "greedy_indep", "residual_max", "random"]
# contrast -> short label for the forest plot (order top->bottom)
CONTRAST_LABEL = {
    "best_hqq_vs_hqq4":       "best@HQQ − HQQ-4  (F1)",
    "best_hqq_vs_random_hqq": "best@HQQ − random@HQQ  (F1, matched)",
    "resmax_gptq_vs_gptq4":   "resmax@GPTQ − GPTQ-4  (F4)",
    "greedy_gptq_vs_gptq4":   "greedy@GPTQ − GPTQ-4  (F3 pred.)",
}
MODEL_STYLE = {  # model -> (short label, colour) for multi-model figures
    "meta-llama/Llama-3.2-3B": ("Llama-3.2-3B", "#00798c"),
    "meta-llama/Llama-3.2-1B": ("Llama-3.2-1B", "#d1495b"),
    "meta-llama/Llama-2-7b-hf": ("Llama-2-7B", "#edae49"),
}


def _short(model: str) -> str:
    return model.split("/")[-1]


def _load_selector_series(sweeps_root: Path, base: str, model_short: str, selector: str):
    """Return {frac: [ppl, ...across seeds]} for one selector under one base."""
    series: dict[float, list[float]] = {}
    for jp in sorted(sweeps_root.glob(f"{base}/{model_short}/{selector}/seed-*/channel_pareto.json")):
        try:
            d = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        for r in d.get("results", []):
            k = r.get("k_frac")
            ppl = r.get("ppl")
            if k is None or ppl is None:
                continue
            try:
                k = float(k); ppl = float(ppl)
            except (TypeError, ValueError):
                continue
            if k <= 0:
                continue
            series.setdefault(k, []).append(ppl)
    return series


def fig_dose_response(plt, sweeps_root: Path, model: str, out: Path) -> bool:
    bases = [("hqq", "HQQ-4 base (data-free)"), ("gptq_llmc", "GPTQ-4 base (error-compensated)")]
    ms = _short(model)
    have_any = False
    # Unshared y: the GPTQ blow-up (~50) would otherwise flatten the HQQ panel and
    # hide the F1 ordering. Each panel autoscales (both log) to tell its own story.
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3), sharey=False)
    for ax, (base, title) in zip(axes, bases):
        for selector in DOSE_SELECTORS:
            series = _load_selector_series(sweeps_root, base, ms, selector)
            if not series:
                continue
            have_any = True
            fracs = sorted(series)
            means = [sum(series[f]) / len(series[f]) for f in fracs]
            label, colour = SELECTOR_STYLE[selector]
            ax.plot(fracs, means, marker="o", ms=4, color=colour, label=label)
            if selector == "random" and any(len(series[f]) > 1 for f in fracs):
                lo = [min(series[f]) for f in fracs]
                hi = [max(series[f]) for f in fracs]
                ax.fill_between(fracs, lo, hi, color=colour, alpha=0.20)
        ax.set_yscale("log")
        ax.set_xlabel("protected fraction of input channels")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, which="both")
    for ax in axes:  # y unshared, so label both
        ax.set_ylabel("WikiText-2 perplexity (log)")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Dose-response of channel protection — {ms}", fontsize=11)
    fig.tight_layout()
    if have_any:
        fig.savefig(out / f"fig1_dose_response_{ms}.pdf")
    plt.close(fig)
    return have_any


def fig_forest(plt, downstream_json: Path, out: Path) -> bool:
    try:
        d = json.loads(downstream_json.read_text(encoding="utf-8"))
    except Exception:
        return False
    contrasts = d.get("contrasts", {})
    if not contrasts:
        return False
    # keep only contrasts we have labels for, in the defined order
    order = [c for c in CONTRAST_LABEL if any(c in contrasts.get(m, {}) for m in contrasts)]
    if not order:
        return False
    models = [m for m in MODEL_STYLE if m in contrasts] or list(contrasts)
    fig, ax = plt.subplots(figsize=(7.6, 0.9 + 0.85 * len(order)))
    y_base = list(range(len(order)))[::-1]  # first contrast at top
    offsets = {m: (i - (len(models) - 1) / 2) * 0.18 for i, m in enumerate(models)}
    for m in models:
        label, colour = MODEL_STYLE.get(m, (_short(m), "#333333"))
        xs, ys, lo, hi = [], [], [], []
        for c, yb in zip(order, y_base):
            cell = contrasts.get(m, {}).get(c)
            if not cell or not cell.get("available"):
                continue
            # `macro_avg` can be present but null when a point was excluded (e.g. a
            # checkpoint that failed reload validation), so guard on the value.
            ma = cell.get("macro_avg")
            if not ma or ma.get("diff") is None:
                continue
            xs.append(ma["diff"] * 100.0); ys.append(yb + offsets[m])
            lo.append((ma["diff"] - ma["lo"]) * 100.0); hi.append((ma["hi"] - ma["diff"]) * 100.0)
        if xs:
            ax.errorbar(xs, ys, xerr=[lo, hi], fmt="o", ms=5, capsize=3,
                        color=colour, label=label, lw=1.4)
    ax.axvline(0.0, color="#000000", lw=1.0, ls="--", alpha=0.7)
    ax.set_yticks(y_base)
    ax.set_yticklabels([CONTRAST_LABEL[c] for c in order], fontsize=9)
    ax.set_xlabel("Δ macro accuracy (pts), paired bootstrap 95% CI")
    ax.set_title("Downstream paired contrasts (F1 / F4 / F3)", fontsize=11)
    ax.grid(alpha=0.25, axis="x")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out / "fig3_downstream_forest.pdf")
    plt.close(fig)
    return True


def fig_forensic(plt, f3_root: Path, out: Path) -> bool:
    rows = []
    for jp in sorted(f3_root.glob("f3check_*/channel_pareto.json")):
        try:
            d = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        res = d.get("results") or []
        if not res:
            continue
        r = res[0]
        fp16 = d.get("baseline_fp16_ppl")
        ppl = r.get("ppl"); mat = r.get("materialized_ppl")
        if fp16 is None or ppl is None:
            continue
        rows.append((d.get("model", jp.parent.name), float(fp16), float(ppl),
                     float(mat) if mat is not None else None))
    if not rows:
        return False
    # stable, meaningful order: by model style if known, else as found
    rank = {m: i for i, m in enumerate(MODEL_STYLE)}
    rows.sort(key=lambda x: rank.get(x[0], 99))
    labels = [MODEL_STYLE.get(m, (_short(m),))[0] for m, *_ in rows]
    import numpy as np  # matplotlib always ships numpy
    x = np.arange(len(rows)); w = 0.27
    fig, ax = plt.subplots(figsize=(1.9 + 1.5 * len(rows), 4.4))
    ax.bar(x - w, [r[1] for r in rows], w, label="FP16 baseline", color="#8d99ae")
    ax.bar(x,     [r[2] for r in rows], w, label="greedy@GPTQ runtime PPL", color="#d1495b")
    ax.bar(x + w, [(r[3] if r[3] is not None else 0.0) for r in rows], w,
           label="materialized (exported) PPL", color="#00798c")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("WikiText-2 perplexity (log)")
    ax.set_title("F3 forensic: catastrophe is model-dependent, export is faithful", fontsize=10.5)
    ax.grid(alpha=0.25, axis="y", which="both")
    ax.legend(fontsize=8)
    for xi, r in zip(x, rows):  # annotate runtime PPL
        ax.annotate(f"{r[2]:.1f}", (xi, r[2]), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig4_f3_forensic.pdf")
    plt.close(fig)
    return True


def fig_decoupling(plt, downstream_json: Path, out: Path) -> bool:
    """Headline figure: WikiText-2 perplexity (log x) vs downstream macro accuracy (y).

    Each operating point is one marker. The catastrophic-perplexity point sits far right yet at
    the same height as every other point — the decoupling, in one picture.
    """
    # (label, wikitext ppl, source) — perplexities from the committed sweeps/reload artifacts.
    ppl_3b = {
        "fp16": 7.817, "gptq4": 8.172, "hqq4": 8.328,
        "resmax_gptq": 8.100, "best_hqq": 7.994, "random_hqq": 8.247,
        "greedy_gptq": 51.76,          # disk-reload verified
    }
    try:
        d = json.loads(downstream_json.read_text(encoding="utf-8"))
    except Exception:
        return False
    acc = {}
    for p in d.get("points", []):
        if not str(p.get("model", "")).endswith("3B"):
            continue
        vals = [v["value"] for v in (p.get("results") or {}).values()
                if isinstance(v, dict) and "value" in v]
        if vals:
            acc[p.get("point")] = 100.0 * sum(vals) / len(vals)
    pts = [(k, ppl_3b[k], acc[k]) for k in ppl_3b if k in acc]
    if len(pts) < 3:
        return False
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for label, x, y in pts:
        crit = label == "greedy_gptq"
        ax.scatter([x], [y], s=150 if crit else 70,
                   marker="*" if crit else "o",
                   color="#d1495b" if crit else "#00798c", zorder=3)
        ax.annotate(f"{label}\n(ppl {x:.2f})" if crit else label,
                    (x, y), textcoords="offset points",
                    xytext=(-12 if crit else 10, 10 if crit else 6), fontsize=8.5,
                    ha="right" if crit else "left",
                    color="#d1495b" if crit else "#333333",
                    fontweight="bold" if crit else "normal")
    ys = [y for _, _, y in pts]
    ax.axhspan(min(ys), max(ys), color="#8d99ae", alpha=0.12, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("WikiText-2 perplexity (log scale)")
    ax.set_ylabel("downstream macro accuracy, 6 tasks (%)")
    ax.set_title("A 6.3x perplexity collapse costs no downstream accuracy\n"
                 "Llama-3.2-3B, matched weight-only storage", fontsize=10.5)
    ax.grid(alpha=0.25, which="both")
    ax.margins(y=0.25)
    fig.tight_layout()
    fig.savefig(out / "fig0_decoupling.pdf")
    plt.close(fig)
    return True


def fig_coupling(plt, out: Path) -> bool:
    """Perplexity by how much of the Hessian objective the selector uses (both models)."""
    groups = ["none\n(random,\nact_max,\nresidual_max)", "H diagonal\nonly\n(hessian_diag)",
              "full H,\none-shot\n(greedy_indep)", "full H,\niterative\n(greedy)"]
    series = {"Llama-3.2-3B": ([8.181, 8.153, 41.50, 51.68], 8.172),
              "Llama-3.2-1B": ([10.41, 10.441, 11.28, 63.95], 10.557)}
    import numpy as np
    x = np.arange(len(groups)); w = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for i, (name, (vals, base)) in enumerate(series.items()):
        colour = "#00798c" if "3B" in name else "#d1495b"
        ax.bar(x + (i - 0.5) * w, vals, w, label=f"{name} (base {base:.2f})", color=colour)
        ax.axhline(base, color=colour, ls=":", lw=1.1, alpha=0.8)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=8)
    ax.set_ylabel("WikiText-2 perplexity (log)")
    ax.set_title("Damage requires the selector's off-diagonal Hessian coupling\n"
                 "GPTQ-4 base, 2% protection, matched storage (dotted = unprotected base)",
                 fontsize=10.5)
    ax.grid(alpha=0.25, axis="y", which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig5_coupling.pdf")
    plt.close(fig)
    return True


def fig_pareto(plt, args, out: Path) -> bool:
    """Fig 2 — unchanged behaviour: PPL vs actual bits scatter per model."""
    points: dict[str, list[dict]] = {}
    if args.input and args.input.exists():
        with args.input.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    points.setdefault(row["model"], []).append({
                        "bits": float(row["bits"]), "ppl": float(row["ppl"]),
                        "method": str(row["method"]),
                        "ci_low": float(row["ppl_ci_low"]) if row.get("ppl_ci_low") else None,
                        "ci_high": float(row["ppl_ci_high"]) if row.get("ppl_ci_high") else None,
                    })
                except (TypeError, ValueError, KeyError):
                    pass
    else:
        for p in args.root.glob("**/channel_pareto.json"):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            model = d.get("model", p.parent.name)
            for r in d.get("results", []):
                try:
                    bits = r.get("actual_effective_bits", r.get("effective_bits"))
                    points.setdefault(model, []).append({
                        "bits": float(bits), "ppl": float(r["ppl"]),
                        "method": str(r.get("signal", "unknown")), "ci_low": None, "ci_high": None,
                    })
                except (TypeError, ValueError, KeyError):
                    pass
    wrote = False
    for model, rows in points.items():
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(6.2, 4.2)); labels = sorted(set(x["method"] for x in rows))
        for label in labels:
            vals = [x for x in rows if x["method"] == label]; vals.sort(key=lambda x: x["bits"])
            xs = [x["bits"] for x in vals]; ys = [x["ppl"] for x in vals]
            ax.plot(xs, ys, marker="o", label=label)
            if vals and all(x["ci_low"] is not None and x["ci_high"] is not None for x in vals):
                ax.fill_between(xs, [x["ci_low"] for x in vals], [x["ci_high"] for x in vals], alpha=.15)
        ax.set_xlabel("Average bits per quantized linear weight (metadata included)")
        ax.set_ylabel("WikiText-2 perplexity"); ax.set_title(model); ax.grid(alpha=.25)
        ax.legend(fontsize=8); fig.tight_layout()
        slug = model.replace("/", "_").replace(" ", "_")
        fig.savefig(out / f"ppl_vs_actual_bits_{slug}.pdf"); plt.close(fig); wrote = True
    return wrote


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("runs"))
    ap.add_argument("--input", type=Path, help="authoritative comparison CSV (Fig 2, preferred)")
    ap.add_argument("--sweeps-root", type=Path, default=Path("runs/final/sweeps"),
                    help="root of per-selector sweep JSON (Fig 1)")
    ap.add_argument("--downstream", type=Path, default=Path("results/downstream.json"),
                    help="downstream table JSON (Fig 3)")
    ap.add_argument("--f3check-root", type=Path, default=Path("results"),
                    help="dir holding f3check_*/channel_pareto.json (Fig 4)")
    ap.add_argument("--models", default="meta-llama/Llama-3.2-3B,meta-llama/Llama-3.2-1B",
                    help="comma list for the dose-response figure (Fig 1)")
    ap.add_argument("--output-dir", "--out-dir", dest="output_dir", type=Path, default=Path("figures"))
    ap.add_argument("--require-output", action="store_true", help="fail if nothing was written")
    args = ap.parse_args()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable; plots pending: {exc}")
        return 1 if args.require_output else 0
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    written = []
    for model in [m for m in args.models.split(",") if m.strip()]:
        if args.sweeps_root.exists() and fig_dose_response(plt, args.sweeps_root, model.strip(), out):
            written.append(f"fig1_dose_response_{_short(model.strip())}")
    if args.downstream.exists() and fig_decoupling(plt, args.downstream, out):
        written.append("fig0_decoupling (headline)")
    if fig_coupling(plt, out):
        written.append("fig5_coupling")
    if fig_pareto(plt, args, out):
        written.append("ppl_vs_actual_bits (fig2)")
    if args.downstream.exists() and fig_forest(plt, args.downstream, out):
        written.append("fig3_downstream_forest")
    if fig_forensic(plt, args.f3check_root, out):
        written.append("fig4_f3_forensic")
    if written:
        print(f"wrote {len(written)} figure group(s) to {out}:")
        for w in written:
            print(f"  - {w}")
    else:
        print("no figures written (no source data found)")
    if args.require_output and not written:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
