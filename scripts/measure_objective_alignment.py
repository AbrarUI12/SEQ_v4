#!/usr/bin/env python3
"""E1 — measure objective collision directly.

For each linear layer, a sequential GPTQ pass records how much error compensation each input
column absorbed before it was quantized:

    comp_j = ‖W_pre_quant[:, j] − W_orig[:, j]‖₂

Each selection signal is then scored on the same layer and rank-correlated (Spearman) with
`comp`. Objective collision predicts that selectors sharing GPTQ's objective
(‖ΔW X‖² = tr(ΔW H ΔWᵀ)) rank compensation-bearing columns, i.e. high |ρ|, while selectors
orthogonal to it sit near zero:

    greedy_gain (full objective)   >  hessian_diag (diagonal only)
                                   >  residual_max / residual_rms (no Hessian)
                                   ≈  act-free controls ≈ 0

Run per model; results are aggregated (median/mean ρ over layers) and written as JSON.
No perplexity evaluation, so this is cheap relative to a sweep.
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from seq_core.gptq import build_gptq_calibration, gptq_quantize_model_sequential
from seq_core.pipeline import load_model_and_tokenizer, resolve_device, resolve_dtype, unload_model
from seq_core.stats_utils import spearman

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("objective_alignment")


def selector_scores(dW: torch.Tensor, hdiag: torch.Tensor) -> dict:
    """Per-input-channel score for each selector, on one layer.

    `dW` is [out, in] (float32, CPU), `hdiag` is [in]. Scores are defined to match the
    selectors used in the sweeps; `greedy_gain` is the first-step marginal gain of the
    Hessian-weighted objective, which is exactly what `greedy`/`greedy_indep` rank by
    (an iterative selector's first pick is its argmax).
    """
    energy = (dW * dW).sum(dim=0)                      # ‖ΔW_j‖²
    return {
        # --- share GPTQ's objective ---------------------------------------- #
        "greedy_gain": (energy * hdiag),               # ∝ first-step gain, full coupling
        "hessian_diag": hdiag.clone(),                 # Hessian diagonal only
        # --- read the residual but not the Hessian -------------------------- #
        "residual_rms": energy.sqrt(),
        "residual_max": dW.abs().max(dim=0).values,
        # --- read neither ---------------------------------------------------- #
        "weight_magnitude": torch.zeros_like(energy),  # placeholder, filled by caller
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base_bits", type=int, default=4)
    ap.add_argument("--group_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n_calib", type=int, default=128)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype("float16", device)
    model, tok = load_model_and_tokenizer(args.model, device, dtype)
    skip = [n for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and "lm_head" in n]

    comp: dict = {}
    resid: dict = {}
    hdiag: dict = {}
    LOGGER.info("sequential GPTQ pass, recording per-column compensation ...")
    gptq_quantize_model_sequential(
        model, tok,
        build_gptq_calibration(tok, n_samples=args.n_calib, seq_len=args.seq_len, seed=args.seed),
        bits=args.base_bits, group_size=args.group_size, seq_len=args.seq_len,
        device=str(device), max_prompts=args.n_calib, skip=skip,
        compensation_out=comp, residual_out=resid, hessian_diag_out=hdiag,
    )
    unload_model(model, tok)
    LOGGER.info("recorded %d layers; computing rank correlations ...", len(comp))

    per_layer: dict = {}
    rows: dict = {}
    for name, c in comp.items():
        if name not in resid or name not in hdiag:
            continue
        c_v = c.to(torch.float32)
        if float(c_v.abs().sum()) == 0.0:
            continue                                   # nothing was compensated here
        scores = selector_scores(resid[name].to(torch.float32), hdiag[name].to(torch.float32))
        scores.pop("weight_magnitude", None)
        target = c_v.tolist()
        entry = {}
        for sel, s in scores.items():
            rho = spearman(s.tolist(), target)
            if rho is not None:
                entry[sel] = round(float(rho), 4)
                rows.setdefault(sel, []).append(float(rho))
        per_layer[name] = entry

    summary = {}
    for sel, vals in rows.items():
        if not vals:
            continue
        summary[sel] = {
            "median_rho": round(statistics.median(vals), 4),
            "mean_rho": round(statistics.fmean(vals), 4),
            "n_layers": len(vals),
            "frac_positive": round(sum(v > 0 for v in vals) / len(vals), 4),
        }
    order = sorted(summary, key=lambda s: -abs(summary[s]["median_rho"]))
    result = {
        "model": args.model, "base_bits": args.base_bits, "group_size": args.group_size,
        "seed": args.seed, "n_calib": args.n_calib, "skip_lm_head": True,
        "gptq_path": "sequential",
        "target": "per-column compensation magnitude ||W_pre_quant - W_orig||_2",
        "summary": summary,
        "ranking_by_abs_median_rho": order,
        "prediction": ("objective collision predicts greedy_gain > hessian_diag > "
                       "residual_rms/residual_max in |rho|"),
        "per_layer": per_layer,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for sel in order:
        LOGGER.info("  %-16s median rho=%+.4f (n=%d)", sel,
                    summary[sel]["median_rho"], summary[sel]["n_layers"])
    LOGGER.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
