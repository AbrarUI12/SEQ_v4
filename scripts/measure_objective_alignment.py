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
from seq_core.proglog import banner, quiet_http_logs
from seq_core.stats_utils import spearman

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("objective_alignment")
quiet_http_logs()


def selector_scores(cols: dict, hdiag: torch.Tensor) -> dict:
    """Per-input-channel score for each selector, from the pass's column summaries.

    `cols` holds `[in]`-sized vectors computed while ΔW and H were live on the GPU.

    **Correction (2026-07-28).** An earlier version of this script scored `greedy_gain` as
    ``‖ΔW_j‖²·H_jj`` and claimed that was the greedy objective. It is not: the true first-step
    gain is ``2⟨ΔW_j,(ΔW H)_j⟩ − ‖ΔW_j‖²H_jj``, and the diagonal expression is only the
    subtracted term. On correlated Hessians the two disagree on the top pick in ~80% of layers
    (median Spearman ≈ 0.41), so the earlier run did not measure the catastrophic selector at
    all. `greedy_gain` now comes from `greedy_select.first_step_gains`, computed in-pass against
    the same damped Hessian the selector receives. The old expression is retained as
    `diag_proxy` so the previously published correlation stays auditable.
    """
    energy = cols["residual_energy"].to(torch.float32)
    scores = {
        # --- the selector's actual objective (off-diagonal coupling included) --- #
        "hessian_diag": hdiag.clone(),                 # Hessian diagonal only
        # --- read the residual but not the Hessian ------------------------------ #
        "residual_rms": energy.sqrt(),
        "residual_max": cols["residual_absmax"].to(torch.float32),
        # --- superseded: what the published run actually measured --------------- #
        "diag_proxy": (energy * hdiag),
    }
    gain = cols.get("greedy_first_step_gain")
    if gain is not None:
        scores["greedy_gain"] = gain.to(torch.float32)
    return scores


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base_bits", type=int, default=4)
    ap.add_argument("--group_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n_calib", type=int, default=128)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max_blocks", type=int, default=0,
                    help="only process the first N decoder blocks (0 = all). Alignment is a "
                         "per-layer statistic, so a median over ~20 layers is already stable; "
                         "use this for a bounded diagnostic run. Recorded in the output.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype("float16", device)
    banner(LOGGER, 1, 3, f"load {args.model}")
    model, tok = load_model_and_tokenizer(args.model, device, dtype)
    skip = [n for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and "lm_head" in n]

    banner(LOGGER, 2, 3, "sequential GPTQ pass, recording per-column compensation")
    comp: dict = {}
    cols: dict = {}
    hdiag: dict = {}
    gptq_quantize_model_sequential(
        model, tok,
        build_gptq_calibration(tok, n_samples=args.n_calib, seq_len=args.seq_len, seed=args.seed),
        bits=args.base_bits, group_size=args.group_size, seq_len=args.seq_len,
        device=str(device), max_prompts=args.n_calib, skip=skip,
        compensation_out=comp, column_scores_out=cols, hessian_diag_out=hdiag,
        max_blocks=(args.max_blocks or None),
    )
    unload_model(model, tok)

    banner(LOGGER, 3, 3, f"rank correlations over {len(comp)} layers")
    per_layer: dict = {}
    rows: dict = {}
    for name, c in comp.items():
        if name not in cols or name not in hdiag:
            continue
        c_v = c.to(torch.float32)
        if float(c_v.abs().sum()) == 0.0:
            continue                                   # nothing was compensated here
        scores = selector_scores(cols[name], hdiag[name].to(torch.float32))
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
        "max_blocks": int(args.max_blocks) or None,
        "layers_analyzed": len(per_layer),
        "target": "per-column compensation magnitude ||W_pre_quant - W_orig||_2",
        "summary": summary,
        "ranking_by_abs_median_rho": order,
        "prediction": ("objective collision predicts greedy_gain (the TRUE first-step gain, "
                       "including off-diagonal coupling) ranks compensation-bearing columns "
                       "highly. diag_proxy is the superseded ||dW_j||^2*H_jj expression the "
                       "earlier published run used by mistake; it is reported for audit only "
                       "and must not be read as the greedy objective."),
        "greedy_gain_definition": "2<dW_j,(dW H)_j> - ||dW_j||^2 H_jj  (greedy_select.first_step_gains)",
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
