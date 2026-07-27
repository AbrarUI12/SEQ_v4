#!/usr/bin/env python3
"""How a 6x perplexity collapse can cost nothing downstream: per-token loss decomposition.

Perplexity is exp(mean NLL) over every supervised token, so it is dominated by whichever tokens
are worst. Downstream multiple-choice accuracy is not. If the protection damage is concentrated
in a small minority of tokens, both observations can be true of the same checkpoint.

This script scores two checkpoints on the *identical* token stream — the canonical WikiText-2
protocol used everywhere else in the paper (full test split joined with "\\n\\n", tokenized once,
contiguous non-overlapping `seq_len` windows, final short tail dropped) — and reports, per token:

  * each model's NLL, and their difference
  * what fraction of tokens account for what fraction of the total NLL increase
  * the perplexity that would result if the worst-k% of tokens were excluded
  * how many tokens actually got *better*, and the median change

The headline number is the last one: if removing e.g. the worst 1% of tokens collapses the
perplexity gap, the damage is concentrated, not diffuse, and a metric that is not dominated by
tail tokens (accuracy) can be untouched.

Usage (base first, then the damaged checkpoint):

    python scripts/per_token_loss_decomposition.py \\
      --base  runs/final/llmc/Llama-3.2-3B/gptq/artifacts/fake_quant_model \\
      --other runs/final/downstream/checkpoints/Llama-3.2-3B/greedy_gptq \\
      --tokenizer meta-llama/Llama-3.2-3B \\
      --out results/pertoken_Llama-3.2-3B.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from seq_core.pipeline import load_model_and_tokenizer, resolve_device, resolve_dtype, unload_model
from seq_core.proglog import Stage, banner, heartbeat, quiet_http_logs

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("per_token")
quiet_http_logs()


def build_stream(tokenizer, dataset_name: str, split: str, seq_len: int):
    """Canonical stream: identical construction to benchmarks.core.compute_canonical_ppl."""
    from benchmarks.core import _load_text_dataset

    ds, field = _load_text_dataset(dataset_name, split)
    # Must match benchmarks.core.compute_canonical_ppl EXACTLY: non-string rows are kept as ""
    # so the "\n\n" separator count is preserved. Dropping them instead shifts the whole token
    # stream and the per-token NLLs would no longer correspond to the paper's perplexity.
    text = "\n\n".join((row if isinstance(row, str) else "") for row in ds[field])
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    n_full = ids.shape[0] // seq_len
    LOGGER.info("stream: %d tokens -> %d non-overlapping windows of %d (tail %d dropped)",
                ids.shape[0], n_full, seq_len, ids.shape[0] - n_full * seq_len)
    return ids[: n_full * seq_len].view(n_full, seq_len)


@torch.no_grad()
def per_token_nll(model, windows: torch.Tensor, device) -> torch.Tensor:
    """NLL of every supervised target token, concatenated across windows (float32, CPU).

    Each window contributes seq_len-1 targets, matching the shifted-label convention used by
    the paper's perplexity: mean of this vector, exponentiated, reproduces that perplexity.
    """
    out = []
    t0 = __import__("time").monotonic()
    for i in range(windows.shape[0]):
        ids = windows[i : i + 1].to(device)
        logits = model(ids).logits.float()
        # shift: predict token t+1 from position t
        lp = torch.log_softmax(logits[:, :-1, :], dim=-1)
        tgt = ids[:, 1:]
        nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        out.append(nll.squeeze(0).cpu())
        del logits, lp, nll
        heartbeat(LOGGER, "windows", i + 1, windows.shape[0], t0,
                  every=max(1, windows.shape[0] // 10))
    return torch.cat(out).to(torch.float32)


def decompose(base: torch.Tensor, other: torch.Tensor) -> dict:
    """Concentration statistics for the NLL increase of `other` over `base`."""
    delta = other - base
    total_increase = float(delta.clamp(min=0).sum())
    order = torch.argsort(delta, descending=True)
    n = delta.numel()

    # what share of the *total NLL increase* the worst-k% of tokens carry
    share = {}
    for pct in (0.1, 0.5, 1.0, 5.0, 10.0):
        k = max(1, int(n * pct / 100.0))
        share[f"worst_{pct}pct_share_of_increase"] = round(
            float(delta[order[:k]].clamp(min=0).sum()) / max(1e-9, total_increase), 4)

    # perplexity after excluding the worst-k% of tokens, for both models
    trimmed = {}
    for pct in (0.0, 0.1, 0.5, 1.0, 5.0):
        k = int(n * pct / 100.0)
        keep = order[k:] if k else order
        trimmed[f"excl_worst_{pct}pct"] = {
            "base_ppl": round(float(torch.exp(base[keep].mean())), 4),
            "other_ppl": round(float(torch.exp(other[keep].mean())), 4),
        }
    return {
        "n_tokens": int(n),
        "base_ppl": round(float(torch.exp(base.mean())), 4),
        "other_ppl": round(float(torch.exp(other.mean())), 4),
        "mean_nll_base": round(float(base.mean()), 6),
        "mean_nll_other": round(float(other.mean()), 6),
        "median_delta_nll": round(float(delta.median()), 6),
        "frac_tokens_worse": round(float((delta > 0).float().mean()), 4),
        "frac_tokens_better": round(float((delta < 0).float().mean()), 4),
        "delta_nll_quantiles": {
            q: round(float(torch.quantile(delta, q)), 6)
            for q in (0.01, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999)
        },
        "concentration": share,
        "trimmed_perplexity": trimmed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="reference checkpoint (the unprotected base)")
    ap.add_argument("--other", required=True, help="damaged checkpoint (e.g. greedy@GPTQ export)")
    ap.add_argument("--tokenizer", default="", help="tokenizer id (defaults to --base)")
    ap.add_argument("--dataset", default="wikitext2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save_arrays", action="store_true",
                    help="also save the raw per-token NLL vectors (large; for figures)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype("float16", device)

    banner(LOGGER, 1, 3, f"base: {args.base}")
    model, tok = load_model_and_tokenizer(args.base, device, dtype)
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
    windows = build_stream(tok, args.dataset, args.split, args.seq_len)
    with Stage(LOGGER, "per-token NLL (base)"):
        nll_base = per_token_nll(model, windows, device)
    unload_model(model, tok)

    banner(LOGGER, 2, 3, f"other: {args.other}")
    model, tok2 = load_model_and_tokenizer(args.other, device, dtype)
    with Stage(LOGGER, "per-token NLL (other)"):
        nll_other = per_token_nll(model, windows, device)
    unload_model(model, tok2)

    banner(LOGGER, 3, 3, "decomposition")
    stats = decompose(nll_base, nll_other)
    result = {"base": args.base, "other": args.other, "dataset": args.dataset,
              "split": args.split, "seq_len": args.seq_len,
              "protocol": "canonical continuous stream, non-overlapping windows, tail dropped",
              **stats}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.save_arrays:
        torch.save({"base": nll_base, "other": nll_other},
                   args.out.with_suffix(".pt"))

    LOGGER.info("base ppl=%.3f  other ppl=%.3f  (%.1fx)", stats["base_ppl"], stats["other_ppl"],
                stats["other_ppl"] / max(1e-9, stats["base_ppl"]))
    LOGGER.info("tokens worse: %.1f%%  better: %.1f%%  median dNLL=%+.4f",
                100 * stats["frac_tokens_worse"], 100 * stats["frac_tokens_better"],
                stats["median_delta_nll"])
    for k, v in stats["concentration"].items():
        LOGGER.info("  %s = %.1f%%", k, 100 * v)
    for k, v in stats["trimmed_perplexity"].items():
        LOGGER.info("  %-22s base=%8.3f  other=%8.3f", k, v["base_ppl"], v["other_ppl"])
    LOGGER.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
