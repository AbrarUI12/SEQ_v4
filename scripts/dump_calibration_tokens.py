#!/usr/bin/env python3
"""Persist a calibration set as exact token ids, so a later run can be token-identical.

The §5.3 experiment can only be *size*-matched today: the LightCompress base was calibrated
externally (its own preprocessing and seed) and its calibration tokens were never saved, so no
downstream run can reproduce them. This tool removes that limitation going forward — dump the
token ids once, hash them, and feed the same file to every stage that needs a Hessian:

    python scripts/dump_calibration_tokens.py --model meta-llama/Llama-3.2-3B \\
        --n_samples 128 --seq_len 2048 --seed 1234 --out calib/base_tokens.pt

    python -m seq_core.channel_sweep ... --selector_calib_tokens calib/base_tokens.pt

The saved payload records the SHA-256 of the ids, so a run that consumes it can prove which
calibration set it used rather than asserting it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoTokenizer

from seq_core.gptq import build_gptq_calibration


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="tokenizer to use")
    ap.add_argument("--n_samples", type=int, default=128)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    texts = build_gptq_calibration(tok, n_samples=args.n_samples,
                                   seq_len=args.seq_len, seed=args.seed)
    rows = [tok(t, return_tensors="pt", truncation=True,
                max_length=args.seq_len)["input_ids"][0] for t in texts]
    width = min(int(r.shape[0]) for r in rows)
    ids = torch.stack([r[:width] for r in rows])          # [n_samples, width]
    sha = hashlib.sha256(ids.numpy().tobytes()).hexdigest()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"input_ids": ids, "sha256": sha, "model": args.model,
                "n_samples": args.n_samples, "seq_len": args.seq_len, "seed": args.seed},
               args.out)
    meta = args.out.with_suffix(".json")
    meta.write_text(json.dumps({"sha256": sha, "shape": list(ids.shape), "model": args.model,
                                "n_samples": args.n_samples, "seq_len": args.seq_len,
                                "seed": args.seed}, indent=2), encoding="utf-8")
    print(f"wrote {args.out}  shape={tuple(ids.shape)}  tokens={ids.numel()}")
    print(f"sha256={sha}")
    print(f"metadata -> {meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
