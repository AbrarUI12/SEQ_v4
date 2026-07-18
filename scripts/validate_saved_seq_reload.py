#!/usr/bin/env python3
"""Check that an exported dense SEQ checkpoint reproduces its measured PPL."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seq_core.pipeline import load_model_and_tokenizer, resolve_device, resolve_dtype, unload_model
from seq_core.sensitivity import make_ppl_fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path", type=Path)
    ap.add_argument("--expected", type=float, required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tolerance", type=float, default=1e-3)
    args = ap.parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype("float16", device)
    model, tokenizer = load_model_and_tokenizer(args.model_path, device, dtype)
    try:
        ppl = make_ppl_fn(
            dataset_name="wikitext2", split="test", seq_len=2048, device=device,
            dtype=dtype, mode="canonical", full_corpus=True,
        )(model, tokenizer)
    finally:
        unload_model(model, tokenizer)
    result = {"model_path": str(args.model_path), "reload_ppl": ppl,
              "expected_ppl": args.expected, "absolute_difference": abs(ppl - args.expected),
              "tolerance": args.tolerance,
              "status": "PASS" if abs(ppl - args.expected) <= args.tolerance else "FAIL"}
    print(json.dumps(result, indent=2))
    (args.model_path / "reload_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
