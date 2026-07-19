#!/usr/bin/env python3
"""Standalone perplexity runner for SEQ.

This keeps only the PPL calculation path from the old compare-matrix workflow.
It intentionally does not import or run quantization methods.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .core import compute_ppl
from seq_core.pipeline import (
    load_model_and_tokenizer,
    now_timestamp,
    resolve_device,
    resolve_dtype,
    sanitize_name,
    set_seed,
    unload_model,
)


def _parse_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_optional_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null", "na"}:
        return None
    return int(text)


def _split_models(args: argparse.Namespace) -> List[str]:
    raw: List[str] = []
    if args.model:
        raw.append(args.model)
    if args.models:
        raw.extend(args.models)
    models: List[str] = []
    for item in raw:
        for part in str(item).split(","):
            part = part.strip()
            if part:
                models.append(part)
    if not models:
        raise SystemExit("Provide --model or --models.")
    return models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone SEQ perplexity evaluation.")
    parser.add_argument("--model", type=str, default="", help="Single HF model name or local path.")
    parser.add_argument("--models", type=str, nargs="*", default=[], help="One or more models; comma-separated values are accepted.")
    parser.add_argument("--device", type=str, default="auto", help="auto|cuda|cpu")
    parser.add_argument("--dtype", type=str, default="float16", help="float16|bfloat16|float32")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--trust_remote_code", action="store_true")

    parser.add_argument("--ppl_mode", type=str, choices=["canonical", "proxy"], default="canonical")
    parser.add_argument("--ppl_dataset", type=str, default="wikitext2")
    parser.add_argument("--ppl_split", type=str, default=None)
    parser.add_argument("--ppl_full_corpus", type=str, default=None)
    parser.add_argument("--ppl_seq_len", type=int, default=None)
    parser.add_argument("--ppl_stride", type=str, default=None)
    parser.add_argument("--ppl_max_examples", type=str, default=None)

    parser.add_argument("--output_dir", type=str, default="ppl_results")
    parser.add_argument("--no_write_json", action="store_true", help="Print only; do not write result JSON files.")
    return parser.parse_args()


def resolve_ppl_config(args: argparse.Namespace) -> Dict[str, Any]:
    ppl_mode = str(args.ppl_mode).strip().lower()
    ppl_split = args.ppl_split or ("test" if ppl_mode == "canonical" else "validation")
    ppl_seq_len = int(args.ppl_seq_len if args.ppl_seq_len is not None else (2048 if ppl_mode == "canonical" else 256))
    ppl_stride = _parse_optional_int(args.ppl_stride)
    ppl_max_examples = _parse_optional_int(args.ppl_max_examples)
    ppl_full_corpus = _parse_bool(args.ppl_full_corpus, default=(ppl_mode == "canonical"))

    # Match old run_compare_matrix behavior exactly for canonical PPL:
    # full corpus, no stride override, no example cap.
    if ppl_mode == "canonical":
        ppl_full_corpus = True
        ppl_stride = None
        ppl_max_examples = None

    return {
        "ppl_mode": ppl_mode,
        "ppl_dataset": args.ppl_dataset,
        "ppl_split": ppl_split,
        "ppl_full_corpus": ppl_full_corpus,
        "ppl_seq_len": ppl_seq_len,
        "ppl_stride": ppl_stride,
        "ppl_max_examples": ppl_max_examples,
    }


def run_model_ppl(model_name: str, args: argparse.Namespace, ppl_config: Dict[str, Any]) -> Dict[str, Any]:
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    set_seed(int(args.seed))

    model = None
    tokenizer = None
    try:
        model, tokenizer = load_model_and_tokenizer(
            model_name,
            device,
            dtype,
            trust_remote_code=bool(args.trust_remote_code),
        )
        ppl = compute_ppl(
            model,
            tokenizer,
            dataset_name=str(ppl_config["ppl_dataset"]),
            split=str(ppl_config["ppl_split"]),
            seq_len=int(ppl_config["ppl_seq_len"]),
            max_examples=ppl_config["ppl_max_examples"],
            stride=ppl_config["ppl_stride"],
            device=device,
            dtype=dtype,
            seed=int(args.seed),
            mode=str(ppl_config["ppl_mode"]),
            full_corpus=bool(ppl_config["ppl_full_corpus"]),
        )
        return {
            "model_name": model_name,
            "device": device,
            "dtype": str(dtype).replace("torch.", ""),
            "seed": int(args.seed),
            "ppl_config": ppl_config,
            "perplexity": ppl,
        }
    finally:
        if model is not None or tokenizer is not None:
            unload_model(model, tokenizer)


def write_result(output_dir: Path, result: Dict[str, Any]) -> Path:
    model_slug = sanitize_name(str(result["model_name"]))
    cfg = result["ppl_config"]
    scope = "full" if cfg.get("ppl_full_corpus") else f"n{cfg.get('ppl_max_examples') or 'all'}"
    filename = f"{now_timestamp()}__{model_slug}__{cfg['ppl_mode']}_L{cfg['ppl_seq_len']}_{scope}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return path


def main() -> int:
    args = parse_args()
    models = _split_models(args)
    ppl_config = resolve_ppl_config(args)
    output_dir = Path(args.output_dir)

    all_results = []
    for model_name in models:
        result = run_model_ppl(model_name, args, ppl_config)
        all_results.append(result)
        if not args.no_write_json:
            path = write_result(output_dir, result)
            result["result_path"] = str(path)
        print(json.dumps(result, indent=2, sort_keys=True))

    if len(all_results) > 1 and not args.no_write_json:
        summary_path = output_dir / f"{now_timestamp()}__ppl_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(all_results, handle, indent=2, sort_keys=True)
        print(json.dumps({"summary_path": str(summary_path)}, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
