#!/usr/bin/env python3
"""Add logical weight-only storage breakdowns to LLMC baseline summaries.

LLMC's saved artifacts are fake-quant BF16 weights, so their serialized byte
size is not a compact 4-bit checkpoint.  This script reads safetensors tensor
metadata without materializing the weights, counts quantized linear weights at
the requested logical bit width, and includes scale/zero/permutation metadata
in the primary comparison axis.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safetensors import safe_open

from seq_core.storage_accounting import account_storage


_DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "U16": 2, "I16": 2, "U32": 4,
    "I32": 4, "U64": 8, "I64": 8, "F16": 2, "BF16": 2, "F32": 4,
    "F64": 8,
}


def _numel(shape: list[int]) -> int:
    return math.prod(int(x) for x in shape)


def _artifact_storage(model_path: Path, bits: int) -> dict[str, Any]:
    safes = sorted(model_path.glob("*.safetensors"))
    if not safes:
        raise FileNotFoundError(f"no safetensors checkpoint under {model_path}")
    qvals = scale_values = zero_values = 0
    scale_bits = zero_bits = 16
    metadata_bytes = 0
    total_weights = embedding_values = lm_head_values = 0
    with safe_open(str(safes[0]), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for key in keys:
            if key.startswith("model.") and key.endswith(".weight") and not ".buf_" in key:
                shape = list(handle.get_slice(key).get_shape())
                n = _numel(shape)
                total_weights += n
                if "embed_tokens.weight" in key:
                    embedding_values += n
                if key.endswith("lm_head.weight"):
                    lm_head_values += n
                scales_key = key[:-len("weight")] + "buf_scales"
                if scales_key in keys:
                    qvals += n
                    s = handle.get_slice(scales_key)
                    scale_values += _numel(list(s.get_shape()))
                    scale_bits = _DTYPE_BYTES.get(str(s.get_dtype()), 2) * 8
                    zeros_key = key[:-len("weight")] + "buf_zeros"
                    if zeros_key in keys:
                        z = handle.get_slice(zeros_key)
                        zero_values += _numel(list(z.get_shape()))
                        zero_bits = _DTYPE_BYTES.get(str(z.get_dtype()), 2) * 8
                    prefix = key[:-len("weight")]
                    for meta in keys:
                        if meta.startswith(prefix + "buf_") and meta not in {scales_key, zeros_key}:
                            m = handle.get_slice(meta)
                            metadata_bytes += _numel(list(m.get_shape())) * _DTYPE_BYTES.get(str(m.get_dtype()), 2)
    if qvals <= 0:
        raise ValueError(f"no quantized linear weights identified under {model_path}")
    unquantized = max(0, total_weights - qvals - embedding_values - lm_head_values)
    return account_storage(
        quantized_values=qvals, quantized_bits=bits,
        scale_values=scale_values, scale_bits=scale_bits,
        zero_point_values=zero_values, zero_point_bits=zero_bits,
        group_metadata_bytes=metadata_bytes,
        unquantized_parameter_values=unquantized,
        embedding_values=embedding_values, lm_head_values=lm_head_values,
        parameter_count=total_weights,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--bits", type=int, default=4)
    args = ap.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    for model, rows in data.items():
        for row in rows:
            path = row.get("model_path")
            if not path:
                continue
            try:
                storage = _artifact_storage(Path(path), args.bits)
            except (FileNotFoundError, ValueError) as exc:
                print(f"WARNING: {model} {row.get('method')}: {exc}")
                continue
            row["storage"] = storage
            row["bits"] = storage["actual_weight_bits_per_param"]
            row["nominal_bits"] = float(args.bits)
            row["model_bits"] = storage["actual_model_bits_per_parameter"]
            row["accounting_status"] = "recomputed_from_safetensors_metadata"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
