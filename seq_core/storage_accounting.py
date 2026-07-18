#!/usr/bin/env python3
"""Single source of truth for SEQ storage and effective-bit accounting.

The accounting is deliberately independent of a quantizer implementation.  A
caller supplies counts of values and metadata; the returned dictionary contains
both nominal weight bits and the complete checkpoint estimate.  Keeping this in
one small, pure-Python module prevents Pareto tables from silently mixing the
old nominal formula with serialized-size measurements.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Optional


@dataclass
class StorageBreakdown:
    dense_quantized_weight_bytes: int = 0
    quantization_scale_bytes: int = 0
    zero_point_bytes: int = 0
    group_metadata_bytes: int = 0
    fp16_residual_bytes: int = 0
    int8_tier_bytes: int = 0
    channel_index_bytes: int = 0
    layer_metadata_bytes: int = 0
    bias_bytes: int = 0
    unquantized_parameter_bytes: int = 0
    embedding_bytes: int = 0
    lm_head_bytes: int = 0
    alignment_padding_bytes: int = 0
    serialized_checkpoint_bytes: Optional[int] = None

    @property
    def estimated_bytes(self) -> int:
        return int(sum(v for k, v in asdict(self).items()
                       if k != "serialized_checkpoint_bytes" and isinstance(v, int)))

    @property
    def actual_bytes(self) -> int:
        return int(self.serialized_checkpoint_bytes or self.estimated_bytes)


def _ceil_bytes(bits: float) -> int:
    return int(math.ceil(max(0.0, float(bits)) / 8.0))


def account_storage(
    *,
    quantized_values: int = 0,
    quantized_bits: int = 0,
    scale_values: int = 0,
    scale_bits: int = 16,
    zero_point_values: int = 0,
    zero_point_bits: int = 16,
    group_metadata_bytes: int = 0,
    fp16_residual_values: int = 0,
    int8_values: int = 0,
    channel_index_values: int = 0,
    channel_index_bits: int = 16,
    layer_metadata_bytes: int = 0,
    bias_values: int = 0,
    unquantized_parameter_values: int = 0,
    embedding_values: int = 0,
    lm_head_values: int = 0,
    serialized_checkpoint_bytes: Optional[int] = None,
    alignment_bytes: int = 1,
    parameter_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Return a complete, auditable storage report.

    ``quantized_values`` are packed at ``quantized_bits`` per value.  Residuals,
    biases, embeddings, and lm_head are FP16 unless their count is zero.  The
    function reports bits per parameter using *all* supplied parameters, not
    only quantized linear weights.
    """
    b = StorageBreakdown(
        dense_quantized_weight_bytes=_ceil_bytes(quantized_values * quantized_bits),
        quantization_scale_bytes=_ceil_bytes(scale_values * scale_bits),
        zero_point_bytes=_ceil_bytes(zero_point_values * zero_point_bits),
        group_metadata_bytes=max(0, int(group_metadata_bytes)),
        fp16_residual_bytes=_ceil_bytes(fp16_residual_values * 16),
        int8_tier_bytes=_ceil_bytes(int8_values * 8),
        channel_index_bytes=_ceil_bytes(channel_index_values * channel_index_bits),
        layer_metadata_bytes=max(0, int(layer_metadata_bytes)),
        bias_bytes=_ceil_bytes(bias_values * 16),
        unquantized_parameter_bytes=_ceil_bytes(unquantized_parameter_values * 16),
        embedding_bytes=_ceil_bytes(embedding_values * 16),
        lm_head_bytes=_ceil_bytes(lm_head_values * 16),
        serialized_checkpoint_bytes=(int(serialized_checkpoint_bytes)
                                     if serialized_checkpoint_bytes is not None else None),
    )
    if alignment_bytes > 1:
        rem = b.estimated_bytes % int(alignment_bytes)
        b.alignment_padding_bytes = (int(alignment_bytes) - rem) % int(alignment_bytes)
    stored_values = (int(quantized_values) + int(fp16_residual_values) + int(int8_values)
                     + int(unquantized_parameter_values) + int(embedding_values)
                     + int(lm_head_values) + int(bias_values))
    total_params = int(parameter_count) if parameter_count is not None else stored_values
    nominal_bits = ((quantized_values * quantized_bits + fp16_residual_values * 16
                     + int8_values * 8 + unquantized_parameter_values * 16
                     + embedding_values * 16 + lm_head_values * 16 + bias_values * 16)
                    / total_params if total_params else 0.0)
    report = asdict(b)
    report.update({
        "estimated_total_bytes": b.estimated_bytes,
        "actual_total_bytes": b.actual_bytes,
        "total_parameters": total_params,
        "nominal_weight_bits": float(nominal_bits),
        "quantized_linear_bits": (float(quantized_values * quantized_bits / quantized_values)
                                   if quantized_values else 0.0),
        "actual_model_bits_per_parameter": float(b.actual_bytes * 8 / total_params)
        if total_params else 0.0,
        "serialized_checkpoint_bits_per_parameter": (
            float(b.serialized_checkpoint_bytes * 8 / total_params)
            if b.serialized_checkpoint_bytes is not None and total_params else None),
    })
    return report


def account_layer(in_features: int, out_features: int, base_bits: int,
                  protected_fp16: int = 0, protected_int8: int = 0,
                  *, group_size: int = 128, scale_bits: int = 16,
                  include_zero_point: bool = True,
                  index_bits: Optional[int] = None) -> Dict[str, Any]:
    """Convenience accounting for one column-protected linear layer."""
    n = max(0, int(in_features)); out = max(0, int(out_features))
    protected_fp16 = min(n, max(0, int(protected_fp16)))
    protected_int8 = min(n - protected_fp16, max(0, int(protected_int8)))
    # ChannelProtectedLinear stores the complete low-bit base and sparse
    # corrections; protected columns do not replace bytes in that base.
    qvals = n * out
    groups = math.ceil(n / max(1, group_size)) * out
    idx_bits = int(index_bits if index_bits is not None else max(1, math.ceil(math.log2(max(2, n)))))
    return account_storage(
        quantized_values=qvals, quantized_bits=base_bits,
        scale_values=groups, scale_bits=scale_bits,
        zero_point_values=groups if include_zero_point else 0,
        fp16_residual_values=protected_fp16 * out,
        int8_values=protected_int8 * out,
        channel_index_values=protected_fp16 + protected_int8,
        channel_index_bits=idx_bits,
        parameter_count=n * out,
    )


__all__ = ["StorageBreakdown", "account_storage", "account_layer"]
