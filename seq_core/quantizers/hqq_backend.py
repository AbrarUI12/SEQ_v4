#!/usr/bin/env python3
"""HQQ backend — arbitrary low-bit (1-8) with group-wise quantization.

HQQ (Half-Quadratic Quantization) needs no backprop and little/no calibration,
supports 2/3/4/5/6/8-bit uniformly, and exposes group size + grouping axis.
This is what lets SEQ explore 3/5/6-bit and per-channel bit allocation that
bitsandbytes cannot, so the mixed-precision policy can be evaluated on a real
arbitrary-bit substrate (RQ3).
"""
from __future__ import annotations

from typing import Any, Optional, Set

import torch

from .base import QuantBackend

try:
    from hqq.core.quantize import BaseQuantizeConfig, HQQLinear

    HQQ_AVAILABLE = True
except Exception:  # noqa: BLE001
    BaseQuantizeConfig = None  # type: ignore[assignment]
    HQQLinear = None  # type: ignore[assignment]
    HQQ_AVAILABLE = False


class HqqBackend(QuantBackend):
    name = "hqq"

    def __init__(self, axis: int = 1, quant_zero: bool = False, quant_scale: bool = False):
        # axis=1 groups along the input dimension (per-output-row groups), the
        # common high-quality default; axis=0 groups along outputs.
        self.axis = axis
        self.quant_zero = quant_zero
        self.quant_scale = quant_scale

    def is_available(self) -> bool:
        return bool(HQQ_AVAILABLE)

    def supported_bits(self) -> Set[int]:
        return {1, 2, 3, 4, 5, 6, 8}

    def quantize_linear(
        self,
        layer: torch.nn.Linear,
        bits: int,
        *,
        device: str,
        compute_dtype: torch.dtype,
        group_size: Optional[int] = 64,
        **kwargs: Any,
    ) -> torch.nn.Module:
        self.check_bits(bits)
        if not self.is_available():
            raise RuntimeError("hqq is not installed (pip install hqq)")
        quant_config = BaseQuantizeConfig(
            nbits=int(bits),
            group_size=int(group_size) if group_size else None,
            quant_zero=kwargs.get("quant_zero", self.quant_zero),
            quant_scale=kwargs.get("quant_scale", self.quant_scale),
            axis=kwargs.get("axis", self.axis),
        )
        hqq_layer = HQQLinear(
            layer,
            quant_config=quant_config,
            compute_dtype=compute_dtype,
            device=device,
            initialize=True,
            del_orig=True,
        )
        return hqq_layer
