#!/usr/bin/env python3
"""bitsandbytes backend — the incumbent. Supports 4-bit (nf4/fp4) and 8-bit only.

Its restricted bit grid ({4, 8, 16}) is exactly the limitation RQ3 examines:
no 3/5/6/7-bit, and round-to-nearest (no error compensation).
"""
from __future__ import annotations

from typing import Any, Optional, Set

import torch

from .base import QuantBackend

try:  # reuse the existing, tested bnb code paths
    from ..quantize_model import (
        BNB_AVAILABLE,
        quantize_linear_to_4bit,
        quantize_linear_to_8bit,
    )
except Exception:  # noqa: BLE001
    BNB_AVAILABLE = False
    quantize_linear_to_4bit = None  # type: ignore[assignment]
    quantize_linear_to_8bit = None  # type: ignore[assignment]


class BnbBackend(QuantBackend):
    name = "bnb"

    def __init__(self, quant_type: str = "nf4", double_quant: bool = True, int8_threshold: float = 6.0):
        self.quant_type = quant_type
        self.double_quant = double_quant
        self.int8_threshold = int8_threshold

    def is_available(self) -> bool:
        return bool(BNB_AVAILABLE)

    def supported_bits(self) -> Set[int]:
        return {4, 8}

    def quantize_linear(
        self,
        layer: torch.nn.Linear,
        bits: int,
        *,
        device: str,
        compute_dtype: torch.dtype,
        group_size: Optional[int] = None,  # bnb uses fixed block size (64) internally
        **kwargs: Any,
    ) -> torch.nn.Module:
        self.check_bits(bits)
        if not self.is_available():
            raise RuntimeError("bitsandbytes is not available")
        if bits == 4:
            return quantize_linear_to_4bit(
                layer,
                device=device,
                compute_dtype=compute_dtype,
                quant_type=kwargs.get("quant_type", self.quant_type),
                double_quant=kwargs.get("double_quant", self.double_quant),
            )
        return quantize_linear_to_8bit(
            layer,
            device=device,
            threshold=kwargs.get("int8_threshold", self.int8_threshold),
        )

    def dequantize_weight(self, module: torch.nn.Module) -> Optional[torch.Tensor]:
        # 4-bit: reconstruct via bnb.functional.dequantize_4bit(weight, quant_state).
        try:
            import bitsandbytes.functional as bnb_f

            weight = getattr(module, "weight", None)
            quant_state = getattr(weight, "quant_state", None)
            if weight is not None and quant_state is not None:
                w = bnb_f.dequantize_4bit(weight.data, quant_state)
                if isinstance(w, torch.Tensor) and w.dim() == 2:
                    return w.detach()
        except Exception:  # noqa: BLE001
            pass
        return None  # 8-bit (LLM.int8) reconstruction is not supported here
