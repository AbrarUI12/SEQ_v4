#!/usr/bin/env python3
"""Pluggable quantization backends for SEQ.

Usage::

    from seq_core.quantizers import get_backend, apply_bit_map
    backend = get_backend("hqq")
    apply_bit_map(model, {"model.layers.0.mlp.down_proj": 3, ...}, backend,
                  device="cuda", group_size=64)
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

from .base import (
    QuantBackend,
    apply_bit_map,
    detect_seq_bits,
    effective_bits_from_map,
    tag_quantized,
    verify_bit_map,
)
from .bnb_backend import BnbBackend
from .hqq_backend import HqqBackend

_REGISTRY: Dict[str, Callable[..., QuantBackend]] = {
    "bnb": BnbBackend,
    "bitsandbytes": BnbBackend,
    "hqq": HqqBackend,
}


def register_backend(name: str, factory: Callable[..., QuantBackend]) -> None:
    _REGISTRY[name.lower()] = factory


def available_backends() -> List[str]:
    return sorted(_REGISTRY.keys())


def get_backend(name: str, **kwargs: Any) -> QuantBackend:
    key = str(name).lower()
    if key not in _REGISTRY:
        raise ValueError(f"unknown backend '{name}'; available: {available_backends()}")
    return _REGISTRY[key](**kwargs)


__all__ = [
    "QuantBackend",
    "BnbBackend",
    "HqqBackend",
    "get_backend",
    "register_backend",
    "available_backends",
    "apply_bit_map",
    "verify_bit_map",
    "effective_bits_from_map",
    "detect_seq_bits",
    "tag_quantized",
]
