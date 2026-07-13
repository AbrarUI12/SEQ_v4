#!/usr/bin/env python3
"""Backend-agnostic quantization interface for SEQ.

The SEQ *policy* decides a per-module bit-width; a ``QuantBackend`` decides how
those bits are realized. Decoupling the two lets us answer RQ3 (is bitsandbytes
a limitation?) by swapping bnb for HQQ / GPTQ / torchao without touching the
allocation logic, and by using arbitrary bit-widths (3/5/6-bit) that bnb cannot.

Every quantized module is tagged with ``_seq_bits`` / ``_seq_backend`` so
verification and effective-bit accounting are backend-independent.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set

import torch

LOGGER = logging.getLogger(__name__)

SEQ_BITS_ATTR = "_seq_bits"
SEQ_BACKEND_ATTR = "_seq_backend"


def tag_quantized(module: torch.nn.Module, bits: int, backend: str) -> torch.nn.Module:
    setattr(module, SEQ_BITS_ATTR, int(bits))
    setattr(module, SEQ_BACKEND_ATTR, str(backend))
    return module


def detect_seq_bits(module: torch.nn.Module) -> Optional[int]:
    """Bits recorded by SEQ, or 16 for a plain (unquantized) Linear."""
    bits = getattr(module, SEQ_BITS_ATTR, None)
    if bits is not None:
        return int(bits)
    if isinstance(module, torch.nn.Linear):
        return 16
    return None


class QuantBackend(ABC):
    """Turn a single ``nn.Linear`` into a quantized module at ``bits`` width."""

    name: str = "base"

    @abstractmethod
    def supported_bits(self) -> Set[int]:
        ...

    def is_available(self) -> bool:
        return True

    @abstractmethod
    def quantize_linear(
        self,
        layer: torch.nn.Linear,
        bits: int,
        *,
        device: str,
        compute_dtype: torch.dtype,
        group_size: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.nn.Module:
        ...

    def check_bits(self, bits: int) -> None:
        if bits not in self.supported_bits():
            raise ValueError(
                f"backend '{self.name}' does not support {bits}-bit "
                f"(supports {sorted(self.supported_bits())})"
            )

    def dequantize_weight(self, module: torch.nn.Module) -> Optional[torch.Tensor]:
        """Return the dequantized weight [out, in] of a module this backend made.

        Used by the reconstruction-error sensitivity harness to compute the real
        quantization error ΔW = W − Q(W). Backends that cannot reconstruct return
        None (the harness then skips that unit with a warning).
        """
        return None


# --------------------------------------------------------------------------- #
# Module tree helpers (kept local to avoid import cycles).
# --------------------------------------------------------------------------- #
def get_module_by_name(model: torch.nn.Module, name: str) -> torch.nn.Module:
    module = model
    for part in name.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def set_module_by_name(model: torch.nn.Module, name: str, new_module: torch.nn.Module) -> None:
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


# --------------------------------------------------------------------------- #
# Apply an arbitrary per-module bit-width map with a chosen backend.
# --------------------------------------------------------------------------- #
def apply_bit_map(
    model: torch.nn.Module,
    bit_map: Dict[str, int],
    backend: QuantBackend,
    *,
    device: str,
    fp16_dtype: torch.dtype = torch.float16,
    compute_dtype: torch.dtype = torch.float16,
    group_size: Optional[int] = 64,
    keep_full_precision_at: int = 16,
    **backend_kwargs: Any,
) -> Dict[str, Any]:
    """Quantize each named Linear to its target bit-width.

    ``bits >= keep_full_precision_at`` keeps the module in ``fp16_dtype``.
    Returns a summary with per-bit counts and any per-module errors.
    """
    counts: Dict[str, int] = {}
    errors: List[Dict[str, Any]] = []
    for name, bits in bit_map.items():
        try:
            module = get_module_by_name(model, name)
        except Exception as exc:  # noqa: BLE001
            errors.append({"module": name, "bits": bits, "error": f"lookup: {exc}"})
            continue
        if not isinstance(module, torch.nn.Linear):
            counts["skipped"] = counts.get("skipped", 0) + 1
            continue
        bits = int(bits)
        if bits >= keep_full_precision_at:
            module.to(device=device, dtype=fp16_dtype)
            tag_quantized(module, 16, "fp16")
            counts[16] = counts.get(16, 0) + 1
            continue
        try:
            backend.check_bits(bits)
            new_module = backend.quantize_linear(
                module,
                bits,
                device=device,
                compute_dtype=compute_dtype,
                group_size=group_size,
                **backend_kwargs,
            )
            tag_quantized(new_module, bits, backend.name)
            set_module_by_name(model, name, new_module)
            counts[bits] = counts.get(bits, 0) + 1
        except Exception as exc:  # noqa: BLE001
            errors.append({"module": name, "bits": bits, "error": str(exc)})
    if errors:
        LOGGER.warning("apply_bit_map: %d modules failed", len(errors))
    return {"backend": backend.name, "counts": counts, "errors": errors}


def verify_bit_map(model: torch.nn.Module, bit_map: Dict[str, int]) -> Dict[str, Any]:
    mismatches = []
    for name, expected in bit_map.items():
        try:
            module = get_module_by_name(model, name)
        except Exception:  # noqa: BLE001
            mismatches.append({"module": name, "expected": int(expected), "found": "missing"})
            continue
        found = detect_seq_bits(module)
        exp = 16 if int(expected) >= 16 else int(expected)
        if found != exp:
            mismatches.append({"module": name, "expected": exp, "found": found})
    return {"mismatches": mismatches, "num_mismatches": len(mismatches)}


def effective_bits_from_map(
    bit_map: Dict[str, int],
    param_counts: Dict[str, int],
) -> Dict[str, Any]:
    """Parameter-weighted effective bits for an arbitrary-bit map."""
    total = 0
    weighted = 0.0
    by_bits: Dict[int, int] = {}
    for name, bits in bit_map.items():
        c = int(param_counts.get(name, 0))
        b = 16 if int(bits) >= 16 else int(bits)
        total += c
        weighted += float(b) * c
        by_bits[b] = by_bits.get(b, 0) + c
    if total == 0:
        return {"total_params": 0, "effective_bits": None, "params_by_bits": {}}
    return {
        "total_params": total,
        "effective_bits": weighted / total,
        "params_by_bits": by_bits,
        "percent_by_bits": {b: c / total for b, c in by_bits.items()},
    }
