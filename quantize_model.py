#!/usr/bin/env python3
import logging
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOGGER = logging.getLogger(__name__)

try:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear4bit, Linear8bitLt, Params4bit
    try:
        from bitsandbytes.nn import Int8Params
    except Exception:
        Int8Params = None
    BNB_AVAILABLE = True
except Exception:
    Linear4bit = None
    Linear8bitLt = None
    Params4bit = None
    Int8Params = None
    BNB_AVAILABLE = False


def get_module_by_name(model: torch.nn.Module, module_name: str):
    parts = module_name.split(".")
    module = model
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def set_module_by_name(model: torch.nn.Module, module_name: str, new_module) -> None:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def _require_bnb() -> None:
    if not BNB_AVAILABLE:
        raise RuntimeError("bitsandbytes is required for INT4/INT8 quantization")


def quantize_linear_to_4bit(
    layer: torch.nn.Linear,
    device: str,
    compute_dtype: torch.dtype,
    quant_type: str,
    double_quant: bool,
) -> torch.nn.Module:
    _require_bnb()
    in_features = layer.in_features
    out_features = layer.out_features
    has_bias = layer.bias is not None

    weight_data = layer.weight.detach().cpu().contiguous()

    new_layer = Linear4bit(
        input_features=in_features,
        output_features=out_features,
        bias=has_bias,
        compute_dtype=compute_dtype,
        compress_statistics=double_quant,
        quant_type=quant_type,
    )

    new_layer.weight = Params4bit(
        weight_data,
        requires_grad=False,
        quant_type=quant_type,
        compress_statistics=double_quant,
    )

    new_layer = new_layer.to(device)

    if has_bias:
        new_layer.bias = torch.nn.Parameter(layer.bias.detach().clone().to(device))

    return new_layer


def quantize_linear_to_8bit(
    layer: torch.nn.Linear,
    device: str,
    threshold: float,
) -> torch.nn.Module:
    _require_bnb()
    in_features = layer.in_features
    out_features = layer.out_features
    has_bias = layer.bias is not None

    new_layer = Linear8bitLt(
        in_features,
        out_features,
        bias=has_bias,
        has_fp16_weights=False,
        threshold=threshold,
    )

    weight_cpu = layer.weight.detach().cpu().contiguous()

    if Int8Params is not None:
        new_layer.weight = Int8Params(weight_cpu, requires_grad=False)
    else:
        try:
            new_layer.weight = torch.nn.Parameter(weight_cpu, requires_grad=False)
        except Exception as exc:
            raise RuntimeError("Int8Params not available; cannot quantize to int8") from exc

    if has_bias:
        new_layer.bias = torch.nn.Parameter(layer.bias.detach().cpu().clone(), requires_grad=False)

    new_layer = new_layer.to(device)
    return new_layer


def apply_mixed_precision(
    model: torch.nn.Module,
    precision_map: Dict[str, str],
    device: str,
    dtype_fp16: torch.dtype,
    bnb_4bit: str = "nf4",
    bnb_compute_dtype: torch.dtype = torch.float16,
    int8_threshold: float = 6.0,
    double_quant: bool = True,
) -> Dict[str, Any]:
    _require_bnb()
    replacement_counts = {"int4": 0, "int8": 0, "fp16": 0, "skipped": 0}
    errors = []

    for name, tier in precision_map.items():
        module = get_module_by_name(model, name)
        if not isinstance(module, torch.nn.Linear):
            replacement_counts["skipped"] += 1
            continue

        if tier == "int4":
            try:
                q = quantize_linear_to_4bit(
                    module,
                    device=device,
                    compute_dtype=bnb_compute_dtype,
                    quant_type=bnb_4bit,
                    double_quant=double_quant,
                )
                set_module_by_name(model, name, q)
                replacement_counts["int4"] += 1
            except Exception as exc:
                errors.append({"module": name, "tier": tier, "error": str(exc)})
        elif tier == "int8":
            try:
                q = quantize_linear_to_8bit(
                    module,
                    device=device,
                    threshold=int8_threshold,
                )
                set_module_by_name(model, name, q)
                replacement_counts["int8"] += 1
            except Exception as exc:
                errors.append({"module": name, "tier": tier, "error": str(exc)})
        else:
            module = module.to(device=device, dtype=dtype_fp16)
            replacement_counts["fp16"] += 1

    if errors:
        LOGGER.warning("Quantization errors: %d", len(errors))

    return {"replacement_counts": replacement_counts, "errors": errors}


def _verify_layer_kind(model: torch.nn.Module, name: str) -> str:
    module = get_module_by_name(model, name)
    if Linear4bit is not None and isinstance(module, Linear4bit):
        return "int4"
    if Linear8bitLt is not None and isinstance(module, Linear8bitLt):
        return "int8"
    if isinstance(module, torch.nn.Linear):
        return "fp16"
    return f"other:{type(module).__name__}"


def verify_replacements(
    model: torch.nn.Module,
    precision_map: Dict[str, str],
) -> Dict[str, Any]:
    mismatches = []
    counts = {"int4": 0, "int8": 0, "fp16": 0, "other": 0}
    for name, expected in precision_map.items():
        found = _verify_layer_kind(model, name)
        if found.startswith("other"):
            counts["other"] += 1
        else:
            counts[found] = counts.get(found, 0) + 1
        if expected != found:
            mismatches.append({"module": name, "expected": expected, "found": found})
    return {"counts": counts, "mismatches": mismatches}


def compute_effective_bits(
    precision_map: Dict[str, str],
    param_counts: Dict[str, int],
) -> Dict[str, Any]:
    totals = {"int4": 0, "int8": 0, "fp16": 0}
    total_params = 0
    for name, tier in precision_map.items():
        count = int(param_counts.get(name, 0))
        total_params += count
        if tier in totals:
            totals[tier] += count

    if total_params == 0:
        return {
            "total_params": 0,
            "effective_bits": None,
            "percent_params": {},
        }

    effective = (
        totals["int4"] * 4.0 + totals["int8"] * 8.0 + totals["fp16"] * 16.0
    ) / float(total_params)

    percent = {
        "int4": totals["int4"] / float(total_params),
        "int8": totals["int8"] / float(total_params),
        "fp16": totals["fp16"] / float(total_params),
    }

    return {
        "total_params": int(total_params),
        "effective_bits": float(effective),
        "percent_params": percent,
        "params_by_tier": totals,
    }


def save_quantized(model: torch.nn.Module, tokenizer, out_dir: str) -> None:
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)


def safe_disable_grad(model: torch.nn.Module) -> None:
    for param in model.parameters():
        try:
            if param.is_floating_point() or torch.is_complex(param):
                param.requires_grad_(False)
        except Exception as exc:
            LOGGER.warning("Failed to disable grad for param: %s", exc)


def reload_quantized(
    model_name: str,
    precision_map: Dict[str, str],
    device: str,
    dtype: torch.dtype,
    sanity_prompt: str = "Sanity check: respond with OK.",
    max_new_tokens: int = 16,
) -> Tuple[Optional[str], Optional[str]]:
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        model.to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as exc:
        return None, f"reload_failed: {exc}"

    try:
        safe_disable_grad(model)
        apply_mixed_precision(
            model,
            precision_map,
            device=device,
            dtype_fp16=dtype,
            bnb_4bit="nf4",
            bnb_compute_dtype=torch.float16,
        )
    except Exception as exc:
        return None, f"reload_quantization_failed: {exc}"

    inputs = tokenizer(sanity_prompt, return_tensors="pt").to(device)
    try:
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                remove_invalid_values=True,
                renormalize_logits=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        return text, None
    except Exception as exc:
        return None, f"sanity_generation_failed: {exc}"
