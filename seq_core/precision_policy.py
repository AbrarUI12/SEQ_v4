#!/usr/bin/env python3
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import torch

from .entropy_metrics import parse_block_index

LOGGER = logging.getLogger(__name__)

TIER_ORDER = {"int4": 0, "int8": 1, "fp16": 2}


def assign_precision_tiers(
    weight_high: Dict[str, bool],
    act_high: Dict[str, bool],
) -> Dict[str, str]:
    tiers: Dict[str, str] = {}
    keys = set(weight_high.keys()) | set(act_high.keys())
    for name in keys:
        w_high = bool(weight_high.get(name, False))
        a_high = bool(act_high.get(name, False))
        if w_high and a_high:
            tier = "fp16"
        elif w_high ^ a_high:
            tier = "int8"
        else:
            tier = "int4"
        tiers[name] = tier
    return tiers


def _upgrade_tier(current: str, minimum: str) -> str:
    if TIER_ORDER[current] < TIER_ORDER[minimum]:
        return minimum
    return current


def _module_type_guess(name: str) -> str:
    lname = name.lower()
    if "attn" in lname or "attention" in lname:
        return "attention"
    if "mlp" in lname or "ffn" in lname or "feed_forward" in lname:
        return "mlp"
    if "proj" in lname:
        return "projection"
    return "other"


def _is_attention_output_projection(name: str) -> bool:
    lname = name.lower()
    patterns = [
        r"\.o_proj$",
        r"\.out_proj$",
        r"output\.proj$",
        r"output\._proj$",
        r"attention\.output",
        r"attn\.o_proj",
    ]
    for pat in patterns:
        if re.search(pat, lname):
            return True
    return False


def _embedding_names(model: torch.nn.Module) -> List[str]:
    names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            names.append(name)
    return names


def apply_protections(
    precision_map: Dict[str, str],
    model: torch.nn.Module,
    config: Dict[str, Any],
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, int]]:
    embedding_names = _embedding_names(model) if config.get("exclude_embeddings", True) else []
    protected_blocks = set()

    block_indices = []
    block_index_map: Dict[str, Optional[int]] = {}
    for name in precision_map:
        idx = parse_block_index(name)
        block_index_map[name] = idx
        if idx is not None:
            block_indices.append(idx)

    if config.get("protect_first_last_blocks", True) and block_indices:
        max_block = max(block_indices)
        num_blocks = int(config.get("num_protected_blocks", 1))
        for i in range(num_blocks):
            protected_blocks.add(i)
            protected_blocks.add(max_block - i)

    overrides: Dict[str, str] = {}
    counts: Dict[str, int] = {}
    final_map: Dict[str, str] = {}
    unreliable_modules = set(config.get("unreliable_modules", []))

    for name, tier in precision_map.items():
        original = tier
        reasons = []

        if any(name.startswith(emb) for emb in embedding_names):
            tier = "fp16"
            reasons.append("embedding_excluded")

        if "lm_head" in name:
            min_tier = config.get("lm_head_min_tier", "fp16")
            tier = _upgrade_tier(tier, min_tier)
            if tier != original:
                reasons.append("lm_head_min_tier")

        if config.get("protect_attn_out_proj", True) and _is_attention_output_projection(name):
            if tier == "int4":
                tier = "int8"
                reasons.append("attn_out_proj_min_int8")

        if name in unreliable_modules:
            if tier == "int4":
                tier = "int8"
                reasons.append("unreliable_entropy_min_int8")

        idx = block_index_map.get(name)
        if idx is not None and idx in protected_blocks:
            if tier == "int4":
                tier = "int8"
                reasons.append("first_last_block_min_int8")

        final_map[name] = tier
        if reasons:
            overrides[name] = ";".join(reasons)
        counts[tier] = counts.get(tier, 0) + 1

    if overrides:
        LOGGER.info("Applied %d policy overrides", len(overrides))

    return final_map, overrides, counts


def verify_policy_constraints(
    precision_map: Dict[str, str],
    model: torch.nn.Module,
    config: Dict[str, Any],
) -> None:
    embedding_names = _embedding_names(model) if config.get("exclude_embeddings", True) else []

    block_indices = []
    block_index_map: Dict[str, Optional[int]] = {}
    for name in precision_map:
        idx = parse_block_index(name)
        block_index_map[name] = idx
        if idx is not None:
            block_indices.append(idx)

    protected_blocks = set()
    if config.get("protect_first_last_blocks", True) and block_indices:
        max_block = max(block_indices)
        num_blocks = int(config.get("num_protected_blocks", 1))
        for i in range(num_blocks):
            protected_blocks.add(i)
            protected_blocks.add(max_block - i)

    unreliable_modules = set(config.get("unreliable_modules", []))

    for name, tier in precision_map.items():
        if any(name.startswith(emb) for emb in embedding_names):
            assert tier == "fp16", f"Embedding module {name} must be fp16"

        if "lm_head" in name:
            assert tier in {"int8", "fp16"}, f"lm_head {name} must be >= int8"

        if config.get("protect_attn_out_proj", True) and _is_attention_output_projection(name):
            assert tier in {"int8", "fp16"}, f"attn out proj {name} must be >= int8"

        if name in unreliable_modules:
            assert tier in {"int8", "fp16"}, f"unreliable entropy {name} must be >= int8"

        idx = block_index_map.get(name)
        if idx is not None and idx in protected_blocks:
            assert tier in {"int8", "fp16"}, f"protected block {name} must be >= int8"


def build_precision_table(
    model: torch.nn.Module,
    weight_table: Dict[str, Dict[str, Any]],
    act_table: Dict[str, Dict[str, Any]],
    weight_high: Dict[str, bool],
    act_high: Dict[str, bool],
    base_map: Dict[str, str],
    final_map: Dict[str, str],
    overrides: Dict[str, str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        weight_row = weight_table.get(name, {})
        act_row = act_table.get(name, {})
        param_count = module.weight.numel()
        if module.bias is not None:
            param_count += module.bias.numel()
        rows.append(
            {
                "module_name": name,
                "weight_entropy": weight_row.get("entropy_bits"),
                "act_entropy": act_row.get("entropy_bits"),
                "tail_act_entropy": act_row.get("tail_entropy_p95"),
                "weight_high": weight_high.get(name),
                "act_high": act_high.get(name),
                "base_tier": base_map.get(name),
                "final_tier": final_map.get(name),
                "override_reason": overrides.get(name, ""),
                "block_index_guess": parse_block_index(name),
                "module_type_guess": _module_type_guess(name),
                "param_count": int(param_count),
            }
        )
    return rows


def build_random_policy(
    base_map: Dict[str, str],
    seed: int,
) -> Dict[str, str]:
    import random

    counts: Dict[str, int] = {"int4": 0, "int8": 0, "fp16": 0}
    for tier in base_map.values():
        counts[tier] = counts.get(tier, 0) + 1

    tiers = ["int4"] * counts.get("int4", 0)
    tiers += ["int8"] * counts.get("int8", 0)
    tiers += ["fp16"] * counts.get("fp16", 0)

    rng = random.Random(seed)
    rng.shuffle(tiers)
    names = list(base_map.keys())
    return dict(zip(names, tiers))
