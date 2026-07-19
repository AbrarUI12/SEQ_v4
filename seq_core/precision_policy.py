#!/usr/bin/env python3
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import torch

from .entropy_metrics import parse_block_index

LOGGER = logging.getLogger(__name__)

TIER_ORDER = {"int4": 0, "int8": 1, "fp16": 2}


def tier_rank(tier: str) -> int:
    return TIER_ORDER[str(tier).lower()]


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


def upgrade_tier(current: str, minimum: str) -> str:
    if tier_rank(current) < tier_rank(minimum):
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


def is_attention_output_projection(name: str) -> bool:
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


def is_attention_projection(name: str) -> bool:
    lname = name.lower()
    markers = [
        ".q_proj",
        ".k_proj",
        ".v_proj",
        ".o_proj",
        ".out_proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "out_proj",
    ]
    return any(marker in lname for marker in markers)


def is_gate_down_projection(name: str) -> bool:
    lname = name.lower()
    return "gate_proj" in lname or "down_proj" in lname


def is_lm_head(name: str) -> bool:
    return "lm_head" in name.lower()


def is_first_or_last_block(name: str, num_layers: int) -> bool:
    idx = parse_block_index(name)
    if idx is None:
        return False
    return idx == 0 or idx == max(0, int(num_layers) - 1)


def compute_percentile_ranks(values_by_module: Dict[str, float]) -> Dict[str, float]:
    valid = {
        name: float(value)
        for name, value in values_by_module.items()
        if value is not None and torch.isfinite(torch.tensor(float(value))).item()
    }
    if not valid:
        return {}
    ordered = sorted(valid.items(), key=lambda item: (item[1], item[0]))
    if len(ordered) == 1:
        return {ordered[0][0]: 1.0}

    ranks: Dict[str, float] = {}
    idx = 0
    denom = float(len(ordered) - 1)
    while idx < len(ordered):
        value = ordered[idx][1]
        end = idx + 1
        while end < len(ordered) and ordered[end][1] == value:
            end += 1
        midpoint = (idx + end - 1) / 2.0
        rank = float(midpoint / denom)
        for pos in range(idx, end):
            ranks[ordered[pos][0]] = rank
        idx = end
    return ranks


def assign_risk_score_tiers(
    weight_table: Dict[str, Dict[str, Any]],
    act_table: Dict[str, Dict[str, Any]],
    unreliable_modules: set,
    fp16_threshold: float = 0.90,
    int8_threshold: float = 0.65,
) -> Tuple[Dict[str, str], Dict[str, Dict[str, Optional[float]]]]:
    reliable_names = set(weight_table.keys()) | set(act_table.keys())
    reliable_names = {name for name in reliable_names if name not in unreliable_modules}
    reliable_names = {
        name
        for name in reliable_names
        if weight_table.get(name, {}).get("entropy_bits") is not None
        and act_table.get(name, {}).get("entropy_bits") is not None
    }
    weight_values = {name: weight_table[name]["entropy_bits"] for name in reliable_names}
    act_values = {name: act_table[name]["entropy_bits"] for name in reliable_names}
    weight_ranks = compute_percentile_ranks(weight_values)
    act_ranks = compute_percentile_ranks(act_values)

    tiers: Dict[str, str] = {}
    risk_info: Dict[str, Dict[str, Optional[float]]] = {}
    for name in sorted(set(weight_table.keys()) | set(act_table.keys())):
        weight_rank = weight_ranks.get(name)
        act_rank = act_ranks.get(name)
        risk_score = None
        if weight_rank is not None and act_rank is not None:
            risk_score = max(weight_rank, act_rank)
            if risk_score >= fp16_threshold:
                tiers[name] = "fp16"
            elif risk_score >= int8_threshold:
                tiers[name] = "int8"
            else:
                tiers[name] = "int4"
        else:
            tiers[name] = "int4"
        risk_info[name] = {
            "weight_rank": weight_rank,
            "act_rank": act_rank,
            "risk_score": risk_score,
            "risk_policy_fp16_threshold": float(fp16_threshold),
            "risk_policy_int8_threshold": float(int8_threshold),
        }
    return tiers, risk_info


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

        if is_lm_head(name):
            min_tier = config.get("lm_head_min_tier", "fp16")
            tier = upgrade_tier(tier, min_tier)
            if tier != original:
                reasons.append("lm_head_min_tier")

        before = tier
        if config.get("protect_attn_out_proj", True) and is_attention_output_projection(name):
            tier = upgrade_tier(tier, "int8")
            if tier != before:
                reasons.append("attn_out_proj_min_int8")

        before = tier
        if config.get("protect_all_attention_proj", False) and is_attention_projection(name):
            tier = upgrade_tier(tier, "int8")
            if tier != before:
                reasons.append("protect_all_attention_proj_int8")

        before = tier
        if config.get("protect_gate_down_proj", False) and is_gate_down_projection(name):
            tier = upgrade_tier(tier, "int8")
            if tier != before:
                reasons.append("protect_gate_down_int8")

        if config.get("protect_unreliable", True) and name in unreliable_modules:
            before = tier
            tier = upgrade_tier(tier, "int8")
            if tier != before:
                reasons.append("unreliable_entropy_min_int8")

        idx = block_index_map.get(name)
        if idx is not None and idx in protected_blocks:
            before = tier
            tier = upgrade_tier(tier, "int8")
            if tier != before:
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

        if is_lm_head(name):
            assert tier in {"int8", "fp16"}, f"lm_head {name} must be >= int8"

        if config.get("protect_attn_out_proj", True) and is_attention_output_projection(name):
            assert tier in {"int8", "fp16"}, f"attn out proj {name} must be >= int8"

        if config.get("protect_all_attention_proj", False) and is_attention_projection(name):
            assert tier in {"int8", "fp16"}, f"attention projection {name} must be >= int8"

        if config.get("protect_gate_down_proj", False) and is_gate_down_projection(name):
            assert tier in {"int8", "fp16"}, f"gate/down projection {name} must be >= int8"

        if config.get("protect_unreliable", True) and name in unreliable_modules:
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
    risk_info: Optional[Dict[str, Dict[str, Optional[float]]]] = None,
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
        risk_row = (risk_info or {}).get(name, {})
        row = {
            "module_name": name,
            "weight_entropy": weight_row.get("entropy_bits"),
            "activation_entropy": act_row.get("entropy_bits"),
            "act_entropy": act_row.get("entropy_bits"),
            "tail_act_entropy": act_row.get("tail_entropy_p95"),
            "weight_high": weight_high.get(name),
            "activation_high": act_high.get(name),
            "act_high": act_high.get(name),
            "base_tier": base_map.get(name),
            "final_tier": final_map.get(name),
            "override_reason": overrides.get(name, ""),
            "unreliable": bool(
                weight_row.get("flags", {}).get("unreliable")
                or act_row.get("flags", {}).get("unreliable")
            ),
            "num_parameters": int(param_count),
            "block_index_guess": parse_block_index(name),
            "module_type_guess": _module_type_guess(name),
            "param_count": int(param_count),
        }
        for key in (
            "weight_rank",
            "act_rank",
            "risk_score",
            "risk_policy_fp16_threshold",
            "risk_policy_int8_threshold",
        ):
            if key in risk_row:
                row[key] = risk_row.get(key)
        rows.append(row)
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
