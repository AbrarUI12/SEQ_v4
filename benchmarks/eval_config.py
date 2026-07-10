from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Optional, Set


METRIC_GROUPS = {
    "seq_core",
    "ppl",
    "tail_risk",
    "json_stress",
    "temperature_sweep",
    "long_context",
    "latency_memory",
    "size",
    "quant_accounting",
    "lm_eval",
    "mmlu",
    "zero_shot",
}

DEFAULT_GROUPS = {
    "seq_core",
    "latency_memory",
    "size",
    "quant_accounting",
}

ALL_GROUPS = set(METRIC_GROUPS)


def split_csv(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, Iterable):
        return [str(part).strip() for part in value if str(part).strip()]
    return [str(value).strip()]


def _evaluation_block(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get("evaluation", config) if isinstance(config, dict) else {}


def _expand_groups(groups: Iterable[str]) -> Set[str]:
    expanded: Set[str] = set()
    for group in groups:
        value = str(group).strip()
        if not value:
            continue
        if value == "all":
            expanded.update(ALL_GROUPS)
        elif value == "seq_core":
            expanded.update(
                {
                    "seq_core",
                    "ppl",
                    "tail_risk",
                    "json_stress",
                    "temperature_sweep",
                    "long_context",
                    "mmlu",
                    "zero_shot",
                }
            )
        else:
            expanded.add(value)
    return expanded


def _groups_from_config(config: Dict[str, Any]) -> Set[str]:
    evaluation = _evaluation_block(config)
    configured = evaluation.get("enabled_metric_groups")
    if configured:
        groups = _expand_groups(split_csv(configured))
    else:
        groups = _expand_groups(DEFAULT_GROUPS)

    lm_eval_cfg = evaluation.get("lm_eval") or {}
    if isinstance(lm_eval_cfg, dict) and bool(lm_eval_cfg.get("enabled", False)):
        groups.add("lm_eval")
    return groups


def resolve_metric_plan(config: Dict[str, Any], cli_args: Optional[Any] = None) -> Dict[str, Any]:
    groups = _groups_from_config(config)

    metrics_arg = getattr(cli_args, "metrics", None) if cli_args is not None else None
    if metrics_arg:
        groups = _expand_groups(split_csv(metrics_arg))

    if cli_args is not None and getattr(cli_args, "lm_eval", False):
        groups.add("lm_eval")
    if cli_args is not None and getattr(cli_args, "no_lm_eval", False):
        groups.discard("lm_eval")

    if cli_args is not None and getattr(cli_args, "seq_only", False):
        groups.discard("lm_eval")

    if cli_args is not None and getattr(cli_args, "lm_eval_only", False):
        groups = {"lm_eval", "quant_accounting"}

    skip_arg = getattr(cli_args, "skip_metrics", None) if cli_args is not None else None
    if skip_arg:
        groups.difference_update(_expand_groups(split_csv(skip_arg)))

    run_seq_core = bool(groups.intersection({"seq_core", "ppl", "tail_risk", "json_stress", "temperature_sweep", "long_context", "mmlu", "zero_shot"}))
    return {
        "enabled_groups": sorted(groups),
        "run_seq_core": run_seq_core,
        "run_seq_ppl": "ppl" in groups,
        "run_tail_risk": "tail_risk" in groups,
        "run_json_stress": "json_stress" in groups,
        "run_temperature_sweep": "temperature_sweep" in groups,
        "run_long_context": "long_context" in groups,
        "run_latency_memory": "latency_memory" in groups,
        "run_size": "size" in groups,
        "run_quant_accounting": "quant_accounting" in groups,
        "run_lm_eval": "lm_eval" in groups,
        "run_mmlu": "mmlu" in groups,
        "run_zero_shot": "zero_shot" in groups,
    }


def default_lm_eval_presets() -> Dict[str, Dict[str, Any]]:
    return {
        "smoke": {"tasks": ["hellaswag"], "limit": 10, "num_fewshot": 0},
        "standard": {
            "tasks": ["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande"],
            "limit": None,
            "num_fewshot": 0,
        },
        "paper": {
            "tasks": ["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande", "lambada_openai"],
            "limit": None,
            "num_fewshot": 0,
        },
    }


def resolve_lm_eval_config(config: Dict[str, Any], cli_args: Optional[Any] = None) -> Dict[str, Any]:
    evaluation = _evaluation_block(config)
    lm_eval_cfg = copy.deepcopy(evaluation.get("lm_eval") or {})
    presets = copy.deepcopy(default_lm_eval_presets())
    presets.update(copy.deepcopy(evaluation.get("lm_eval_presets") or {}))

    preset_name = lm_eval_cfg.get("task_preset")
    cli_preset = getattr(cli_args, "lm_eval_preset", None) if cli_args is not None else None
    if cli_preset:
        preset_name = cli_preset

    if preset_name and preset_name in presets:
        preset_cfg = copy.deepcopy(presets[preset_name])
        preset_cfg.update(lm_eval_cfg)
        lm_eval_cfg = preset_cfg
    lm_eval_cfg["task_preset"] = preset_name
    lm_eval_cfg["presets"] = presets

    if cli_args is not None:
        if getattr(cli_args, "lm_eval", False) or getattr(cli_args, "lm_eval_only", False):
            lm_eval_cfg["enabled"] = True
        if getattr(cli_args, "no_lm_eval", False) or getattr(cli_args, "seq_only", False):
            lm_eval_cfg["enabled"] = False
        if getattr(cli_args, "lm_eval_tasks", None):
            lm_eval_cfg["tasks"] = split_csv(cli_args.lm_eval_tasks)
        if getattr(cli_args, "lm_eval_limit", None) is not None:
            lm_eval_cfg["limit"] = cli_args.lm_eval_limit
        if getattr(cli_args, "lm_eval_num_fewshot", None) is not None:
            lm_eval_cfg["num_fewshot"] = cli_args.lm_eval_num_fewshot
        if getattr(cli_args, "lm_eval_batch_size", None) is not None:
            lm_eval_cfg["batch_size"] = cli_args.lm_eval_batch_size
        if getattr(cli_args, "lm_eval_log_samples", False):
            lm_eval_cfg["log_samples"] = True
        if getattr(cli_args, "lm_eval_fail_policy", None):
            lm_eval_cfg["fail_policy"] = cli_args.lm_eval_fail_policy

    lm_eval_cfg.setdefault("enabled", False)
    lm_eval_cfg.setdefault("tasks", ["hellaswag"])
    lm_eval_cfg["tasks"] = split_csv(lm_eval_cfg.get("tasks")) or ["hellaswag"]
    lm_eval_cfg.setdefault("num_fewshot", 0)
    lm_eval_cfg.setdefault("batch_size", 1)
    lm_eval_cfg.setdefault("limit", None)
    lm_eval_cfg.setdefault("device", None)
    lm_eval_cfg.setdefault("apply_chat_template", False)
    lm_eval_cfg.setdefault("log_samples", False)
    lm_eval_cfg.setdefault("use_cache", None)
    lm_eval_cfg.setdefault("model_backend", "hf")
    lm_eval_cfg.setdefault("extra_model_args", {})
    lm_eval_cfg.setdefault("output_subdir", "lm_eval")
    lm_eval_cfg.setdefault("fail_policy", "warn")
    lm_eval_cfg.setdefault("quantized_reloadable", False)
    return lm_eval_cfg
