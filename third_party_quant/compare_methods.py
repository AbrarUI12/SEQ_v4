from __future__ import annotations

import copy
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

from benchmarks.core import build_bench_summary, summarize_model_disk_footprint
from benchmarks.eval_config import split_csv
from benchmarks.evaluation_suite import run_full_suite

LOGGER = logging.getLogger(__name__)

SUPPORTED_COMPARE_METHODS = {"omniquant"}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def _coerce_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return split_csv(value)
    if isinstance(value, Sequence):
        return [str(part).strip() for part in value if str(part).strip()]
    return [str(value).strip()]


def _resolve_path(root: Path, value: Optional[str], *, default: Optional[Path] = None) -> Optional[Path]:
    if value is None or str(value).strip() == "":
        return default
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (root / path).resolve()
    return path


def resolve_compare_method_plan(config: Dict[str, Any], cli_args: Optional[Any] = None) -> Dict[str, Any]:
    compare_cfg = config.get("compare_methods") or {}
    enabled_from_config = [
        str(name).strip()
        for name, method_cfg in compare_cfg.items()
        if isinstance(method_cfg, dict) and bool(method_cfg.get("enabled", False))
    ]

    requested = enabled_from_config
    cli_compare = getattr(cli_args, "compare_methods", None) if cli_args is not None else None
    if cli_compare:
        requested = split_csv(cli_compare)

    skip = split_csv(getattr(cli_args, "skip_compare_methods", None)) if cli_args is not None else []
    requested = [name for name in requested if name not in set(skip)]

    supported = [name for name in requested if name in SUPPORTED_COMPARE_METHODS]
    unsupported = [name for name in requested if name not in SUPPORTED_COMPARE_METHODS]
    return {
        "requested": requested,
        "supported": supported,
        "unsupported": unsupported,
        "configured": sorted(compare_cfg.keys()),
    }


def run_omniquant_compare(
    *,
    root: Path,
    run_dir: Path,
    model_name: str,
    device: str,
    dtype: torch.dtype,
    eval_config: Dict[str, Any],
    method_cfg: Dict[str, Any],
    global_seed: int,
    load_model_and_tokenizer: Callable[[str, str, torch.dtype, bool], Any],
    unload_model: Callable[[Any, Any], None],
) -> Dict[str, Any]:
    from third_party_quant.adapters.omniquant_adapter import (
        DEFAULT_UPSTREAM_DIR,
        ENVIRONMENT_NAME,
        OmniQuantRequest,
        run_omniquant,
    )

    method_name = "omniquant"
    method_dir = run_dir if run_dir.name == method_name else run_dir / "compare_methods" / method_name
    adapter_dir = method_dir / "adapter_run"
    saved_model_dir = method_dir / "saved_model"
    eval_dir = method_dir / "eval"
    bench_dir = method_dir / "bench"
    for path in (method_dir, adapter_dir, saved_model_dir, eval_dir, bench_dir):
        path.mkdir(parents=True, exist_ok=True)

    upstream_dir = _resolve_path(
        root,
        method_cfg.get("upstream_dir") or os.getenv("OMNIQUANT_UPSTREAM_DIR"),
        default=DEFAULT_UPSTREAM_DIR.resolve(),
    )
    python_executable = str(method_cfg.get("python_executable") or os.getenv("OMNIQUANT_PYTHON") or sys.executable)
    cache_dir = _resolve_path(root, method_cfg.get("cache_dir") or os.getenv("OMNIQUANT_CACHE_DIR"))
    if cache_dir is None:
        cache_dir = Path.home() / "seq-cache" / "omniquant"
    resume = _resolve_path(root, method_cfg.get("resume"))
    act_scales = _resolve_path(root, method_cfg.get("act_scales"))
    act_shifts = _resolve_path(root, method_cfg.get("act_shifts"))
    dry_run = bool(method_cfg.get("dry_run", False))

    if not upstream_dir.exists():
        raise FileNotFoundError(
            f"OmniQuant upstream checkout not found at {upstream_dir}. "
            f"Clone the pinned repo into third_party_quant/OmniQuant first."
        )

    let_enabled = bool(method_cfg.get("let", False))
    if let_enabled and (act_scales is None or act_shifts is None):
        raise ValueError(
            "OmniQuant LET was requested but act_scales/act_shifts were not provided. "
            "Either disable LET for weight-only W4A16 or point to the upstream act files."
        )

    raw_method_cfg = copy.deepcopy(method_cfg)
    request = OmniQuantRequest(
        model=model_name,
        output_dir=adapter_dir,
        python_executable=python_executable,
        upstream_dir=upstream_dir,
        cache_dir=cache_dir,
        save_dir=saved_model_dir,
        cuda_visible_devices=method_cfg.get("cuda_visible_devices"),
        environment_name=str(method_cfg.get("environment_name") or ENVIRONMENT_NAME),
        wbits=int(method_cfg.get("wbits", 4)),
        abits=int(method_cfg.get("abits", 16)),
        group_size=int(method_cfg["group_size"]) if method_cfg.get("group_size") is not None else None,
        alpha=float(method_cfg.get("alpha", 0.5)),
        epochs=int(method_cfg.get("epochs", 0 if resume else 20)),
        calib_dataset=str(method_cfg.get("calib_dataset", "wikitext2")),
        nsamples=int(method_cfg.get("nsamples", method_cfg.get("max_samples", 128))),
        batch_size=int(method_cfg.get("batch_size", 1)),
        seed=int(method_cfg.get("seed", global_seed)),
        tasks=",".join(_coerce_list(method_cfg.get("upstream_tasks", method_cfg.get("tasks", "")))),
        num_fewshot=int(method_cfg.get("num_fewshot", 0)),
        limit=int(method_cfg.get("limit", -1)),
        attn_implementation=str(method_cfg.get("attn_implementation", "eager")),
        net=method_cfg.get("net"),
        resume=resume,
        act_scales=act_scales,
        act_shifts=act_shifts,
        eval_ppl=bool(method_cfg.get("upstream_eval_ppl", False)),
        lwc=bool(method_cfg.get("lwc", True)),
        let=let_enabled,
        aug_loss=bool(method_cfg.get("aug_loss", False)),
        symmetric=bool(method_cfg.get("symmetric", False)),
        disable_zero_point=bool(method_cfg.get("disable_zero_point", False)),
        real_quant=bool(method_cfg.get("real_quant", False)),
        multigpu=bool(method_cfg.get("multigpu", False)),
        deactive_amp=bool(method_cfg.get("deactive_amp", False)),
        extra_args=_coerce_list(method_cfg.get("extra_args")),
    )

    _write_json(
        method_dir / "requested_config.json",
        {
            "method": method_name,
            "model_name": model_name,
            "device": device,
            "dtype": str(dtype).replace("torch.", ""),
            "requested": raw_method_cfg,
            "resolved": {
                "python_executable": python_executable,
                "upstream_dir": str(upstream_dir),
                "cache_dir": str(cache_dir),
                "save_dir": str(saved_model_dir),
                "eval_out_dir": str(eval_dir),
                "bench_dir": str(bench_dir),
                "dry_run": dry_run,
            },
        },
    )

    adapter_result = run_omniquant(request, dry_run=dry_run)
    _write_json(
        method_dir / "adapter_summary.json",
        {
            "method": method_name,
            "status": "adapter_completed_dry_run" if dry_run else "adapter_completed",
            "adapter_result": {
                "command": adapter_result.command,
                "cwd": adapter_result.cwd,
                "output_dir": adapter_result.output_dir,
                "save_dir": adapter_result.save_dir,
                "provenance_path": adapter_result.provenance_path,
                "stdout_path": adapter_result.stdout_path,
                "stderr_path": adapter_result.stderr_path,
                "returncode": adapter_result.returncode,
                "dry_run": adapter_result.dry_run,
            },
        },
    )

    if dry_run:
        return {
            "method": method_name,
            "status": "dry_run",
            "adapter_output_dir": str(adapter_dir),
            "saved_model_dir": str(saved_model_dir),
            "eval_dir": str(eval_dir),
            "bench_dir": str(bench_dir),
            "adapter_result_path": str(method_dir / "adapter_summary.json"),
            "notes": ["adapter_executed_in_dry_run_mode"],
        }

    if not (saved_model_dir / "config.json").exists():
        raise FileNotFoundError(
            f"OmniQuant finished but did not save a Hugging Face model at {saved_model_dir}. "
            "Check adapter logs and upstream save_dir behavior."
        )

    model = None
    tokenizer = None
    try:
        model, tokenizer = load_model_and_tokenizer(str(saved_model_dir), device, dtype, True)
        compare_eval_config = {
            **copy.deepcopy(eval_config),
            "bench_dir": str(bench_dir),
            "quant_model_dir": str(saved_model_dir),
            "lm_eval_model_name_or_path": str(saved_model_dir),
            "lm_eval_tokenizer_name_or_path": str(saved_model_dir),
            "lm_eval_skip_reason": None,
        }
        eval_summary = run_full_suite(
            model,
            tokenizer,
            out_dir=str(eval_dir),
            config=compare_eval_config,
            device=device,
            dtype=dtype,
        )
    finally:
        unload_model(model, tokenizer)

    size_info = summarize_model_disk_footprint(str(saved_model_dir))
    ppl_info = eval_summary.get("perplexity", {}) or {"ppl": None, "loss": None, "error": "missing"}
    latency_path = bench_dir / "latency.json"
    memory_path = bench_dir / "memory.json"
    latency_info = None
    memory_info = None
    if latency_path.exists():
        with latency_path.open("r") as handle:
            latency_info = json.load(handle)
    if memory_path.exists():
        with memory_path.open("r") as handle:
            memory_info = json.load(handle)

    bench_summary = build_bench_summary(
        size_info=size_info,
        ppl_info=ppl_info,
        latency_info=latency_info,
        memory_info=memory_info,
        effective_bits_per_param=None,
        notes=[
            "compare_method:omniquant",
            "disk_size_reflects_saved_upstream_artifact",
            "effective_bits_not_computed_no_proxy_estimate",
        ],
    )
    _write_json(bench_dir / "size.json", size_info)
    _write_json(bench_dir / "ppl.json", ppl_info)
    _write_json(bench_dir / "bench_summary.json", bench_summary)

    summary = {
        "method": method_name,
        "status": "ok",
        "adapter_output_dir": str(adapter_dir),
        "saved_model_dir": str(saved_model_dir),
        "eval_dir": str(eval_dir),
        "bench_dir": str(bench_dir),
        "adapter_result_path": str(method_dir / "adapter_summary.json"),
        "eval_summary_path": str(eval_dir / "eval_summary.json"),
        "bench_summary_path": str(bench_dir / "bench_summary.json"),
        "effective_bits": None,
        "perplexity": eval_summary.get("perplexity", {}).get("ppl"),
        "warnings": eval_summary.get("warnings", []),
    }
    _write_json(method_dir / "summary.json", summary)
    LOGGER.info("Compare method complete: %s", method_name)
    return summary
