#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

from benchmarks.seq_lm_eval import run_lm_eval_suite
from seq_core.pipeline import (
    get_git_commit,
    get_pkg_version,
    now_timestamp,
    run_experiment,
    resolve_device,
    resolve_dtype,
    sanitize_name,
)
from third_party_quant.adapters.omniquant_adapter import (
    DEFAULT_UPSTREAM_DIR,
    ENVIRONMENT_NAME,
    OmniQuantRequest,
    run_omniquant,
)


LOGGER = logging.getLogger("run_compare_matrix")
SUPPORTED_METHODS = ("seq", "omniquant")
METHOD_ALIASES = {"omniqunat": "omniquant"}
_UNICODE_SPACE_RE = re.compile(r"[\u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]+")
_LONG_OPTS_WITH_VALUE = (
    "--models",
    "--methods",
    "--benchmarks",
    "--device",
    "--dtype",
    "--experiments_file",
    "--output_dir",
    "--degeneracy_mode",
    "--lm_eval_limit",
    "--lm_eval_num_fewshot",
    "--lm_eval_batch_size",
    "--lm_eval_model_backend",
    "--lm_eval_fail_policy",
    "--omniquant_python",
    "--omniquant_upstream_dir",
    "--omniquant_cache_dir",
    "--omniquant_wbits",
    "--omniquant_abits",
    "--omniquant_epochs",
    "--omniquant_nsamples",
)


def _split_attached_long_option(token: str) -> List[str]:
    if not token.startswith("--"):
        return [token]
    for opt in _LONG_OPTS_WITH_VALUE:
        if token == opt or token.startswith(f"{opt}="):
            return [token]
        if token.startswith(opt):
            attached = token[len(opt) :]
            if attached:
                return [opt, attached]
    return [token]


def _normalize_cli_argv(argv: List[str]) -> List[str]:
    normalized: List[str] = []
    for token in argv:
        cleaned = token.replace("\u200b", "").replace("\ufeff", "")
        if _UNICODE_SPACE_RE.search(cleaned):
            for part in [p for p in _UNICODE_SPACE_RE.split(cleaned) if p]:
                normalized.extend(_split_attached_long_option(part))
        else:
            normalized.extend(_split_attached_long_option(cleaned))
    return normalized


def _split_csv(text: Optional[str]) -> List[str]:
    if text is None:
        return []
    if isinstance(text, (list, tuple, set)):
        values: List[str] = []
        for item in text:
            values.extend(_split_csv(str(item)))
        return values
    out: List[str] = []
    for part in str(text).split(","):
        value = part.strip()
        if value:
            out.append(value)
    return out


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _resolve_path(root: Path, value: Optional[str], default: Optional[Path] = None) -> Optional[Path]:
    if value is None or str(value).strip() == "":
        return default
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-method comparison: SEQ and upstream OmniQuant.")
    parser.add_argument("--models", required=True, help="Comma-separated Hugging Face model names or local model paths.")
    parser.add_argument("--methods", default="seq,omniquant", help="Comma-separated methods. Supported: seq,omniquant.")
    parser.add_argument(
        "--benchmarks",
        default="hellaswag",
        help="Comma-separated EleutherAI lm-eval task names, e.g. hellaswag,arc_easy,piqa.",
    )
    parser.add_argument("--device", default="auto", help="auto|cuda|cpu or lm-eval-compatible device string.")
    parser.add_argument("--dtype", default="float16", help="float16|bfloat16|float32.")
    parser.add_argument("--experiments_file", default="experiments.yaml")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--degeneracy_mode", choices=["old", "rms"], default="rms")
    parser.add_argument("--trust_remote_code", action="store_true")

    parser.add_argument("--lm_eval_limit", type=int, default=None)
    parser.add_argument("--lm_eval_num_fewshot", type=int, default=0)
    parser.add_argument("--lm_eval_batch_size", default="1")
    parser.add_argument("--lm_eval_model_backend", default="hf")
    parser.add_argument("--lm_eval_fail_policy", choices=["warn", "error", "skip"], default="warn")
    parser.add_argument("--lm_eval_apply_chat_template", action="store_true")
    parser.add_argument("--lm_eval_log_samples", action="store_true")

    parser.add_argument("--omniquant_python", default=None)
    parser.add_argument("--omniquant_upstream_dir", default=None)
    parser.add_argument("--omniquant_cache_dir", default=None)
    parser.add_argument("--omniquant_wbits", type=int, default=None)
    parser.add_argument("--omniquant_abits", type=int, default=None)
    parser.add_argument("--omniquant_epochs", type=int, default=None)
    parser.add_argument("--omniquant_nsamples", type=int, default=None)
    parser.add_argument("--omniquant_real_quant", action="store_true")
    parser.add_argument("--omniquant_dry_run", action="store_true")

    args, unknown = parser.parse_known_args(_normalize_cli_argv(sys.argv[1:]))
    if unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    return args


def _validate_methods(raw_methods: List[str]) -> List[str]:
    methods = [METHOD_ALIASES.get(method.lower(), method.lower()) for method in raw_methods]
    unknown = [method for method in methods if method not in SUPPORTED_METHODS]
    if unknown:
        raise SystemExit(
            "Unsupported --methods entries: "
            + ", ".join(unknown)
            + ". This runner only supports real methods: "
            + ",".join(SUPPORTED_METHODS)
        )
    deduped: List[str] = []
    for method in methods:
        if method not in deduped:
            deduped.append(method)
    return deduped


def _lm_eval_config(args: argparse.Namespace, tasks: List[str], device: str) -> Dict[str, Any]:
    return {
        "enabled": True,
        "tasks": tasks,
        "num_fewshot": int(args.lm_eval_num_fewshot),
        "batch_size": args.lm_eval_batch_size,
        "limit": args.lm_eval_limit,
        "device": device,
        "model_backend": args.lm_eval_model_backend,
        "fail_policy": args.lm_eval_fail_policy,
        "apply_chat_template": bool(args.lm_eval_apply_chat_template),
        "log_samples": bool(args.lm_eval_log_samples),
        "output_subdir": "lm_eval",
        "extra_model_args": {"trust_remote_code": True} if args.trust_remote_code else {},
    }


def _make_seq_config(
    base_config: Dict[str, Any],
    *,
    model_name: str,
    device: str,
    dtype: str,
    degeneracy_mode: str,
    trust_remote_code: bool,
) -> Dict[str, Any]:
    config = copy.deepcopy(base_config)
    config.setdefault("model", {})
    config["model"].update(
        {
            "name": model_name,
            "device": device,
            "dtype": dtype,
            "trust_remote_code": bool(trust_remote_code),
        }
    )
    config.setdefault("calibration", {})
    config["calibration"]["degeneracy_mode"] = degeneracy_mode

    # The comparison benchmark is lm-eval only. SEQ still performs the real
    # quantization pipeline, but local proxy benchmark tasks are disabled here.
    config.setdefault("benchmarks", {})
    config["benchmarks"]["run_ppl"] = False
    config["benchmarks"]["run_size"] = True
    config.setdefault("evaluation", {})
    seq_metrics = dict(config["evaluation"].get("seq_metrics") or {})
    for name in ("ppl", "tail_risk", "json_stress", "temperature_sweep", "long_context", "latency_memory"):
        block = dict(seq_metrics.get(name) or {})
        block["enabled"] = False
        seq_metrics[name] = block
    config["evaluation"]["seq_metrics"] = seq_metrics
    config["evaluation"]["mmlu"] = {**(config["evaluation"].get("mmlu") or {}), "enabled": False}
    config["evaluation"]["zero_shot"] = {**(config["evaluation"].get("zero_shot") or {}), "enabled": False}
    return config


def _find_new_run(before: set[Path], runs_root: Path) -> Path:
    after = set(runs_root.glob("run_*"))
    new_runs = sorted(after - before, key=lambda p: p.name)
    if new_runs:
        return new_runs[-1]
    if after:
        return sorted(after, key=lambda p: p.name)[-1]
    raise RuntimeError(f"No SEQ run directories found under {runs_root}")


def _run_seq(
    *,
    root: Path,
    model_name: str,
    model_dir: Path,
    base_config: Dict[str, Any],
    args: argparse.Namespace,
    device: str,
) -> Dict[str, Any]:
    runs_root = model_dir / "seq" / "seq_runs"
    reports_root = model_dir / "seq" / "seq_reports"
    runs_root.mkdir(parents=True, exist_ok=True)
    before = set(runs_root.glob("run_*"))
    seq_config = _make_seq_config(
        base_config,
        model_name=model_name,
        device=device,
        dtype=args.dtype,
        degeneracy_mode=args.degeneracy_mode,
        trust_remote_code=bool(args.trust_remote_code),
    )
    _write_json(model_dir / "seq" / "seq_compare_config.json", seq_config)
    run_experiment(
        root,
        seq_config,
        {"name": "compare_seq", "policy": "dual_entropy"},
        root / "calibration_prompts.json",
        root / "eval_prompts.json",
        reports_root,
        runs_root,
    )
    seq_run = _find_new_run(before, runs_root)
    quant_model_dir = seq_run / "model_quantized"
    return {
        "method": "seq",
        "status": "quantized",
        "run_dir": str(model_dir / "seq"),
        "seq_run_dir": str(seq_run),
        "model_path": str(quant_model_dir) if (quant_model_dir / "config.json").exists() else None,
    }


def _run_omniquant(
    *,
    root: Path,
    model_name: str,
    model_dir: Path,
    base_config: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    method_dir = model_dir / "omniquant"
    adapter_dir = method_dir / "adapter_run"
    saved_model_dir = method_dir / "saved_model"
    method_cfg = dict((base_config.get("compare_methods") or {}).get("omniquant") or {})

    upstream_dir = _resolve_path(
        root,
        args.omniquant_upstream_dir or method_cfg.get("upstream_dir") or os.getenv("OMNIQUANT_UPSTREAM_DIR"),
        DEFAULT_UPSTREAM_DIR.resolve(),
    )
    python_executable = (
        args.omniquant_python
        or method_cfg.get("python_executable")
        or os.getenv("OMNIQUANT_PYTHON")
        or sys.executable
    )
    cache_dir = _resolve_path(
        root,
        args.omniquant_cache_dir or method_cfg.get("cache_dir") or os.getenv("OMNIQUANT_CACHE_DIR"),
        Path.home() / "seq-cache" / "omniquant",
    )
    request = OmniQuantRequest(
        model=model_name,
        output_dir=adapter_dir,
        python_executable=str(python_executable),
        upstream_dir=upstream_dir,
        cache_dir=cache_dir,
        save_dir=saved_model_dir,
        cuda_visible_devices=method_cfg.get("cuda_visible_devices"),
        environment_name=str(method_cfg.get("environment_name") or ENVIRONMENT_NAME),
        wbits=int(args.omniquant_wbits or method_cfg.get("wbits", 4)),
        abits=int(args.omniquant_abits or method_cfg.get("abits", 16)),
        group_size=int(method_cfg["group_size"]) if method_cfg.get("group_size") is not None else 128,
        alpha=float(method_cfg.get("alpha", 0.5)),
        epochs=int(args.omniquant_epochs if args.omniquant_epochs is not None else method_cfg.get("epochs", 20)),
        calib_dataset=str(method_cfg.get("calib_dataset", "wikitext2")),
        nsamples=int(args.omniquant_nsamples if args.omniquant_nsamples is not None else method_cfg.get("nsamples", method_cfg.get("max_samples", 128))),
        batch_size=int(method_cfg.get("batch_size", 1)),
        seed=int(method_cfg.get("seed", (base_config.get("seeds") or {}).get("global", 1234))),
        tasks="",
        num_fewshot=0,
        limit=-1,
        attn_implementation=str(method_cfg.get("attn_implementation", "eager")),
        net=method_cfg.get("net"),
        resume=_resolve_path(root, method_cfg.get("resume")),
        act_scales=_resolve_path(root, method_cfg.get("act_scales")),
        act_shifts=_resolve_path(root, method_cfg.get("act_shifts")),
        eval_ppl=False,
        lwc=_parse_bool(method_cfg.get("lwc"), True),
        let=_parse_bool(method_cfg.get("let"), False),
        aug_loss=_parse_bool(method_cfg.get("aug_loss"), False),
        symmetric=_parse_bool(method_cfg.get("symmetric"), False),
        disable_zero_point=_parse_bool(method_cfg.get("disable_zero_point"), False),
        real_quant=bool(args.omniquant_real_quant or method_cfg.get("real_quant", False)),
        multigpu=_parse_bool(method_cfg.get("multigpu"), False),
        deactive_amp=_parse_bool(method_cfg.get("deactive_amp"), False),
        extra_args=_split_csv(method_cfg.get("extra_args")),
    )
    _write_json(
        method_dir / "requested_config.json",
        {
            "method": "omniquant",
            "model_name": model_name,
            "upstream_dir": str(upstream_dir),
            "python_executable": str(python_executable),
            "cache_dir": str(cache_dir),
            "save_dir": str(saved_model_dir),
            "raw_method_config": method_cfg,
        },
    )
    adapter_result = run_omniquant(request, dry_run=bool(args.omniquant_dry_run or method_cfg.get("dry_run", False)))
    _write_json(
        method_dir / "adapter_summary.json",
        {
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
    )
    return {
        "method": "omniquant",
        "status": "quantized" if (saved_model_dir / "config.json").exists() else "no_reloadable_model",
        "run_dir": str(method_dir),
        "model_path": str(saved_model_dir) if (saved_model_dir / "config.json").exists() else None,
    }


def _run_lm_eval_for_method(
    *,
    method_payload: Dict[str, Any],
    method_dir: Path,
    tasks: List[str],
    args: argparse.Namespace,
    device: str,
) -> Dict[str, Any]:
    config = _lm_eval_config(args, tasks, device)
    if not method_payload.get("model_path"):
        config["skip_reason"] = "method_did_not_produce_reloadable_model"
    summary = run_lm_eval_suite(
        model_name_or_path=method_payload.get("model_path"),
        tokenizer_name_or_path=method_payload.get("model_path"),
        out_dir=str(method_dir),
        config=config,
        device=device,
        dtype=args.dtype,
    )
    payload = {**method_payload, "lm_eval": summary}
    _write_json(method_dir / "summary.json", payload)
    return payload


def _row_from_result(model_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
    lm_eval = result.get("lm_eval") or {}
    row: Dict[str, Any] = {
        "model": model_name,
        "method": result.get("method"),
        "status": lm_eval.get("status") or result.get("status"),
        "reason": lm_eval.get("reason"),
        "tasks": ",".join(lm_eval.get("tasks") or []),
        "model_path": result.get("model_path"),
        "run_dir": result.get("run_dir"),
    }
    flat = lm_eval.get("flat") or {}
    for key, value in flat.items():
        row[key] = value
    return row


def _print_table(title: str, rows: List[Dict[str, Any]]) -> None:
    print(f"\n{title}")
    if not rows:
        print("(no results)")
        return
    metric_cols = [key for key in rows[0] if key.startswith("lm_eval__") and key not in {"lm_eval__tasks"}]
    columns = ["model", "method", "status"] + metric_cols[:6] + ["run_dir"]

    def fmt(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)

    rendered = [{col: fmt(row.get(col)) for col in columns} for row in rows]
    widths = {col: max(len(col), *(len(row[col]) for row in rendered)) for col in columns}
    print("  ".join(col.ljust(widths[col]) for col in columns))
    print("  ".join("-" * widths[col] for col in columns))
    for row in rendered:
        print("  ".join(row[col].ljust(widths[col]) for col in columns))


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    root = Path(".").resolve()
    models = _split_csv(args.models)
    methods = _validate_methods(_split_csv(args.methods))
    tasks = _split_csv(args.benchmarks)
    if not tasks:
        raise SystemExit("Provide at least one lm-eval task through --benchmarks")

    device = resolve_device(args.device)
    _ = resolve_dtype(args.dtype, device)
    base_config = _read_yaml(root / args.experiments_file)

    model_slug = sanitize_name("-".join(m.split("/")[-1] for m in models), max_len=96) or "models"
    method_slug = sanitize_name("-".join(methods), max_len=64) or "methods"
    task_slug = sanitize_name("-".join(tasks), max_len=96) or "tasks"
    run_root = Path(args.output_dir) / f"compare_real__{model_slug}__{method_slug}__lm-eval-{task_slug}__{now_timestamp()}"
    run_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_root / "metadata.json",
        {
            "models": models,
            "methods": methods,
            "lm_eval_tasks": tasks,
            "device": device,
            "dtype": args.dtype,
            "git_commit": get_git_commit(root),
            "versions": {
                "torch": get_pkg_version("torch"),
                "transformers": get_pkg_version("transformers"),
                "lm-eval": get_pkg_version("lm-eval"),
            },
        },
    )

    rows: List[Dict[str, Any]] = []
    for model_name in models:
        model_dir = run_root / sanitize_name(model_name.replace("/", "_"), max_len=96)
        model_dir.mkdir(parents=True, exist_ok=True)
        for method in methods:
            LOGGER.info("Running method=%s model=%s", method, model_name)
            method_dir = model_dir / method
            try:
                if method == "seq":
                    payload = _run_seq(
                        root=root,
                        model_name=model_name,
                        model_dir=model_dir,
                        base_config=base_config,
                        args=args,
                        device=device,
                    )
                elif method == "omniquant":
                    payload = _run_omniquant(
                        root=root,
                        model_name=model_name,
                        model_dir=model_dir,
                        base_config=base_config,
                        args=args,
                    )
                else:
                    raise RuntimeError(f"Unsupported method escaped validation: {method}")
                result = _run_lm_eval_for_method(
                    method_payload=payload,
                    method_dir=method_dir,
                    tasks=tasks,
                    args=args,
                    device=device,
                )
            except Exception as exc:
                LOGGER.exception("Method failed: %s / %s", model_name, method)
                result = {
                    "model": model_name,
                    "method": method,
                    "status": "error",
                    "reason": str(exc),
                    "run_dir": str(method_dir),
                    "lm_eval": {"status": "error", "reason": str(exc), "tasks": tasks, "flat": {}},
                }
                _write_json(method_dir / "summary.json", result)
            rows.append(_row_from_result(model_name, result))

    _write_json(run_root / "global_summary.json", {"rows": rows})
    _write_csv(run_root / "global_summary.csv", rows)
    _print_table("lm-eval Results", rows)
    LOGGER.info("Comparison complete: %s", run_root)


if __name__ == "__main__":
    main()
