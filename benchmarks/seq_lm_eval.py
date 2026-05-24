from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .eval_config import resolve_lm_eval_config, split_csv


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def _command_to_text(command: Optional[Any]) -> str:
    if not command:
        return ""
    if isinstance(command, str):
        return command
    return " ".join(str(part) for part in command)


def _write_outputs(
    lm_eval_dir: Path,
    *,
    raw: Optional[Dict[str, Any]],
    summary: Dict[str, Any],
    command: Optional[Any],
) -> Dict[str, Any]:
    lm_eval_dir.mkdir(parents=True, exist_ok=True)
    raw_payload = raw if isinstance(raw, dict) else summary
    _write_json(lm_eval_dir / "lm_eval_raw.json", raw_payload)
    _write_json(lm_eval_dir / "lm_eval_summary.json", summary)
    _write_json(
        lm_eval_dir / "lm_eval_status.json",
        {
            "status": summary.get("status"),
            "reason": summary.get("reason"),
            "requested": summary.get("requested", True),
            "tasks": summary.get("tasks", []),
        },
    )
    with (lm_eval_dir / "lm_eval_command.txt").open("w") as f:
        f.write(_command_to_text(command))
        if command:
            f.write("\n")
    return summary


def detect_lm_eval_cli(timeout_sec: int = 20) -> Optional[List[str]]:
    candidates: List[List[str]] = [[sys.executable, "-m", "lm_eval"]]
    lm_eval_exe = shutil.which("lm_eval")
    if lm_eval_exe:
        candidates.append([lm_eval_exe])
    lm_eval_dash = shutil.which("lm-eval")
    if lm_eval_dash:
        candidates.append([lm_eval_dash])
        candidates.append([lm_eval_dash, "run"])

    seen = set()
    unique_candidates: List[List[str]] = []
    for command in candidates:
        key = tuple(command)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(command)

    for command in unique_candidates:
        try:
            result = subprocess.run(
                command + ["--help"],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
        except Exception:
            continue
        if result.returncode == 0:
            return command
    return None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _normalize_metric_key(metric: str) -> str:
    return metric.split(",", 1)[0].strip()


def _extract_results(raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    results = raw.get("results", {})
    if not isinstance(results, dict):
        return {}
    normalized: Dict[str, Dict[str, Any]] = {}
    for task_name, task_payload in results.items():
        if not isinstance(task_payload, dict):
            continue
        task_metrics: Dict[str, Any] = {}
        for metric_name, value in task_payload.items():
            if not _is_number(value):
                continue
            key = _normalize_metric_key(str(metric_name))
            if not key:
                continue
            task_metrics[key] = value
        normalized[str(task_name)] = task_metrics
    return normalized


def flatten_lm_eval_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {
        "lm_eval__status": summary.get("status"),
        "lm_eval__reason": summary.get("reason"),
        "lm_eval__tasks": ",".join(summary.get("tasks", []) or []),
    }
    results = summary.get("results") or {}
    if not isinstance(results, dict):
        return flat
    for task_name, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        safe_task = str(task_name).replace("/", "_")
        for metric_name, value in metrics.items():
            if _is_number(value):
                safe_metric = str(metric_name).replace("/", "_")
                flat[f"lm_eval__{safe_task}__{safe_metric}"] = value
    return flat


def normalize_lm_eval_result(raw: Dict[str, Any], cfg: Dict[str, Any], command: Any) -> Dict[str, Any]:
    results = _extract_results(raw)
    summary: Dict[str, Any] = {
        "status": "ok",
        "backend": cfg.get("model_backend", "hf"),
        "lm_eval_source": cfg.get("lm_eval_source"),
        "tasks": list(cfg.get("tasks") or sorted(results.keys())),
        "num_fewshot": cfg.get("num_fewshot"),
        "limit": cfg.get("limit"),
        "results": results,
        "requested": True,
        "command": _command_to_text(command),
        "fail_policy": cfg.get("fail_policy", "warn"),
    }
    summary["flat"] = flatten_lm_eval_summary(summary)
    return summary


def _status_payload(
    status: str,
    reason: str,
    cfg: Dict[str, Any],
    *,
    command: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "status": status,
        "reason": reason,
        "backend": cfg.get("model_backend", "hf"),
        "lm_eval_source": cfg.get("lm_eval_source"),
        "tasks": list(cfg.get("tasks") or []),
        "num_fewshot": cfg.get("num_fewshot"),
        "limit": cfg.get("limit"),
        "requested": True,
        "command": _command_to_text(command),
        "results": {},
        "fail_policy": cfg.get("fail_policy", "warn"),
    }
    if extra:
        payload.update(extra)
    payload["flat"] = flatten_lm_eval_summary(payload)
    return payload


def _handle_failure(
    lm_eval_dir: Path,
    cfg: Dict[str, Any],
    reason: str,
    *,
    command: Optional[Any] = None,
    raw: Optional[Dict[str, Any]] = None,
    exc: Optional[BaseException] = None,
) -> Dict[str, Any]:
    fail_policy = str(cfg.get("fail_policy", "warn")).strip().lower()
    if fail_policy in {"error", "raise"}:
        if exc is not None:
            raise exc
        raise RuntimeError(reason)
    status = "skipped" if fail_policy == "skip" else "error"
    extra: Dict[str, Any] = {}
    if exc is not None:
        extra["traceback"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    summary = _status_payload(status, reason, cfg, command=command, extra=extra)
    return _write_outputs(lm_eval_dir, raw=raw, summary=summary, command=command)


def _handle_skip(
    lm_eval_dir: Path,
    cfg: Dict[str, Any],
    reason: str,
    *,
    command: Optional[Any] = None,
    raw: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    fail_policy = str(cfg.get("fail_policy", "warn")).strip().lower()
    if fail_policy in {"error", "raise"}:
        raise RuntimeError(reason)
    summary = _status_payload("skipped", reason, cfg, command=command)
    return _write_outputs(lm_eval_dir, raw=raw, summary=summary, command=command)


def _has_numeric_metrics(results: Dict[str, Dict[str, Any]]) -> bool:
    for metrics in results.values():
        if not isinstance(metrics, dict):
            continue
        for value in metrics.values():
            if _is_number(value):
                return True
    return False


def _model_args_string(
    model_name_or_path: str,
    tokenizer_name_or_path: Optional[str],
    dtype: Optional[str],
    extra: Dict[str, Any],
) -> str:
    args: Dict[str, Any] = {"pretrained": model_name_or_path}
    if tokenizer_name_or_path and tokenizer_name_or_path != model_name_or_path:
        args["tokenizer"] = tokenizer_name_or_path
    if dtype:
        args["dtype"] = dtype
    args.update(extra or {})

    parts: List[str] = []
    for key, value in args.items():
        if value is None:
            continue
        if isinstance(value, bool):
            value = "True" if value else "False"
        parts.append(f"{key}={value}")
    return ",".join(parts)


def _find_json_outputs(path: Path) -> List[Path]:
    if path.is_file():
        siblings = sorted(path.parent.glob(f"{path.stem}*.json"))
        ordered: List[Path] = []
        for candidate in siblings:
            if candidate not in ordered:
                ordered.append(candidate)
        if path not in ordered:
            ordered.append(path)
        return ordered
    if path.suffix.lower() == ".json":
        siblings = sorted(path.parent.glob(f"{path.stem}*.json"))
        if siblings:
            return siblings
    if not path.exists():
        return []
    return sorted(p for p in path.rglob("*.json") if p.is_file())


def _load_best_raw_json(output_path: Path) -> Dict[str, Any]:
    for candidate in _find_json_outputs(output_path):
        try:
            with candidate.open("r") as f:
                payload = json.load(f)
            if isinstance(payload, dict) and "results" in payload:
                return payload
        except Exception:
            continue
    for candidate in _find_json_outputs(output_path):
        try:
            with candidate.open("r") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                return payload
        except Exception:
            continue
    return {}


def _json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:
        if isinstance(value, dict):
            return {str(key): _json_safe(val) for key, val in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(item) for item in value]
        if hasattr(value, "item") and callable(getattr(value, "item")):
            try:
                return value.item()
            except Exception:
                return str(value)
        return str(value)


def _run_lm_eval_in_memory(
    *,
    model,
    tokenizer,
    lm_eval_dir: Path,
    cfg: Dict[str, Any],
    device: str,
    dtype: Optional[str],
    model_args_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    command = "in_memory_hflm"
    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
    except Exception:
        return _handle_failure(lm_eval_dir, cfg, "lm_eval_not_installed", command=command)

    task_list = split_csv(cfg.get("tasks")) or ["hellaswag"]
    effective_device = cfg.get("device") or device
    extra_model_args = dict(cfg.get("extra_model_args") or {})
    extra_model_args.update(model_args_extra or {})

    hflm_kwargs: Dict[str, Any] = {
        "pretrained": model,
        "tokenizer": tokenizer,
        "batch_size": cfg.get("batch_size", 1),
        "device": effective_device,
        "dtype": dtype,
    }
    for key, value in extra_model_args.items():
        if value is not None:
            hflm_kwargs[key] = value

    try:
        lm = HFLM(**hflm_kwargs)
        raw = evaluator.simple_evaluate(
            model=lm,
            tasks=task_list,
            num_fewshot=int(cfg.get("num_fewshot", 0)),
            limit=cfg.get("limit"),
            batch_size=cfg.get("batch_size", 1),
            device=effective_device,
            use_cache=cfg.get("use_cache"),
            log_samples=bool(cfg.get("log_samples", False)),
            apply_chat_template=cfg.get("apply_chat_template", False),
        )
        raw_payload = _json_safe(raw or {})
        extracted_results = _extract_results(raw_payload)
        if not _has_numeric_metrics(extracted_results):
            return _handle_failure(
                lm_eval_dir,
                cfg,
                "lm_eval_no_numeric_metrics",
                command=command,
                raw=raw_payload,
            )
        summary = normalize_lm_eval_result(raw_payload, {**cfg, "tasks": task_list}, command)
        return _write_outputs(lm_eval_dir, raw=raw_payload, summary=summary, command=command)
    except Exception as exc:
        return _handle_failure(lm_eval_dir, cfg, str(exc), command=command, exc=exc)


def run_lm_eval_suite(
    model_name_or_path: Optional[str],
    tokenizer_name_or_path: Optional[str],
    out_dir: str,
    config: Dict[str, Any],
    device: str,
    dtype: Optional[str] = None,
    model_args_extra: Optional[Dict[str, Any]] = None,
    model=None,
    tokenizer=None,
    lm_eval_source: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = resolve_lm_eval_config({"evaluation": {"lm_eval": config or {}}})
    output_subdir = str(cfg.get("output_subdir", "lm_eval"))
    lm_eval_dir = Path(out_dir) / output_subdir
    requested = bool(cfg.get("enabled", False))

    if not requested:
        summary = _status_payload("skipped", "disabled_by_config", cfg)
        summary["requested"] = False
        return _write_outputs(lm_eval_dir, raw=summary, summary=summary, command=None)

    if (model is None) != (tokenizer is None):
        raise ValueError("run_lm_eval_suite requires both model and tokenizer for in-memory evaluation")

    if model is not None and tokenizer is not None:
        cfg["lm_eval_source"] = lm_eval_source or "in_memory_hflm"
        skip_reason = cfg.get("skip_reason")
        if skip_reason:
            return _handle_skip(lm_eval_dir, cfg, str(skip_reason))
        return _run_lm_eval_in_memory(
            model=model,
            tokenizer=tokenizer,
            lm_eval_dir=lm_eval_dir,
            cfg=cfg,
            device=device,
            dtype=dtype,
            model_args_extra=model_args_extra,
        )

    cfg["lm_eval_source"] = lm_eval_source or "cli_pretrained_path"
    skip_reason = cfg.get("skip_reason")
    if skip_reason:
        return _handle_skip(lm_eval_dir, cfg, str(skip_reason))
    if not model_name_or_path:
        return _handle_skip(lm_eval_dir, cfg, "in_memory_quantized_model_not_reloadable")

    cli = detect_lm_eval_cli()
    if cli is None:
        return _handle_failure(lm_eval_dir, cfg, "lm_eval_not_installed")

    task_list = split_csv(cfg.get("tasks")) or ["hellaswag"]
    effective_device = cfg.get("device") or device
    raw_output_path = lm_eval_dir / "lm_eval_raw.json"
    extra_model_args = dict(cfg.get("extra_model_args") or {})
    extra_model_args.update(model_args_extra or {})

    model_args = _model_args_string(
        str(model_name_or_path),
        tokenizer_name_or_path,
        dtype,
        extra_model_args,
    )
    command: List[str] = list(cli) + [
        "--model",
        str(cfg.get("model_backend", "hf")),
        "--model_args",
        model_args,
        "--tasks",
        ",".join(task_list),
        "--device",
        str(effective_device),
        "--batch_size",
        str(cfg.get("batch_size", 1)),
        "--num_fewshot",
        str(cfg.get("num_fewshot", 0)),
        "--output_path",
        str(raw_output_path),
    ]
    if cfg.get("limit") is not None:
        command.extend(["--limit", str(cfg.get("limit"))])
    if cfg.get("log_samples"):
        command.append("--log_samples")
    if cfg.get("apply_chat_template"):
        command.append("--apply_chat_template")
    if cfg.get("use_cache"):
        command.extend(["--use_cache", str(cfg.get("use_cache"))])

    lm_eval_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        stdout_path = lm_eval_dir / "lm_eval_stdout.txt"
        stderr_path = lm_eval_dir / "lm_eval_stderr.txt"
        stdout_path.write_text(result.stdout or "")
        stderr_path.write_text(result.stderr or "")

        raw = _load_best_raw_json(raw_output_path)
        raw.setdefault("process", {})
        raw["process"].update(
            {
                "returncode": result.returncode,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
            }
        )
        if result.returncode != 0:
            reason = (result.stderr or result.stdout or f"lm_eval exited with {result.returncode}").strip()
            return _handle_failure(lm_eval_dir, cfg, reason, command=command, raw=raw)

        extracted_results = _extract_results(raw)
        if not _has_numeric_metrics(extracted_results):
            return _handle_failure(
                lm_eval_dir,
                cfg,
                "lm_eval_no_numeric_metrics",
                command=command,
                raw=raw,
            )

        summary = normalize_lm_eval_result(raw, {**cfg, "tasks": task_list}, command)
        return _write_outputs(lm_eval_dir, raw=raw, summary=summary, command=command)
    except Exception as exc:
        return _handle_failure(lm_eval_dir, cfg, str(exc), command=command, exc=exc)
