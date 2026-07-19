#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import os
import re
import shutil
import sys
import datetime as dt
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

from benchmarks.seq_lm_eval import run_lm_eval_suite
from benchmarks.core import compute_ppl
from third_party_quant.adapters.omniquant_adapter import (
    DEFAULT_UPSTREAM_DIR,
    ENVIRONMENT_NAME,
    OmniQuantRequest,
    run_omniquant,
)
from third_party_quant.llmc_compare import run_llmc_baseline


LOGGER = logging.getLogger("run_compare_matrix")
SEQ_METHODS = (
    "seq",
    "seq_v0",
    "seq_v1",
    "seq_v2",
    "seq_v3",
    "seq_v4",
    "seq_v5",
)
SEQ_VARIANT_DEFAULT_PATHS = {
    "seq": "seq_variants/SEQ-v0/variant_config.yaml",
    "seq_v0": "seq_variants/SEQ-v0/variant_config.yaml",
    "seq_v1": "seq_variants/SEQ-v1/variant_config.yaml",
    "seq_v2": "seq_variants/SEQ-v2/variant_config.yaml",
    "seq_v3": "seq_variants/SEQ-v3/variant_config.yaml",
    "seq_v4": "seq_variants/SEQ-v4/variant_config.yaml",
    "seq_v5": "seq_variants/SEQ-v5/variant_config.yaml",
}
SUPPORTED_METHODS = (
    "base",
    *SEQ_METHODS,
    "omniquant",
    "gptq_llmc",
    "smoothquant_llmc",
    "awq_llmc",
    "rtn_llmc",
    "llm_int8_llmc",
    "spinquant_llmc",
    "omniquant_llmc",
)
METHOD_ALIASES = {"omniqunat": "omniquant"}
_UNICODE_SPACE_RE = re.compile(r"[\u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]+")
_LONG_OPTS_WITH_VALUE = (
    "--models",
    "--methods",
    "--benchmarks",
    "--eval_tasks",
    "--device",
    "--dtype",
    "--experiments_file",
    "--output_dir",
    "--degeneracy_mode",
    "--lm_eval_limit",
    "--eval_limit",
    "--lm_eval_num_fewshot",
    "--num_fewshot",
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
    "--llmc_repo",
    "--llmc_venv",
    "--llmc_save_mode",
    "--llmc_calib_samples",
    "--llmc_calib_seq_len",
    "--llmc_eval_seq_len",
    "--llmc_eval_dataset",
    "--llmc_calib_dataset",
    "--llmc_model_type",
    "--llmc_tokenizer_mode",
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


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle) or {}


def get_pkg_version(pkg_name: str) -> Optional[str]:
    try:
        import importlib.metadata as metadata
        return metadata.version(pkg_name)
    except Exception:
        return None


def get_git_commit(root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None
    if result.returncode == 0:
        return result.stdout.strip()
    return None


def now_timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def sanitize_name(text: str, max_len: int = 48) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return safe[:max_len] if len(safe) > max_len else safe


class TimestampedTee:
    def __init__(self, stream: Any, log_handle: Any, stream_name: str) -> None:
        self._stream = stream
        self._log_handle = log_handle
        self._stream_name = stream_name
        self._buffer = ""
        self.encoding = getattr(stream, "encoding", "utf-8")
        self.errors = getattr(stream, "errors", "strict")

    def write(self, text: str) -> int:
        if not isinstance(text, str):
            text = str(text)
        if not text:
            return 0
        written = self._stream.write(text)
        self._buffer += text
        self._drain_buffer()
        return written if isinstance(written, int) else len(text)

    def flush(self) -> None:
        self._stream.flush()
        self._drain_buffer(final=True)
        self._log_handle.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._stream, "isatty", lambda: False)())

    def fileno(self) -> int:
        return int(self._stream.fileno())

    def writable(self) -> bool:
        return True

    def _drain_buffer(self, final: bool = False) -> None:
        while True:
            newline_index = self._buffer.find("\n")
            if newline_index < 0:
                break
            line = self._buffer[: newline_index + 1]
            self._buffer = self._buffer[newline_index + 1 :]
            self._write_log_line(line, force_newline=False)
        if final and self._buffer:
            line = self._buffer
            self._buffer = ""
            self._write_log_line(line, force_newline=True)

    def _write_log_line(self, line: str, force_newline: bool) -> None:
        timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        text = line.rstrip("\n")
        suffix = "\n" if force_newline or line.endswith("\n") else ""
        self._log_handle.write(f"{timestamp} | {self._stream_name} | {text}{suffix}")


def _install_terminal_log(log_dir: Path) -> Dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    terminal_log_path = log_dir / "terminal.log"
    log_handle = terminal_log_path.open("a", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    stdout_tee = TimestampedTee(original_stdout, log_handle, "stdout")
    stderr_tee = TimestampedTee(original_stderr, log_handle, "stderr")
    sys.stdout = stdout_tee
    sys.stderr = stderr_tee
    return {
        "terminal_log_path": terminal_log_path,
        "log_handle": log_handle,
        "original_stdout": original_stdout,
        "original_stderr": original_stderr,
    }


def _restore_terminal_log(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    current_stdout = sys.stdout
    current_stderr = sys.stderr
    if hasattr(current_stdout, "flush"):
        current_stdout.flush()
    if hasattr(current_stderr, "flush"):
        current_stderr.flush()
    sys.stdout = state["original_stdout"]
    sys.stderr = state["original_stderr"]
    state["log_handle"].close()


def _compare_model_slug(model_name: str) -> str:
    leaf = str(model_name).strip().split("/")[-1]
    for suffix in ("-hf", "_hf"):
        if leaf.lower().endswith(suffix):
            trimmed = leaf[: -len(suffix)].strip("._-")
            if trimmed:
                return trimmed
    return leaf


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def resolve_dtype(dtype_arg: str, device: str) -> torch.dtype:
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    dtype = dtype_map.get(str(dtype_arg).lower(), torch.float16)
    if device == "cpu" and dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _default_llmc_repo() -> Optional[str]:
    env_value = os.getenv("LLMC_REPO")
    if env_value:
        return env_value
    for candidate in ("/mnt/e/LightCompress", "E:\\LightCompress"):
        if Path(candidate).exists():
            return candidate
    return None


def _default_llmc_venv(llmc_repo: Optional[str]) -> Optional[str]:
    env_value = os.getenv("LLMC_VENV")
    if env_value:
        return env_value
    if not llmc_repo:
        return None
    repo = Path(llmc_repo)
    for candidate in (repo / ".venv-llmc",):
        if candidate.exists():
            return str(candidate)
    if re.match(r"^/mnt/[a-zA-Z]/", llmc_repo):
        return llmc_repo.rstrip("/") + "/.venv-llmc"
    return str(repo / ".venv-llmc")


def _is_ppl_only_benchmark(tasks: List[str]) -> bool:
    normalized = [task.strip().lower() for task in tasks if task.strip()]
    return normalized == ["ppl"]


def _resolve_benchmark_tasks(tasks: List[str]) -> Dict[str, Any]:
    ordered: List[str] = []
    seen = set()
    ppl_requested = False
    lm_eval_tasks: List[str] = []
    for raw_task in tasks:
        task = str(raw_task).strip()
        if not task:
            continue
        lowered = task.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        ordered.append(task)
        if lowered == "ppl":
            ppl_requested = True
        else:
            lm_eval_tasks.append(task)
    return {
        "all_tasks": ordered,
        "ppl_requested": ppl_requested,
        "lm_eval_tasks": lm_eval_tasks,
        "ppl_only": ppl_requested and not lm_eval_tasks,
        "lm_eval_only": bool(lm_eval_tasks) and not ppl_requested,
        "mixed": ppl_requested and bool(lm_eval_tasks),
    }


def _merge_notes(*note_lists: Optional[List[str]]) -> List[str]:
    merged: List[str] = []
    for note_list in note_lists:
        for note in note_list or []:
            text = str(note)
            if text not in merged:
                merged.append(text)
    return merged


def _local_model_dir_is_reloadable(model_path: Optional[str]) -> bool:
    if not model_path:
        return False
    path = Path(str(model_path))
    if not path.exists() or not path.is_dir():
        return False
    tokenizer_markers = (
        path / "tokenizer.json",
        path / "tokenizer_config.json",
        path / "vocab.json",
        path / "merges.txt",
    )
    weight_markers = (
        path / "model.safetensors",
        path / "model.safetensors.index.json",
        path / "pytorch_model.bin",
        path / "pytorch_model.bin.index.json",
    )
    return (path / "config.json").exists() and any(marker.exists() for marker in tokenizer_markers) and any(
        marker.exists() for marker in weight_markers
    )


def _lm_eval_skip_reason_for_payload(method_payload: Dict[str, Any], explicit_skip_reason: Optional[str] = None) -> Optional[str]:
    if explicit_skip_reason:
        return explicit_skip_reason
    model_path = method_payload.get("model_path")
    if not model_path:
        return "method_did_not_produce_reloadable_model"
    path = Path(str(model_path))
    if path.exists() and not _local_model_dir_is_reloadable(str(path)):
        return "method_did_not_produce_reloadable_model"
    return None


def _base_lm_eval_payload(model_name: str, method_dir: Path) -> Dict[str, Any]:
    return {
        "model": model_name,
        "method": "base",
        "status": "base_model",
        "run_dir": str(method_dir),
        "model_path": model_name,
        "notes": ["lm-eval on original Hugging Face model"],
    }


def _merge_ppl_result(method_payload: Dict[str, Any], ppl_result: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(method_payload)
    for key in (
        "benchmark",
        "compare_status",
        "reason",
        "ppl",
        "ppl_source",
        "ppl_dataset",
        "ppl_seq_len",
        "duration_sec",
    ):
        if key in ppl_result:
            merged[key] = ppl_result.get(key)
    if "model" in ppl_result:
        merged["model"] = ppl_result.get("model")
    if "method" in ppl_result:
        merged["method"] = ppl_result.get("method")
    merged["run_dir"] = method_payload.get("run_dir") or ppl_result.get("run_dir")
    merged["notes"] = _merge_notes(method_payload.get("notes"), ppl_result.get("notes"))
    if "lm_eval" in method_payload:
        merged["lm_eval"] = method_payload.get("lm_eval")
    return merged


def _finalize_method_result(
    payload: Dict[str, Any],
    *,
    ppl_requested: bool,
    lm_eval_tasks: List[str],
) -> Dict[str, Any]:
    lm_eval_requested = bool(lm_eval_tasks)
    if ppl_requested and lm_eval_requested:
        payload["benchmark"] = "mixed"
    elif ppl_requested:
        payload["benchmark"] = "ppl"
    elif lm_eval_requested:
        payload["benchmark"] = "lm_eval"

    if lm_eval_requested:
        lm_status = ((payload.get("lm_eval") or {}).get("status") or "").strip().lower()
        base_status = str(payload.get("compare_status") or payload.get("status") or "").strip().lower()
        if lm_status == "ok":
            if ppl_requested and base_status and base_status not in {"success", "ok", "quantized", "base_model"}:
                payload["compare_status"] = payload.get("compare_status") or payload.get("status")
            else:
                payload["compare_status"] = "success"
        elif lm_status:
            payload["compare_status"] = lm_status
    elif ppl_requested and not payload.get("compare_status"):
        payload["compare_status"] = payload.get("status")
    return payload


def _requested_tasks_label(result: Dict[str, Any]) -> str:
    labels: List[str] = []
    benchmark = str(result.get("benchmark") or "").strip().lower()
    if benchmark in {"ppl", "mixed"}:
        labels.append("ppl")
    for task in (result.get("lm_eval") or {}).get("tasks") or []:
        task_text = str(task)
        if task_text and task_text not in labels:
            labels.append(task_text)
    return ",".join(labels)


def _should_abort_for_lm_eval_failure(args: argparse.Namespace, result: Dict[str, Any], lm_eval_tasks: List[str]) -> bool:
    if not lm_eval_tasks:
        return False
    fail_policy = str(getattr(args, "lm_eval_fail_policy", "warn")).strip().lower()
    if fail_policy not in {"raise", "error"}:
        return False
    lm_status = str(((result.get("lm_eval") or {}).get("status") or "")).strip().lower()
    return lm_status in {"error", "skipped"}


def _resolve_ppl_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    bench_config = dict(base_config.get("benchmarks") or {})
    evaluation_cfg = dict(base_config.get("evaluation") or {})
    seq_metrics_cfg = dict(evaluation_cfg.get("seq_metrics") or {})
    ppl_metric_cfg = dict(seq_metrics_cfg.get("ppl") or {})

    ppl_mode = str(ppl_metric_cfg.get("mode", bench_config.get("ppl_mode", "proxy"))).strip().lower()
    ppl_split = ppl_metric_cfg.get(
        "split",
        bench_config.get("ppl_split", "test" if ppl_mode == "canonical" else "validation"),
    )
    ppl_seq_len = int(
        ppl_metric_cfg.get(
            "seq_len",
            bench_config.get("ppl_seq_len", 2048 if ppl_mode == "canonical" else 256),
        )
    )
    ppl_stride = ppl_metric_cfg.get("stride", bench_config.get("ppl_stride"))
    ppl_max_examples = ppl_metric_cfg.get(
        "max_examples",
        bench_config.get("ppl_max_examples", None if ppl_mode == "canonical" else 128),
    )
    ppl_full_corpus = bool(
        ppl_metric_cfg.get(
            "full_corpus",
            bench_config.get("ppl_full_corpus", ppl_mode == "canonical"),
        )
    )
    if ppl_mode == "canonical":
        ppl_stride = None
        ppl_max_examples = None
        ppl_full_corpus = True
        ppl_split = "test"
    return {
        "ppl_mode": ppl_mode,
        "ppl_dataset": str(ppl_metric_cfg.get("dataset", bench_config.get("ppl_dataset", "wikitext2"))),
        "ppl_split": str(ppl_split),
        "ppl_seq_len": int(ppl_seq_len),
        "ppl_stride": ppl_stride,
        "ppl_max_examples": ppl_max_examples,
        "ppl_full_corpus": ppl_full_corpus,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-method comparison: SEQ and upstream OmniQuant.")
    parser.add_argument("--models", required=True, help="Comma-separated Hugging Face model names or local model paths.")
    parser.add_argument("--methods", default="seq,omniquant", help="Comma-separated methods. Supported: base,seq,seq_v0,seq_v1,seq_v2,seq_v3,seq_v4,seq_v5,omniquant,gptq_llmc,smoothquant_llmc,awq_llmc,rtn_llmc,llm_int8_llmc,spinquant_llmc,omniquant_llmc.")
    parser.add_argument(
        "--benchmarks",
        "--eval_tasks",
        dest="benchmarks",
        default="hellaswag",
        help="Comma-separated benchmark names. Use ppl for perplexity or lm-eval tasks such as hellaswag,arc_easy,piqa.",
    )
    parser.add_argument("--device", default="auto", help="auto|cuda|cpu or lm-eval-compatible device string.")
    parser.add_argument("--dtype", default="float16", help="float16|bfloat16|float32.")
    parser.add_argument("--experiments_file", default="experiments.yaml")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--degeneracy_mode", choices=["old", "rms"], default="rms")
    parser.add_argument("--trust_remote_code", action="store_true")

    parser.add_argument("--lm_eval_limit", "--eval_limit", dest="lm_eval_limit", type=int, default=None)
    parser.add_argument("--lm_eval_num_fewshot", "--num_fewshot", dest="lm_eval_num_fewshot", type=int, default=0)
    parser.add_argument("--lm_eval_batch_size", default="1")
    parser.add_argument("--lm_eval_model_backend", default="hf")
    parser.add_argument("--lm_eval_fail_policy", choices=["warn", "raise", "error", "skip"], default="warn")
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
    parser.add_argument("--llmc_repo", default=_default_llmc_repo())
    parser.add_argument("--llmc_venv", default=None)
    parser.add_argument("--llmc_dry_run", action="store_true")
    parser.add_argument("--llmc_save_mode", choices=["none", "fake", "trans"], default="none")
    parser.add_argument("--llmc_calib_samples", type=int, default=4)
    parser.add_argument("--llmc_calib_seq_len", type=int, default=128)
    parser.add_argument("--llmc_eval_seq_len", type=int, default=128)
    parser.add_argument("--llmc_eval_dataset", default="wikitext2")
    parser.add_argument("--llmc_calib_dataset", default="wikitext2")
    parser.add_argument("--llmc_model_type", default=None)
    parser.add_argument("--llmc_tokenizer_mode", default="slow")
    parser.add_argument("--llmc_no_inference_per_block", action="store_true")

    args, unknown = parser.parse_known_args(_normalize_cli_argv(sys.argv[1:]))
    if unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    if args.llmc_venv is None:
        args.llmc_venv = _default_llmc_venv(args.llmc_repo)
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


def is_seq_method(method: str) -> bool:
    return method in SEQ_METHODS


def seq_variant_config_path(method: str, base_config: Dict[str, Any], root: Path) -> Path:
    method_cfg = dict((base_config.get("compare_methods") or {}).get(method) or {})
    configured = method_cfg.get("variant_config") or SEQ_VARIANT_DEFAULT_PATHS.get(method)
    if not configured:
        raise KeyError(f"No SEQ variant config registered for method {method}")
    path = Path(str(configured))
    return path if path.is_absolute() else root / path


def load_seq_variant_config(method: str, base_config: Dict[str, Any], root: Path) -> Dict[str, Any]:
    path = seq_variant_config_path(method, base_config, root)
    if not path.exists():
        raise FileNotFoundError(f"Missing SEQ variant config for {method}: {path}")
    with path.open("r") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"SEQ variant config must be a mapping: {path}")
    loaded = dict(loaded)
    loaded["method_name"] = method
    loaded["variant_config_path"] = str(path)
    return loaded


def _apply_seq_variant_config(config: Dict[str, Any], variant_cfg: Dict[str, Any]) -> Dict[str, Any]:
    config = copy.deepcopy(config)
    config["seq_variant"] = dict(variant_cfg)

    config.setdefault("thresholds", {})
    if "weight_high_pct" in variant_cfg:
        config["thresholds"]["weight_high_pct"] = float(variant_cfg["weight_high_pct"])
    if "act_high_pct" in variant_cfg:
        config["thresholds"]["act_high_pct"] = float(variant_cfg["act_high_pct"])

    config.setdefault("protections", {})
    protection_map = {
        "protect_first_last_blocks": "protect_first_last_blocks",
        "protect_all_attention_proj": "protect_all_attention_proj",
        "protect_gate_down_proj": "protect_gate_down_proj",
        "protect_unreliable": "protect_unreliable",
    }
    for src, dest in protection_map.items():
        if src in variant_cfg:
            config["protections"][dest] = bool(variant_cfg[src])
    if "protect_attention_out_proj" in variant_cfg:
        value = bool(variant_cfg["protect_attention_out_proj"])
        config["protections"]["protect_attn_out_proj"] = value
        config["protections"]["protect_attention_out_proj"] = value
    if "protect_lm_head" in variant_cfg and not bool(variant_cfg["protect_lm_head"]):
        config["protections"]["lm_head_min_tier"] = "int4"
    elif variant_cfg.get("protect_lm_head", True):
        config["protections"]["lm_head_min_tier"] = "fp16"

    config["quantization"] = dict(config.get("quantization") or {})
    for key in ("bnb_4bit_quant_type", "bnb_4bit_use_double_quant", "bnb_compute_dtype"):
        if key in variant_cfg:
            config["quantization"][key] = variant_cfg[key]

    return config


def _seq_experiment(method: str, variant_cfg: Dict[str, Any]) -> Dict[str, Any]:
    tier_policy = str(variant_cfg.get("tier_policy", "hard_and_xor")).strip().lower()
    policy = "risk_score" if tier_policy == "risk_score" else "dual_entropy"
    return {
        "name": f"compare_{method}",
        "policy": policy,
        "seq_variant": variant_cfg.get("variant_name", method),
    }


def _copy_if_exists(src: Path, dest: Path) -> None:
    if src.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)


def _materialize_seq_artifacts(method_dir: Path, seq_run: Path) -> None:
    _copy_if_exists(seq_run / "policy" / "precision_map.json", method_dir / "precision_map.json")
    _copy_if_exists(seq_run / "policy" / "precision_table.csv", method_dir / "precision_table.csv")
    _copy_if_exists(seq_run / "entropy" / "thresholds.json", method_dir / "thresholds.json")
    _copy_if_exists(seq_run / "policy" / "override_summary.json", method_dir / "override_summary.json")
    _copy_if_exists(seq_run / "quant" / "replacement_counts.json", method_dir / "replacement_counts.json")
    _copy_if_exists(seq_run / "quant" / "effective_bits.json", method_dir / "effective_bits.json")
    _copy_if_exists(seq_run / "quant" / "verification.json", method_dir / "verification.json")
    _copy_if_exists(seq_run / "variant_config.yaml", method_dir / "variant_config.yaml")


def _seq_pipeline_lm_eval_summary(seq_run_dir: Path) -> Optional[Dict[str, Any]]:
    eval_quant = _read_json(seq_run_dir / "eval_quant" / "eval_summary.json")
    lm_eval = eval_quant.get("lm_eval")
    return dict(lm_eval) if isinstance(lm_eval, dict) else None


def _missing_in_memory_lm_eval_summary(tasks: List[str], args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "status": "skipped",
        "reason": "in_memory_lm_eval_summary_missing",
        "backend": args.lm_eval_model_backend,
        "lm_eval_source": "in_memory_hflm",
        "tasks": list(tasks),
        "num_fewshot": int(args.lm_eval_num_fewshot),
        "limit": args.lm_eval_limit,
        "requested": True,
        "command": "",
        "results": {},
        "fail_policy": args.lm_eval_fail_policy,
        "flat": {
            "lm_eval__status": "skipped",
            "lm_eval__reason": "in_memory_lm_eval_summary_missing",
            "lm_eval__tasks": ",".join(tasks),
        },
    }


def _seq_pipeline_ppl_result(seq_run_dir: Path, duration_sec: Optional[float] = None) -> Optional[Dict[str, Any]]:
    eval_quant = _read_json(seq_run_dir / "eval_quant" / "eval_summary.json")
    perplexity = dict(eval_quant.get("perplexity") or {})
    if not perplexity:
        return None
    ppl_status = perplexity.get("status")
    return {
        "benchmark": "ppl",
        "status": "success" if perplexity.get("ppl") is not None and ppl_status != "error" else "failed",
        "compare_status": "success" if perplexity.get("ppl") is not None and ppl_status != "error" else "failed",
        "reason": perplexity.get("error"),
        "ppl": perplexity.get("ppl"),
        "ppl_source": "seq_ppl",
        "ppl_dataset": perplexity.get("dataset_name"),
        "ppl_seq_len": perplexity.get("seq_len"),
        "duration_sec": duration_sec,
        "notes": ["PPL from SEQ evaluator"],
    }


def _make_seq_config(
    base_config: Dict[str, Any],
    *,
    model_name: str,
    device: str,
    dtype: str,
    degeneracy_mode: str,
    trust_remote_code: bool,
    ppl_only: bool,
    ppl_requested: bool = False,
    lm_eval_config: Optional[Dict[str, Any]] = None,
    variant_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    config = copy.deepcopy(base_config)
    if variant_config:
        config = _apply_seq_variant_config(config, variant_config)
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

    config.setdefault("benchmarks", {})
    config.setdefault("evaluation", {})
    seq_metrics = dict(config["evaluation"].get("seq_metrics") or {})
    if ppl_only:
        config["benchmarks"]["run_ppl"] = True
        config["benchmarks"]["run_size"] = False
        for name in ("ppl", "tail_risk", "json_stress", "temperature_sweep", "long_context", "latency_memory", "size"):
            block = dict(seq_metrics.get(name) or {})
            block["enabled"] = name == "ppl"
            seq_metrics[name] = block
        config["evaluation"]["enabled_metric_groups"] = ["seq_core"]
        config["evaluation"]["lm_eval"] = {**(config["evaluation"].get("lm_eval") or {}), "enabled": False}
    else:
        # The comparison benchmark is lm-eval only. SEQ still performs the real
        # quantization pipeline. For mixed benchmarks, keep PPL enabled and let
        # the SEQ pipeline evaluate the live quantized model in memory.
        config["benchmarks"]["run_ppl"] = bool(ppl_requested)
        config["benchmarks"]["run_size"] = True
        for name in ("ppl", "tail_risk", "json_stress", "temperature_sweep", "long_context", "latency_memory"):
            block = dict(seq_metrics.get(name) or {})
            block["enabled"] = name == "ppl" and bool(ppl_requested)
            seq_metrics[name] = block
        if lm_eval_config:
            config["evaluation"]["lm_eval"] = {**(config["evaluation"].get("lm_eval") or {}), **lm_eval_config, "enabled": True}
    config["evaluation"]["seq_metrics"] = seq_metrics
    config["evaluation"]["mmlu"] = {**(config["evaluation"].get("mmlu") or {}), "enabled": False}
    config["evaluation"]["zero_shot"] = {**(config["evaluation"].get("zero_shot") or {}), "enabled": False}
    return config


def _find_new_run(before: set[Path], runs_root: Path) -> Path:
    after = {p for p in runs_root.iterdir() if p.is_dir()}
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
    method: str = "seq",
    ppl_only: bool = False,
    ppl_requested: bool = False,
    lm_eval_tasks: Optional[List[str]] = None,
) -> Dict[str, Any]:
    from seq_core.pipeline import run_experiment

    method_dir = model_dir / method
    variant_cfg = load_seq_variant_config(method, base_config, root)
    runs_root = method_dir / "seq_runs"
    reports_root = method_dir / "seq_reports"
    runs_root.mkdir(parents=True, exist_ok=True)
    before = set(runs_root.glob("run_*"))
    seq_lm_eval_cfg = _lm_eval_config(args, list(lm_eval_tasks or []), device) if lm_eval_tasks else None
    seq_config = _make_seq_config(
        base_config,
        model_name=model_name,
        device=device,
        dtype=args.dtype,
        degeneracy_mode=args.degeneracy_mode,
        trust_remote_code=bool(args.trust_remote_code),
        ppl_only=ppl_only,
        ppl_requested=ppl_requested,
        lm_eval_config=seq_lm_eval_cfg,
        variant_config=variant_cfg,
    )
    _write_json(method_dir / "seq_compare_config.json", seq_config)
    start = time.perf_counter()
    run_experiment(
        root,
        seq_config,
        _seq_experiment(method, variant_cfg),
        root / "calibration_prompts.json",
        root / "eval_prompts.json",
        reports_root,
        runs_root,
    )
    duration_sec = time.perf_counter() - start
    seq_run = _find_new_run(before, runs_root)
    _materialize_seq_artifacts(method_dir, seq_run)
    quant_model_dir = seq_run / "model_quantized"
    payload = {
        "method": method,
        "seq_variant": variant_cfg.get("variant_name"),
        "status": "quantized" if _local_model_dir_is_reloadable(str(quant_model_dir)) else "no_reloadable_model",
        "run_dir": str(method_dir),
        "seq_run_dir": str(seq_run),
        "model_path": str(quant_model_dir) if _local_model_dir_is_reloadable(str(quant_model_dir)) else None,
    }
    if not ppl_only:
        if ppl_requested:
            ppl_result = _seq_pipeline_ppl_result(seq_run, duration_sec=duration_sec)
            if ppl_result:
                payload = _merge_ppl_result(payload, ppl_result)
        if lm_eval_tasks:
            lm_eval_summary = _seq_pipeline_lm_eval_summary(seq_run)
            if lm_eval_summary:
                payload["lm_eval"] = lm_eval_summary
        return payload

    ppl_result = _seq_pipeline_ppl_result(seq_run, duration_sec=duration_sec)
    if ppl_result:
        payload.update(ppl_result)
    _write_json(method_dir / "summary.json", payload)
    return payload


def _run_ppl_eval(
    *,
    model_name_or_path: str,
    summary_model_name: str,
    method: str,
    method_dir: Path,
    base_config: Dict[str, Any],
    args: argparse.Namespace,
    device: str,
    notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    method_dir.mkdir(parents=True, exist_ok=True)
    ppl_cfg = _resolve_ppl_config(base_config)
    start = time.perf_counter()
    model = None
    tokenizer = None
    unload_model = None
    try:
        from seq_core.pipeline import load_model_and_tokenizer, set_seed, unload_model as unload_model_fn

        unload_model = unload_model_fn
        dtype = resolve_dtype(args.dtype, device)
        seed = int((base_config.get("seeds") or {}).get("eval", 1234))
        set_seed(seed)
        model, tokenizer = load_model_and_tokenizer(
            model_name_or_path,
            device,
            dtype,
            trust_remote_code=bool(args.trust_remote_code),
        )
        ppl_info = compute_ppl(
            model,
            tokenizer,
            dataset_name=str(ppl_cfg["ppl_dataset"]),
            split=str(ppl_cfg["ppl_split"]),
            seq_len=int(ppl_cfg["ppl_seq_len"]),
            max_examples=ppl_cfg["ppl_max_examples"],
            stride=ppl_cfg["ppl_stride"],
            device=device,
            dtype=dtype,
            seed=seed,
            mode=str(ppl_cfg["ppl_mode"]),
            full_corpus=bool(ppl_cfg["ppl_full_corpus"]),
        )
        payload = {
            "model": summary_model_name,
            "method": method,
            "benchmark": "ppl",
            "status": "success" if ppl_info.get("ppl") is not None and not ppl_info.get("error") else "failed",
            "compare_status": "success" if ppl_info.get("ppl") is not None and not ppl_info.get("error") else "failed",
            "reason": ppl_info.get("error"),
            "tasks": "ppl",
            "run_dir": str(method_dir),
            "model_path": model_name_or_path,
            "ppl": ppl_info.get("ppl"),
            "ppl_source": "seq_ppl",
            "ppl_dataset": ppl_info.get("dataset_name"),
            "ppl_seq_len": ppl_info.get("seq_len"),
            "duration_sec": time.perf_counter() - start,
            "notes": list(notes or ["PPL from SEQ evaluator"]),
        }
    except Exception as exc:
        payload = {
            "model": summary_model_name,
            "method": method,
            "benchmark": "ppl",
            "status": "failed",
            "compare_status": "failed",
            "reason": str(exc),
            "tasks": "ppl",
            "run_dir": str(method_dir),
            "model_path": model_name_or_path,
            "ppl": None,
            "ppl_source": "seq_ppl",
            "ppl_dataset": ppl_cfg.get("ppl_dataset"),
            "ppl_seq_len": ppl_cfg.get("ppl_seq_len"),
            "duration_sec": time.perf_counter() - start,
            "notes": list(notes or ["PPL from SEQ evaluator"]),
        }
    finally:
        if (model is not None or tokenizer is not None) and callable(unload_model):
            unload_model(model, tokenizer)

    _write_json(method_dir / "summary.json", payload)
    return payload


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
        "status": "quantized" if _local_model_dir_is_reloadable(str(saved_model_dir)) else "no_reloadable_model",
        "run_dir": str(method_dir),
        "model_path": str(saved_model_dir) if _local_model_dir_is_reloadable(str(saved_model_dir)) else None,
    }


def _run_llmc_method(
    *,
    model_name: str,
    method_dir: Path,
    method: str,
    args: argparse.Namespace,
    tasks: List[str],
) -> Dict[str, Any]:
    if not args.llmc_repo:
        raise RuntimeError("LLMC repo not found. Pass --llmc_repo /path/to/LightCompress.")
    return run_llmc_baseline(
        method=method,
        model=model_name,
        run_dir=method_dir,
        llmc_repo=Path(args.llmc_repo),
        llmc_venv=Path(args.llmc_venv) if args.llmc_venv else None,
        dry_run=bool(args.llmc_dry_run),
        save_mode=args.llmc_save_mode,
        calib_samples=int(args.llmc_calib_samples),
        calib_seq_len=int(args.llmc_calib_seq_len),
        eval_seq_len=int(args.llmc_eval_seq_len),
        eval_dataset=str(args.llmc_eval_dataset),
        calib_dataset=str(args.llmc_calib_dataset),
        seed=42,
        model_type=args.llmc_model_type,
        tokenizer_mode=str(args.llmc_tokenizer_mode),
        inference_per_block=not bool(args.llmc_no_inference_per_block),
        lm_eval_tasks=tasks,
    )


def _run_lm_eval_for_method(
    *,
    method_payload: Dict[str, Any],
    method_dir: Path,
    tasks: List[str],
    args: argparse.Namespace,
    device: str,
    skip_reason: Optional[str] = None,
) -> Dict[str, Any]:
    config = _lm_eval_config(args, tasks, device)
    resolved_skip_reason = _lm_eval_skip_reason_for_payload(method_payload, explicit_skip_reason=skip_reason)
    if resolved_skip_reason:
        config["skip_reason"] = resolved_skip_reason
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
    reason = result.get("reason")
    if not reason and lm_eval.get("status") in {"error", "skipped"}:
        reason = lm_eval.get("reason")
    row: Dict[str, Any] = {
        "model": model_name,
        "method": result.get("method"),
        "status": result.get("compare_status") or lm_eval.get("status") or result.get("status"),
        "reason": reason,
        "tasks": _requested_tasks_label(result),
        "lm_eval_source": lm_eval.get("lm_eval_source"),
        "model_path": result.get("model_path"),
        "run_dir": result.get("run_dir"),
    }
    for key in (
        "ppl",
        "ppl_source",
        "ppl_dataset",
        "ppl_seq_len",
        "duration_sec",
        "quant_disk_bytes",
        "quant_disk_gb",
        "artifact_kind",
        "artifact_path",
        "backend",
        "llmc_config_path",
        "llmc_log_path",
        "llmc_save_path",
        "llmc_task_id",
        "llmc_returncode",
        "llmc_repo",
        "llmc_commit",
        "llmc_venv",
        "dry_run",
    ):
        if key in result:
            row[key] = result.get(key)
    notes = result.get("notes")
    if notes:
        row["notes"] = "; ".join(str(note) for note in notes)
    flat = lm_eval.get("flat") or {}
    for key, value in flat.items():
        row[key] = value
    return row


def _print_table(title: str, rows: List[Dict[str, Any]]) -> None:
    print(f"\n{title}")
    if not rows:
        print("(no results)")
        return
    metric_cols: List[str] = []
    for row in rows:
        for key in row:
            if key.startswith("lm_eval__") and key not in {"lm_eval__tasks"} and key not in metric_cols:
                metric_cols.append(key)
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

    root = Path(".").resolve()
    models = _split_csv(args.models)
    methods = _validate_methods(_split_csv(args.methods))
    benchmark_request = _resolve_benchmark_tasks(_split_csv(args.benchmarks))
    tasks = benchmark_request["all_tasks"]
    if not tasks:
        raise SystemExit("Provide at least one benchmark through --benchmarks")
    ppl_requested = bool(benchmark_request["ppl_requested"])
    lm_eval_tasks = list(benchmark_request["lm_eval_tasks"])
    ppl_only = bool(benchmark_request["ppl_only"])
    benchmark_mode = "mixed" if benchmark_request["mixed"] else ("ppl" if ppl_only else "lm_eval")

    device = resolve_device(args.device)
    _ = resolve_dtype(args.dtype, device)
    base_config = _read_yaml(root / args.experiments_file)

    model_slug = sanitize_name("-".join(_compare_model_slug(m) for m in models), max_len=96) or "models"
    method_slug = sanitize_name("-".join(methods), max_len=64) or "methods"
    task_slug = sanitize_name("-".join(tasks), max_len=96) or "tasks"
    benchmark_slug = "ppl" if ppl_only else (f"mixed-{task_slug}" if benchmark_request["mixed"] else f"lm-eval-{task_slug}")
    run_timestamp = now_timestamp()
    run_root = Path(args.output_dir) / f"compare_real__{model_slug}__{method_slug}__{benchmark_slug}__{run_timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)
    log_state = _install_terminal_log(run_root / "logs")
    terminal_log_path = Path(log_state["terminal_log_path"])
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s", handlers=[logging.StreamHandler(sys.stdout)], force=True)
    _write_json(
        run_root / "metadata.json",
        {
            "logs_dir": str(run_root / "logs"),
            "terminal_log_path": str(terminal_log_path),
            "run_timestamp": run_timestamp,
            "models": models,
            "methods": methods,
            "benchmarks": tasks,
            "benchmark_mode": benchmark_mode,
            "lm_eval_tasks": lm_eval_tasks,
            "device": device,
            "dtype": args.dtype,
            "llmc": {
                "repo": args.llmc_repo,
                "venv": args.llmc_venv,
                "dry_run": bool(args.llmc_dry_run),
                "save_mode": args.llmc_save_mode,
                "calib_samples": args.llmc_calib_samples,
                "calib_seq_len": args.llmc_calib_seq_len,
                "eval_seq_len": args.llmc_eval_seq_len,
                "calib_dataset": args.llmc_calib_dataset,
                "eval_dataset": args.llmc_eval_dataset,
                "model_type_override": args.llmc_model_type,
                "tokenizer_mode": args.llmc_tokenizer_mode,
                "inference_per_block": not bool(args.llmc_no_inference_per_block),
            },
            "git_commit": get_git_commit(root),
            "versions": {
                "torch": get_pkg_version("torch"),
                "transformers": get_pkg_version("transformers"),
                "lm-eval": get_pkg_version("lm-eval"),
            },
        },
    )
    try:
        LOGGER.info("Terminal log: %s", terminal_log_path)
        rows: List[Dict[str, Any]] = []
        for model_name in models:
            model_dir = run_root / sanitize_name(model_name.replace("/", "_"), max_len=96)
            model_dir.mkdir(parents=True, exist_ok=True)
            for method in methods:
                LOGGER.info("Running method=%s model=%s", method, model_name)
                method_dir = model_dir / method
                try:
                    if method == "base":
                        ppl_result = None
                        if ppl_requested:
                            ppl_result = _run_ppl_eval(
                                model_name_or_path=model_name,
                                summary_model_name=model_name,
                                method="base",
                                method_dir=method_dir,
                                base_config=base_config,
                                args=args,
                                device=device,
                                notes=["PPL from SEQ evaluator"],
                            )
                        if lm_eval_tasks:
                            result = _run_lm_eval_for_method(
                                method_payload=ppl_result or _base_lm_eval_payload(model_name, method_dir),
                                method_dir=method_dir,
                                tasks=lm_eval_tasks,
                                args=args,
                                device=device,
                            )
                            result = _finalize_method_result(result, ppl_requested=ppl_requested, lm_eval_tasks=lm_eval_tasks)
                            _write_json(method_dir / "summary.json", result)
                        else:
                            result = ppl_result
                    elif is_seq_method(method):
                        if ppl_only:
                            result = _run_seq(
                                root=root,
                                model_name=model_name,
                                model_dir=model_dir,
                                base_config=base_config,
                                args=args,
                                device=device,
                                method=method,
                                ppl_only=True,
                                ppl_requested=ppl_requested,
                                lm_eval_tasks=lm_eval_tasks,
                            )
                        else:
                            payload = _run_seq(
                                root=root,
                                model_name=model_name,
                                model_dir=model_dir,
                                base_config=base_config,
                                args=args,
                                device=device,
                                method=method,
                                ppl_only=False,
                                ppl_requested=ppl_requested,
                                lm_eval_tasks=lm_eval_tasks,
                            )
                            result = payload
                            if ppl_requested:
                                if result.get("ppl") is None and not result.get("reason"):
                                    result["reason"] = "in_memory_seq_ppl_summary_missing"
                                    result["status"] = "failed"
                                    result["compare_status"] = "failed"
                            if lm_eval_tasks:
                                if not isinstance(result.get("lm_eval"), dict):
                                    result["lm_eval"] = _missing_in_memory_lm_eval_summary(lm_eval_tasks, args)
                                result = _finalize_method_result(result, ppl_requested=ppl_requested, lm_eval_tasks=lm_eval_tasks)
                                _write_json(method_dir / "summary.json", result)
                    elif method == "omniquant":
                        payload = _run_omniquant(
                            root=root,
                            model_name=model_name,
                            model_dir=model_dir,
                            base_config=base_config,
                            args=args,
                        )
                        if ppl_only:
                            if not payload.get("model_path"):
                                raise RuntimeError("OmniQuant did not produce a reloadable model for PPL evaluation.")
                            result = _run_ppl_eval(
                                model_name_or_path=str(payload["model_path"]),
                                summary_model_name=model_name,
                                method="omniquant",
                                method_dir=method_dir,
                                base_config=base_config,
                                args=args,
                                device=device,
                                notes=["PPL from SEQ evaluator on OmniQuant saved model"],
                            )
                        else:
                            result = payload
                            if ppl_requested:
                                if not payload.get("model_path"):
                                    raise RuntimeError("OmniQuant did not produce a reloadable model for mixed PPL evaluation.")
                                ppl_result = _run_ppl_eval(
                                    model_name_or_path=str(payload["model_path"]),
                                    summary_model_name=model_name,
                                    method="omniquant",
                                    method_dir=method_dir,
                                    base_config=base_config,
                                    args=args,
                                    device=device,
                                    notes=["PPL from SEQ evaluator on OmniQuant saved model"],
                                )
                                result = _merge_ppl_result(payload, ppl_result)
                            if lm_eval_tasks:
                                result = _run_lm_eval_for_method(
                                    method_payload=result,
                                    method_dir=method_dir,
                                    tasks=lm_eval_tasks,
                                    args=args,
                                    device=device,
                                )
                                result = _finalize_method_result(result, ppl_requested=ppl_requested, lm_eval_tasks=lm_eval_tasks)
                                _write_json(method_dir / "summary.json", result)
                    elif method in {"gptq_llmc", "smoothquant_llmc", "awq_llmc", "rtn_llmc", "llm_int8_llmc", "spinquant_llmc", "omniquant_llmc"}:
                        result = _run_llmc_method(
                            model_name=model_name,
                            method_dir=method_dir,
                            method=method,
                            args=args,
                            tasks=lm_eval_tasks,
                        )
                        if lm_eval_tasks:
                            llmc_skip_reason = None
                            if not result.get("model_path"):
                                llmc_skip_reason = (
                                    "llmc_save_mode_none_no_reloadable_artifact"
                                    if str(args.llmc_save_mode).strip().lower() == "none"
                                    else "method_did_not_produce_reloadable_model"
                                )
                            result = _run_lm_eval_for_method(
                                method_payload=result,
                                method_dir=method_dir,
                                tasks=lm_eval_tasks,
                                args=args,
                                device=device,
                                skip_reason=llmc_skip_reason,
                            )
                            result = _finalize_method_result(result, ppl_requested=ppl_requested, lm_eval_tasks=lm_eval_tasks)
                            _write_json(method_dir / "summary.json", result)
                        elif ppl_requested:
                            result["benchmark"] = "ppl"
                    else:
                        raise RuntimeError(f"Unsupported method escaped validation: {method}")
                except Exception as exc:
                    LOGGER.exception("Method failed: %s / %s", model_name, method)
                    result = {
                        "model": model_name,
                        "method": method,
                        "status": "error",
                        "reason": str(exc),
                        "run_dir": str(method_dir),
                        "lm_eval": {
                            "status": "error",
                            "reason": str(exc),
                            "tasks": lm_eval_tasks,
                            "flat": {
                                "lm_eval__status": "error",
                                "lm_eval__reason": str(exc),
                                "lm_eval__tasks": ",".join(lm_eval_tasks),
                            },
                        },
                    }
                    _write_json(method_dir / "summary.json", result)
                rows.append(_row_from_result(model_name, result))
                if _should_abort_for_lm_eval_failure(args, result, lm_eval_tasks):
                    _write_json(run_root / "global_summary.json", {"rows": rows})
                    _write_csv(run_root / "global_summary.csv", rows)
                    raise RuntimeError(
                        f"lm_eval_fail_policy={args.lm_eval_fail_policy} aborted after {model_name}/{method}: "
                        f"{(result.get('lm_eval') or {}).get('reason') or result.get('reason') or 'lm_eval failure'}"
                    )

        _write_json(run_root / "global_summary.json", {"rows": rows})
        _write_csv(run_root / "global_summary.csv", rows)
        _print_table("PPL Results" if ppl_only else ("Mixed Results" if benchmark_request["mixed"] else "lm-eval Results"), rows)
        LOGGER.info("Comparison complete: %s", run_root)
    finally:
        _restore_terminal_log(log_state)


if __name__ == "__main__":
    main()
