#!/usr/bin/env python3
import argparse
import copy
import csv
import datetime as dt
import gc
import hashlib
import json
import logging
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, PretrainedConfig

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "seq_core"

from . import entropy_metrics as entropy_mod
from benchmarks.core import (
    build_bench_summary,
    count_model_parameters,
    estimate_fp16_size,
    summarize_model_disk_footprint,
)
from .entropy_metrics import (
    compute_percentile_thresholds,
    compute_weight_entropy,
    compute_weight_magnitude,
    collect_activation_entropy,
    save_table_csv,
    save_table_json,
)
from .precision_policy import (
    assign_precision_tiers,
    apply_protections,
    build_precision_table,
    build_random_policy,
    verify_policy_constraints,
)
from .quantize_model import (
    apply_mixed_precision,
    compute_effective_bits,
    reload_quantized,
    save_quantized,
    verify_replacements,
)
from benchmarks.evaluation_suite import run_full_suite
from benchmarks.plotting import plot_run_baseline_vs_quant
from benchmarks.reporting import (
    build_ablation_rows,
    build_allreport,
    build_report,
    read_bench_summary,
    read_eval_summary,
    read_research_summary,
)


LOGGER = logging.getLogger("pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entropy-guided mixed-precision pipeline")
    parser.add_argument("--experiment", type=str, default="", help="Experiment name")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--model_name", type=str, default="", help="Override model name")
    parser.add_argument("--experiments_file", type=str, default="experiments.yaml")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    dtype = dtype_map.get(dtype_arg.lower(), torch.float16)
    if device == "cpu" and dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def setup_logging(log_path: Path) -> None:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers = []
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    LOGGER.addHandler(fh)
    LOGGER.addHandler(sh)


def now_timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_name(text: str, max_len: int = 48) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    if len(safe) > max_len:
        safe = safe[:max_len]
    return safe


def build_run_id(model_name: str, experiment_name: str, config: Dict[str, Any]) -> str:
    ts = now_timestamp()
    model_short = sanitize_name(model_name.replace("/", "_"), max_len=32)
    exp_short = sanitize_name(experiment_name, max_len=24)
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    hash8 = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
    return f"{ts}__{model_short}__{exp_short}__{hash8}"


def ensure_unique_run_dir(root: Path, run_id: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / run_id
    if not run_dir.exists():
        run_dir.mkdir(parents=False, exist_ok=False)
        return run_dir
    for _ in range(5):
        time.sleep(1.0)
        run_id = f"{run_id}_{now_timestamp()}"
        run_dir = root / run_id
        if not run_dir.exists():
            run_dir.mkdir(parents=False, exist_ok=False)
            return run_dir
    raise RuntimeError("Failed to create unique run directory")


def init_run_dirs(run_dir: Path) -> Dict[str, Path]:
    paths = {
        "run_dir": run_dir,
        "logs": run_dir / "logs",
        "entropy": run_dir / "entropy",
        "policy": run_dir / "policy",
        "quant": run_dir / "quant",
        "report": run_dir / "report",
        "model_quant": run_dir / "model_quantized",
        "eval_baseline": run_dir / "eval_baseline",
        "eval_quant": run_dir / "eval_quant",
        "bench_baseline": run_dir / "bench_baseline",
        "bench_quant": run_dir / "bench_quant",
    }
    for key, path in paths.items():
        if key == "model_quant":
            continue
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def write_text(path: Path, text: str) -> None:
    with path.open("w") as f:
        f.write(text)


def load_json_or_none(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception:
        return None


def get_pkg_version(pkg_name: str) -> Optional[str]:
    try:
        import importlib.metadata as metadata
        return metadata.version(pkg_name)
    except Exception:
        return None


def get_git_commit(root: Path) -> Optional[str]:
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        return None
    return None


def write_env_info(path: Path, root: Path) -> Dict[str, Any]:
    gpu_name = None
    total_vram = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory
    env = {
        "python_version": sys.version.replace("\n", " "),
        "torch_version": get_pkg_version("torch"),
        "transformers_version": get_pkg_version("transformers"),
        "bitsandbytes_version": get_pkg_version("bitsandbytes"),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
        "total_vram_bytes": total_vram,
        "git_commit": get_git_commit(root),
    }
    write_json(path, env)
    return env


def load_prompts(path: Path) -> List[str]:
    with path.open("r") as f:
        data = json.load(f)
    # Accept both legacy {"prompts": [...]} and list-based calibration files.
    if isinstance(data, dict):
        prompts_raw = data.get("prompts", [])
    elif isinstance(data, list):
        prompts_raw = data
    else:
        raise ValueError(f"Unsupported prompts JSON format in {path}")

    prompts: List[str] = []
    for item in prompts_raw:
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            text = item.get("text") or item.get("prompt")
        else:
            continue
        if isinstance(text, str) and text.strip():
            prompts.append(text)
    return prompts


def _normalize_calibration_mode(mode: Optional[str]) -> str:
    value = str(mode or "prompt_list").strip().lower()
    if value not in {"prompt_list", "corpus_chunks"}:
        raise ValueError(f"Unsupported calibration mode: {mode}")
    return value


def summarize_calibration_texts(
    texts: Sequence[str],
    *,
    tokenizer=None,
    seq_len: Optional[int] = None,
) -> Dict[str, Any]:
    clean_texts = [text for text in texts if isinstance(text, str) and text.strip()]
    char_lengths = [len(text) for text in clean_texts]
    summary: Dict[str, Any] = {
        "num_calibration_samples": int(len(clean_texts)),
        "mean_sample_chars": (sum(char_lengths) / len(char_lengths)) if char_lengths else None,
    }
    if tokenizer is None or not clean_texts:
        return summary

    try:
        token_lengths = [
            len(tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])
            for text in clean_texts
        ]
    except Exception:
        return summary

    summary["mean_token_length"] = (sum(token_lengths) / len(token_lengths)) if token_lengths else None
    if seq_len is not None and int(seq_len) > 0 and token_lengths:
        effective_lengths = [min(int(length), int(seq_len)) for length in token_lengths]
        padded_tokens = sum(max(int(seq_len) - effective, 0) for effective in effective_lengths)
        total_slots = int(seq_len) * len(effective_lengths)
        summary["pad_ratio"] = (padded_tokens / total_slots) if total_slots > 0 else None
    return summary


def _build_corpus_chunks(
    rows: Sequence[str],
    *,
    max_samples: int,
    min_chars: int,
) -> Tuple[List[str], Dict[str, Any]]:
    # Build naturally longer corpus passages from adjacent text rows so corpus
    # calibration uses representative text instead of short instruction prompts.
    clean_rows = [text.strip() for text in rows if isinstance(text, str) and text.strip()]
    chunks: List[str] = []
    current_rows: List[str] = []
    current_chars = 0

    for row in clean_rows:
        if len(row) >= min_chars and not current_rows:
            chunks.append(row)
            if len(chunks) >= max_samples:
                break
            continue

        current_rows.append(row)
        current_chars += len(row) + (2 if current_rows[:-1] else 0)
        if current_chars >= min_chars:
            chunks.append("\n\n".join(current_rows))
            current_rows = []
            current_chars = 0
            if len(chunks) >= max_samples:
                break

    if current_rows and len(chunks) < max_samples:
        chunks.append("\n\n".join(current_rows))

    metadata = {
        "num_source_rows": int(len(clean_rows)),
        "num_chunks_built": int(min(len(chunks), max_samples)),
    }
    return chunks[:max_samples], metadata


def build_calibration_texts(
    *,
    mode: str,
    prompt_file: Optional[str],
    corpus_file: Optional[str],
    max_samples: int,
    min_chars: int = 512,
    seq_len: Optional[int] = None,
    tokenizer=None,
    prompt_texts: Optional[Sequence[str]] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    requested_mode = _normalize_calibration_mode(mode)
    selected_file: Optional[str] = None
    texts: List[str] = []
    metadata: Dict[str, Any] = {
        "calibration_mode": requested_mode,
        "actual_dataset_mode": requested_mode,
        "calibration_source_file": None,
        "max_samples": int(max_samples),
        "seq_len": int(seq_len) if seq_len is not None else None,
    }

    if requested_mode == "prompt_list":
        if prompt_texts is not None:
            texts = [text for text in prompt_texts if isinstance(text, str) and text.strip()]
            selected_file = prompt_file
        else:
            selected_file = prompt_file
            if not selected_file:
                raise ValueError("prompt_list calibration requires prompt_file or prompt_texts")
            texts = load_prompts(Path(selected_file))
    else:
        selected_file = corpus_file or prompt_file
        if not selected_file:
            raise ValueError("corpus_chunks calibration requires corpus_file or prompt_file")
        corpus_rows = load_prompts(Path(selected_file))
        texts, corpus_meta = _build_corpus_chunks(
            corpus_rows,
            max_samples=int(max_samples),
            min_chars=max(int(min_chars), 1),
        )
        metadata.update(corpus_meta)
        metadata["min_chars"] = int(max(min_chars, 1))

    selected_texts = texts[: int(max_samples)]
    metadata["calibration_source_file"] = selected_file
    metadata.update(summarize_calibration_texts(selected_texts, tokenizer=tokenizer, seq_len=seq_len))
    return selected_texts, metadata


def load_eval_prompts(path: Path) -> List[Dict[str, Any]]:
    with path.open("r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        prompts_raw = data.get("prompts", [])
    elif isinstance(data, list):
        prompts_raw = data
    else:
        raise ValueError(f"Unsupported eval prompts JSON format in {path}")

    prompts: List[Dict[str, Any]] = []
    for item in prompts_raw:
        if isinstance(item, str):
            prompts.append({"prompt": item, "expected": None})
            continue
        if not isinstance(item, dict):
            continue
        if "prompt" in item and isinstance(item["prompt"], str):
            prompts.append(item)
            continue
        if "text" in item and isinstance(item["text"], str):
            prompts.append({"prompt": item["text"], "expected": item.get("expected")})
    return prompts


def load_experiments(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def resolve_experiment_config(config: Dict[str, Any], exp: Dict[str, Any]) -> Dict[str, Any]:
    resolved = {
        "model": config["model"],
        "seeds": config["seeds"],
        "calibration": config["calibration"],
        "thresholds": config["thresholds"],
        "protections": config["protections"],
        "evaluation": config["evaluation"],
        "benchmarks": config.get("benchmarks", {}),
    }
    resolved["experiment"] = exp
    if "thresholds" in exp:
        sweep_name = exp["thresholds"]
        resolved["thresholds"] = config["sweeps"][sweep_name]
    return resolved


def _patch_config_dict_for_compat(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    model_type = config_dict.get("model_type")
    if model_type == "qwen3_5":
        config_dict = dict(config_dict)
        config_dict["model_type"] = "qwen3_vl"
        text_config = config_dict.get("text_config")
        if isinstance(text_config, dict) and text_config.get("model_type") == "qwen3_5_text":
            text_config = dict(text_config)
            text_config["model_type"] = "qwen3_vl_text"
            rope_scaling = text_config.get("rope_scaling")
            rope_parameters = text_config.get("rope_parameters")

            # Qwen3.5 checkpoints may store MRoPE fields under rope_parameters,
            # while qwen3_vl expects a dict-like rope_scaling.
            if (not isinstance(rope_scaling, dict) or not rope_scaling) and isinstance(rope_parameters, dict):
                rope_scaling = dict(rope_parameters)
                rope_scaling.pop("partial_rotary_factor", None)
                rope_theta = rope_scaling.pop("rope_theta", None)
                if "type" in rope_scaling and "rope_type" not in rope_scaling:
                    rope_scaling["rope_type"] = rope_scaling["type"]
                rope_scaling.setdefault("rope_type", "default")
                text_config["rope_scaling"] = rope_scaling
                if text_config.get("rope_theta") is None:
                    if rope_theta is not None:
                        text_config["rope_theta"] = rope_theta
                    elif rope_parameters.get("rope_theta") is not None:
                        text_config["rope_theta"] = rope_parameters.get("rope_theta")
            elif isinstance(rope_scaling, dict):
                rope_scaling = dict(rope_scaling)
                rope_theta = rope_scaling.pop("rope_theta", None)
                if "type" in rope_scaling and "rope_type" not in rope_scaling:
                    rope_scaling["rope_type"] = rope_scaling["type"]
                rope_scaling.setdefault("rope_type", "default")
                text_config["rope_scaling"] = rope_scaling
                if text_config.get("rope_theta") is None and rope_theta is not None:
                    text_config["rope_theta"] = rope_theta
            else:
                text_config["rope_scaling"] = {"rope_type": "default"}
            config_dict["text_config"] = text_config
        vision_config = config_dict.get("vision_config")
        if isinstance(vision_config, dict) and vision_config.get("model_type") == "qwen3_5":
            vision_config = dict(vision_config)
            vision_config["model_type"] = "qwen3_vl"
            config_dict["vision_config"] = vision_config

    text_config = config_dict.get("text_config")
    if isinstance(text_config, dict) and text_config.get("model_type") == "ministral3":
        text_config = dict(text_config)
        text_config["model_type"] = "ministral"
        config_dict = dict(config_dict)
        config_dict["text_config"] = text_config
    return config_dict


def _load_config_with_compat(model_name: str, trust_remote_code: bool = False):
    try:
        return AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    except (KeyError, ValueError) as exc:
        # Generic fallback for newer checkpoints not yet mapped in local Transformers.
        if trust_remote_code:
            raise
        try:
            return AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        except Exception:
            pass
        # Compatibility path for known model_type renames.
        config_dict, _ = PretrainedConfig.get_config_dict(model_name)
        config_dict = _patch_config_dict_for_compat(config_dict)
        model_type = config_dict.get("model_type")
        if model_type not in {"ministral", "qwen3_vl"}:
            raise
        if not isinstance(model_type, str) or not model_type:
            raise RuntimeError(f"Invalid model_type in config for {model_name}")
        config_kwargs = dict(config_dict)
        config_kwargs.pop("model_type", None)
        return AutoConfig.for_model(model_type, **config_kwargs)


def load_model_and_tokenizer(
    model_name: str,
    device: str,
    dtype: torch.dtype,
    trust_remote_code: bool = False,
):
    config = _load_config_with_compat(model_name, trust_remote_code=trust_remote_code)
    model_trust_remote_code = bool(trust_remote_code)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=dtype,
            trust_remote_code=model_trust_remote_code,
        )
    except ValueError as exc:
        msg = str(exc)
        if "trust_remote_code=True" in msg and not model_trust_remote_code:
            model_trust_remote_code = True
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    config=config,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                )
            except ValueError as retry_exc:
                retry_msg = str(retry_exc)
                if "Unrecognized configuration class" not in retry_msg:
                    raise
                model = AutoModelForImageTextToText.from_pretrained(
                    model_name,
                    config=config,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                )
        elif "Unrecognized configuration class" in msg:
            model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                config=config,
                torch_dtype=dtype,
                trust_remote_code=model_trust_remote_code,
            )
        else:
            raise
    except ImportError as exc:
        raise RuntimeError(
            f"Failed to load model for {model_name} due to missing dependency. "
            f"Install it in your active environment and retry: {exc}"
        ) from exc
    model.to(device)
    model.eval()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=model_trust_remote_code)
    except ValueError as exc:
        if "trust_remote_code=True" not in str(exc):
            raise
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except ImportError as exc:
        raise RuntimeError(
            f"Failed to load tokenizer for {model_name} due to missing dependency. "
            f"Install it in your active environment and retry: {exc}"
        ) from exc
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def unload_model(model: Optional[torch.nn.Module], tokenizer) -> None:
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def append_run_index(index_path: Path, row: Dict[str, Any]) -> None:
    exists = index_path.exists()
    header_fields: List[str]
    if exists:
        with index_path.open("r", newline="") as f:
            first_line = f.readline().strip()
        header_fields = first_line.split(",") if first_line else list(row.keys())
    else:
        header_fields = list(row.keys())

    mapped_row: Dict[str, Any] = {}
    for field in header_fields:
        if field in row:
            mapped_row[field] = row[field]
        elif field == "run_name" and "experiment_name" in row:
            mapped_row[field] = row["experiment_name"]
        elif field == "experiment_name" and "run_name" in row:
            mapped_row[field] = row["run_name"]
        else:
            mapped_row[field] = ""

    with index_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header_fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(mapped_row)


def read_latest_runs(index_path: Path, model_name: str) -> List[Dict[str, Any]]:
    if not index_path.exists():
        return []
    rows = []
    with index_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("model_name") == model_name:
                rows.append(row)
    latest_by_exp = {}
    for row in rows:
        exp = row.get("experiment_name") or row.get("run_name")
        if exp is None:
            exp = row.get("run_id")
        latest_by_exp[exp] = row
    return list(latest_by_exp.values())


def run_experiment(
    root: Path,
    config: Dict[str, Any],
    exp: Dict[str, Any],
    prompts_path: Path,
    eval_prompts_path: Path,
    report_root: Path,
    runs_root: Path,
    model_override: Optional[str] = None,
) -> None:
    resolved = resolve_experiment_config(config, exp)
    experiment_name = exp["name"]
    original_model_name = resolved["model"]["name"]
    if model_override:
        resolved["model"] = dict(resolved["model"])
        resolved["model"]["name_original"] = original_model_name
        resolved["model"]["name"] = model_override
    model_name = resolved["model"]["name"]
    device = resolve_device(resolved["model"].get("device", "auto"))
    dtype = resolve_dtype(resolved["model"].get("dtype", "float16"), device)
    trust_remote_code = bool(resolved["model"].get("trust_remote_code", False))

    entropy_mod.MIN_SAMPLES = int(resolved["calibration"]["min_samples"])
    entropy_mod.MAX_NAN_FRAC = float(resolved["calibration"]["max_nan_frac"])

    run_id = build_run_id(model_name, experiment_name, resolved)
    run_dir = ensure_unique_run_dir(runs_root, run_id)
    paths = init_run_dirs(run_dir)

    setup_logging(paths["logs"] / "pipeline.log")
    LOGGER.info("Run ID: %s", run_id)
    if torch.cuda.is_available() and not os.getenv("PYTORCH_CUDA_ALLOC_CONF"):
        LOGGER.warning(
            "PYTORCH_CUDA_ALLOC_CONF is not set. Consider: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
        )

    set_seed(int(resolved["seeds"]["global"]))

    run_config = {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "model_name": model_name,
        "model_name_original": original_model_name,
        "model_name_override": model_override or None,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "timestamp": now_timestamp(),
        "resolved": resolved,
    }
    write_json(run_dir / "config.json", run_config)
    env_info = write_env_info(run_dir / "env.json", root)
    write_text(run_dir / "README_run.txt", f"run_id: {run_id}\nexperiment: {experiment_name}\n")

    prompts = load_prompts(prompts_path)
    eval_prompts = load_eval_prompts(eval_prompts_path)

    warnings: List[str] = []
    eval_seed = int(resolved["seeds"]["eval"])
    global_seed = int(resolved["seeds"]["global"])
    degeneracy_mode = str(resolved["calibration"].get("degeneracy_mode", "rms")).strip().lower() or "rms"

    evaluation_cfg = resolved["evaluation"]
    seq_metrics_cfg = evaluation_cfg.get("seq_metrics") or {}
    ppl_metric_cfg = seq_metrics_cfg.get("ppl") or {}
    long_metric_cfg = seq_metrics_cfg.get("long_context") or {}
    latency_metric_cfg = seq_metrics_cfg.get("latency_memory") or {}
    eval_config = {
        "eval_prompts": eval_prompts,
        "max_new_tokens": evaluation_cfg["max_new_tokens"],
        "temperature": evaluation_cfg["temperature"],
        "top_p": evaluation_cfg["top_p"],
        "latency_prompt": evaluation_cfg["latency_prompt"],
        "warmup_runs": int(latency_metric_cfg.get("warmup_runs", evaluation_cfg["warmup_runs"])),
        "measured_runs": int(latency_metric_cfg.get("measured_runs", evaluation_cfg["measured_runs"])),
        "seed": eval_seed,
        "long_context_lengths": long_metric_cfg.get("needle_lengths", evaluation_cfg.get("long_context_lengths")),
        "skip_long_context": not bool(long_metric_cfg.get("enabled", True)),
        "tail_risk_enabled": bool((seq_metrics_cfg.get("tail_risk") or {}).get("enabled", True)),
        "json_stress_enabled": bool((seq_metrics_cfg.get("json_stress") or {}).get("enabled", True)),
        "temperature_sweep_enabled": bool((seq_metrics_cfg.get("temperature_sweep") or {}).get("enabled", True)),
        "latency_memory_enabled": bool(latency_metric_cfg.get("enabled", True)),
        "mmlu": evaluation_cfg.get("mmlu") or {},
        "zero_shot": evaluation_cfg.get("zero_shot") or {},
        "lm_eval": copy.deepcopy(evaluation_cfg.get("lm_eval") or {}),
    }
    bench_config = resolved.get("benchmarks", {})
    run_ppl = bool(bench_config.get("run_ppl", True)) and bool(ppl_metric_cfg.get("enabled", True))
    run_size = bool(bench_config.get("run_size", True))
    eval_config.update(
        {
            "ppl_enabled": run_ppl,
            "ppl_mode": ppl_metric_cfg.get("mode", bench_config.get("ppl_mode", "proxy")),
            "ppl_dataset": ppl_metric_cfg.get("dataset", bench_config.get("ppl_dataset", "wikitext2")),
            "ppl_split": ppl_metric_cfg.get(
                "split",
                bench_config.get(
                "ppl_split",
                "test" if str(bench_config.get("ppl_mode", "proxy")).strip().lower() == "canonical" else "validation",
                ),
            ),
            "ppl_seq_len": int(
                ppl_metric_cfg.get(
                    "seq_len",
                    bench_config.get(
                    "ppl_seq_len",
                    2048 if str(bench_config.get("ppl_mode", "proxy")).strip().lower() == "canonical" else 256,
                    ),
                )
            ),
            "ppl_max_examples": ppl_metric_cfg.get(
                "max_examples",
                bench_config.get(
                "ppl_max_examples",
                None if str(bench_config.get("ppl_mode", "proxy")).strip().lower() == "canonical" else 128,
                ),
            ),
            "ppl_stride": ppl_metric_cfg.get("stride", bench_config.get("ppl_stride")),
            "ppl_full_corpus": ppl_metric_cfg.get(
                "full_corpus",
                bench_config.get(
                "ppl_full_corpus",
                str(bench_config.get("ppl_mode", "proxy")).strip().lower() == "canonical",
                ),
            ),
        }
    )

    # Phase 1: baseline evaluation
    set_seed(eval_seed)
    model, tokenizer = load_model_and_tokenizer(model_name, device, dtype, trust_remote_code=trust_remote_code)
    eval_baseline = run_full_suite(
        model,
        tokenizer,
        out_dir=str(paths["eval_baseline"]),
        config={
            **eval_config,
            "bench_dir": str(paths["bench_baseline"]),
            "quant_model_dir": None,
            "lm_eval_model": model,
            "lm_eval_tokenizer": tokenizer,
            "lm_eval_source": "in_memory_hflm",
        },
        device=device,
        dtype=dtype,
    )
    warnings.extend(eval_baseline.get("warnings", []))

    size_info_base: Dict[str, Any]
    if run_size:
        num_params = count_model_parameters(model)
        size_info_base = estimate_fp16_size(num_params)
        size_info_base["baseline_checkpoint_saved"] = False
    else:
        size_info_base = {"method": "disabled"}
    write_json(paths["bench_baseline"] / "size.json", size_info_base)

    ppl_info_base = eval_baseline.get("perplexity", {})
    if not ppl_info_base:
        ppl_info_base = {"ppl": None, "loss": None, "error": "missing"}
    write_json(paths["bench_baseline"] / "ppl.json", ppl_info_base)

    latency_base = load_json_or_none(paths["bench_baseline"] / "latency.json")
    memory_base = load_json_or_none(paths["bench_baseline"] / "memory.json")
    bench_summary_base = build_bench_summary(
        size_info=size_info_base,
        ppl_info=ppl_info_base,
        latency_info=latency_base,
        memory_info=memory_base,
        effective_bits_per_param=16.0,
        notes=["baseline_fp16_reference"],
    )
    write_json(paths["bench_baseline"] / "bench_summary.json", bench_summary_base)

    unload_model(model, tokenizer)

    # Phase 2: quantization pipeline
    set_seed(global_seed)
    model, tokenizer = load_model_and_tokenizer(model_name, device, dtype, trust_remote_code=trust_remote_code)

    weight_path = paths["entropy"] / "weight_entropy.json"
    act_path = paths["entropy"] / "activation_entropy.json"

    if weight_path.exists():
        with weight_path.open("r") as f:
            weight_table = json.load(f)
    else:
        weight_table = compute_weight_entropy(
            model,
            bins=resolved["calibration"]["bins"],
            clip=resolved["calibration"]["clip"],
            eps=resolved["calibration"]["eps"],
            exclude_embeddings=True,
            include_linear=True,
            degeneracy_mode=degeneracy_mode,
        )
        save_table_json(str(weight_path), weight_table)
        save_table_csv(
            str(paths["entropy"] / "weight_entropy.csv"),
            weight_table,
            ["module_name", "entropy_bits", "mean", "std", "num_params", "shape", "flags"],
        )

    if act_path.exists():
        with act_path.open("r") as f:
            act_table = json.load(f)
    else:
        act_table = collect_activation_entropy(
            model,
            tokenizer,
            prompts,
            seq_len=resolved["calibration"]["seq_len"],
            device=device,
            dtype=dtype,
            bins=resolved["calibration"]["bins"],
            bootstrap_iters=resolved["calibration"]["bootstrap_iters"],
            clip=resolved["calibration"]["clip"],
            eps=resolved["calibration"]["eps"],
            summary_path=str(paths["entropy"] / "activation_entropy_summary.json"),
            degeneracy_mode=degeneracy_mode,
        )
        save_table_json(str(act_path), act_table)
        save_table_csv(
            str(paths["entropy"] / "activation_entropy.csv"),
            act_table,
            [
                "module_name",
                "entropy_bits",
                "entropy_std",
                "tail_entropy_p95",
                "mean",
                "std",
                "sample_count",
                "flags",
            ],
        )

    unreliable_modules = set()
    for name, row in weight_table.items():
        if row.get("flags", {}).get("unreliable"):
            unreliable_modules.add(name)
    for name, row in act_table.items():
        if row.get("flags", {}).get("unreliable"):
            unreliable_modules.add(name)

    if unreliable_modules:
        warnings.append(f"Unreliable entropy modules: {len(unreliable_modules)}")

    policy_type = exp["policy"]
    thresholds = resolved["thresholds"]

    if policy_type == "magnitude_baseline":
        magnitude_table = compute_weight_magnitude(model)
        proxy_table = {
            name: {"entropy_bits": data["mean_abs"], "flags": {"unreliable": False}}
            for name, data in magnitude_table.items()
        }
        threshold_info = compute_percentile_thresholds(
            proxy_table,
            act_table,
            thresholds["weight_high_pct"],
            thresholds["act_high_pct"],
        )
        weight_high = threshold_info["weight_high"]
        act_high = threshold_info["act_high"]
    else:
        threshold_info = compute_percentile_thresholds(
            weight_table,
            act_table,
            thresholds["weight_high_pct"],
            thresholds["act_high_pct"],
        )
        weight_high = threshold_info["weight_high"]
        act_high = threshold_info["act_high"]

    if policy_type == "weight_only":
        act_high = {name: False for name in act_high}
    if policy_type == "activation_only":
        weight_high = {name: False for name in weight_high}

    base_map = assign_precision_tiers(weight_high, act_high)
    if policy_type == "random_policy":
        base_map = build_random_policy(base_map, seed=global_seed)

    protection_config = dict(resolved["protections"])
    protection_config["unreliable_modules"] = sorted(unreliable_modules)

    final_map, overrides, tier_counts = apply_protections(base_map, model, protection_config)
    verify_policy_constraints(final_map, model, protection_config)

    override_counts: Dict[str, int] = {}
    for reason in overrides.values():
        for part in reason.split(";"):
            override_counts[part] = override_counts.get(part, 0) + 1

    write_json(paths["policy"] / "precision_map.json", final_map)
    write_json(paths["policy"] / "override_summary.json", override_counts)
    write_json(paths["entropy"] / "thresholds.json", threshold_info)

    precision_table = build_precision_table(
        model,
        weight_table,
        act_table,
        weight_high,
        act_high,
        base_map,
        final_map,
        overrides,
    )

    with (paths["policy"] / "precision_table.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(precision_table[0].keys()) if precision_table else [],
        )
        if precision_table:
            writer.writeheader()
            for row in precision_table:
                writer.writerow(row)

    param_counts = {row["module_name"]: row["param_count"] for row in precision_table}

    quant_info = apply_mixed_precision(
        model,
        final_map,
        device=device,
        dtype_fp16=dtype,
        bnb_4bit="nf4",
        bnb_compute_dtype=torch.float16,
    )
    verify_info = verify_replacements(model, final_map)
    if verify_info["mismatches"]:
        raise RuntimeError(f"Quantization mismatches: {len(verify_info['mismatches'])}")

    effective_bits = compute_effective_bits(final_map, param_counts)
    write_json(paths["quant"] / "verification.json", verify_info)
    write_json(paths["quant"] / "replacement_counts.json", quant_info)
    write_json(paths["quant"] / "effective_bits.json", effective_bits)

    paths["model_quant"].mkdir(parents=True, exist_ok=True)
    save_quantized(model, tokenizer, str(paths["model_quant"]))

    reload_text, reload_error = reload_quantized(
        model_name,
        final_map,
        device=device,
        dtype=dtype,
    )
    if reload_error:
        warnings.append(reload_error)
    write_text(
        paths["bench_quant"] / "generation_quant_reloaded.txt",
        reload_text if reload_text else reload_error or "reload_failed",
    )

    set_seed(eval_seed)
    eval_quant = run_full_suite(
        model,
        tokenizer,
        out_dir=str(paths["eval_quant"]),
        config={
            **eval_config,
            "bench_dir": str(paths["bench_quant"]),
            "quant_model_dir": str(paths["model_quant"]),
            "lm_eval_model": model,
            "lm_eval_tokenizer": tokenizer,
            "lm_eval_source": "in_memory_hflm",
        },
        device=device,
        dtype=dtype,
    )
    warnings.extend(eval_quant.get("warnings", []))

    size_info_quant: Dict[str, Any]
    if run_size:
        size_info_quant = summarize_model_disk_footprint(str(paths["model_quant"]))
    else:
        size_info_quant = {"method": "disabled"}
    write_json(paths["bench_quant"] / "size.json", size_info_quant)

    ppl_info_quant = eval_quant.get("perplexity", {})
    if not ppl_info_quant:
        ppl_info_quant = {"ppl": None, "loss": None, "error": "missing"}
    write_json(paths["bench_quant"] / "ppl.json", ppl_info_quant)

    latency_quant = load_json_or_none(paths["bench_quant"] / "latency.json")
    memory_quant = load_json_or_none(paths["bench_quant"] / "memory.json")
    bench_summary_quant = build_bench_summary(
        size_info=size_info_quant,
        ppl_info=ppl_info_quant,
        latency_info=latency_quant,
        memory_info=memory_quant,
        effective_bits_per_param=effective_bits.get("effective_bits"),
        notes=["quantized_mixed_precision"],
    )
    write_json(paths["bench_quant"] / "bench_summary.json", bench_summary_quant)

    unload_model(model, tokenizer)

    eval_baseline_summary, eval_baseline_warn = read_eval_summary(
        str(paths["eval_baseline"] / "eval_summary.json")
    )
    eval_quant_summary, eval_quant_warn = read_eval_summary(
        str(paths["eval_quant"] / "eval_summary.json")
    )
    warnings.extend(eval_baseline_warn)
    warnings.extend(eval_quant_warn)

    bench_baseline_summary, bench_baseline_warn = read_bench_summary(
        str(paths["bench_baseline"] / "bench_summary.json")
    )
    bench_quant_summary, bench_quant_warn = read_bench_summary(
        str(paths["bench_quant"] / "bench_summary.json")
    )
    warnings.extend(bench_baseline_warn)
    warnings.extend(bench_quant_warn)

    research_summary = read_research_summary(str(root / "research_scouting.md"))
    latest_runs = read_latest_runs(runs_root / "index.csv", model_name)
    experiment_names = [entry["name"] for entry in config.get("experiments", [])]
    ablation_rows = build_ablation_rows(latest_runs, experiment_names)

    report_text = build_report(
        metadata={
            "run_id": run_id,
            "experiment_name": experiment_name,
            "model_name": model_name,
            "device": device,
            "dtype": str(dtype).replace("torch.", ""),
            "run_dir": str(run_dir),
        },
        thresholds=threshold_info,
        protections=protection_config,
        weight_table=weight_table,
        act_table=act_table,
        precision_table=precision_table,
        effective_bits=effective_bits,
        eval_baseline=eval_baseline_summary,
        eval_quant=eval_quant_summary,
        bench_baseline=bench_baseline_summary,
        bench_quant=bench_quant_summary,
        ablation_rows=ablation_rows,
        warnings=warnings,
        research_summary_lines=research_summary,
    )

    report_path = paths["report"] / "report.md"
    write_text(report_path, report_text)
    fig_paths = plot_run_baseline_vs_quant(
        run_dir,
        eval_baseline_summary,
        eval_quant_summary,
        effective_bits,
        threshold_info,
    )
    if fig_paths:
        with report_path.open("a") as f:
            f.write("\n## Figures\n\n")
            for path in fig_paths:
                rel = f"figures/{Path(path).name}"
                f.write(f"![{Path(path).stem}]({rel})\n\n")
    report_root.mkdir(parents=True, exist_ok=True)
    report_copy = report_root / f"report_{now_timestamp()}_{run_id}.md"
    shutil.copy(report_path, report_copy)

    results_json = {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "model_name": model_name,
        "effective_bits": effective_bits.get("effective_bits"),
        "eval_baseline": eval_baseline_summary,
        "eval_quant": eval_quant_summary,
        "report_path": str(report_copy),
    }
    write_json(report_root / f"results_{now_timestamp()}_{run_id}.json", results_json)

    latency_quant = eval_quant_summary.get("latency", {})
    tokens_quant = latency_quant.get("tokens_per_sec", {}) if isinstance(latency_quant, dict) else {}
    latency_sec_quant = {}
    if isinstance(latency_quant, dict):
        latency_sec_quant = latency_quant.get("decode_sec") or latency_quant.get("prefill_sec") or {}
    memory_quant = eval_quant_summary.get("memory", {})

    index_row = {
        "run_id": run_id,
        "timestamp": now_timestamp(),
        "model_name": model_name,
        "experiment_name": experiment_name,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "weight_high_pct": thresholds["weight_high_pct"],
        "act_high_pct": thresholds["act_high_pct"],
        "effective_bits": effective_bits.get("effective_bits"),
        "tail_exact_match": eval_quant_summary.get("tail_risk", {}).get("exact_span_match_rate"),
        "json_success_rate": eval_quant_summary.get("json_stress", {}).get("success_rate"),
        "latency_p50_sec": latency_sec_quant.get("p50") if isinstance(latency_sec_quant, dict) else None,
        "tokens_per_sec_mean": tokens_quant.get("mean") if isinstance(tokens_quant, dict) else None,
        "peak_memory_bytes": memory_quant.get("peak_memory_bytes"),
        "run_dir": str(run_dir),
    }
    append_run_index(runs_root / "index.csv", index_row)

    latest_runs_post = read_latest_runs(runs_root / "index.csv", model_name)
    allreport_text = build_allreport(latest_runs_post, runs_root=str(runs_root))
    write_text(report_root / "allreport.md", allreport_text)

    latest_link = runs_root / "latest"
    try:
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(run_dir.name)
    except Exception:
        write_text(runs_root / "latest.txt", run_dir.name)

    LOGGER.info("Run complete: %s", run_id)


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    experiments_arg = Path(args.experiments_file)
    experiments_path = experiments_arg if experiments_arg.is_absolute() else root / experiments_arg

    if not experiments_path.exists():
        raise FileNotFoundError(f"Missing experiments file: {experiments_path}")

    config = load_experiments(experiments_path)
    experiments = config.get("experiments", [])
    exp_map = {exp["name"]: exp for exp in experiments}

    if args.all:
        selected = experiments
    elif args.experiment:
        if args.experiment not in exp_map:
            raise ValueError(f"Experiment not found: {args.experiment}")
        selected = [exp_map[args.experiment]]
    else:
        raise ValueError("Specify --experiment <name> or --all")

    prompts_path = root / "calibration_prompts.json"
    eval_prompts_path = root / "eval_prompts.json"
    report_root = root / "reports"
    runs_root = root / "runs"

    model_override = args.model_name.strip() if args.model_name else None
    for exp in selected:
        run_experiment(
            root,
            config,
            exp,
            prompts_path,
            eval_prompts_path,
            report_root,
            runs_root,
            model_override=model_override,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
