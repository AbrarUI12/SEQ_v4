from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


_WSL_DRIVE_RE = re.compile(r"^/mnt/([a-zA-Z])(?:/(.*))?$")

_ARTIFACT_DIRS = (
    ("transformed_model", "transformed_model"),
    ("fake_quant_model", "fake_quant_model"),
    ("vllm_quant_model", "vllm_quant_model"),
    ("autoawq_quant_model", "autoawq_quant_model"),
    ("lightllm_quant_model", "lightllm_quant_model"),
    ("sgl_quant_model", "sgl_quant_model"),
)


def _first_existing_path(candidates: List[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _host_path(value: Path | str) -> Path:
    text = str(value)
    if os.name == "nt":
        match = _WSL_DRIVE_RE.match(text)
        if match:
            drive = match.group(1).upper()
            tail = (match.group(2) or "").replace("/", "\\")
            return Path(f"{drive}:\\{tail}") if tail else Path(f"{drive}:\\")
    return Path(text)


def _to_wsl_path(value: Path | str) -> str:
    text = str(value).replace("\\", "/")
    if text.startswith("/mnt/"):
        return text
    drive_match = re.match(r"^([A-Za-z]):/(.*)$", text)
    if drive_match:
        drive = drive_match.group(1).lower()
        tail = drive_match.group(2)
        return f"/mnt/{drive}/{tail}"
    return Path(text).as_posix()


def disk_usage_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def validate_llmc_repo(llmc_repo: Path) -> Dict[str, Any]:
    repo_path = _host_path(llmc_repo).resolve()
    required_candidates = {
        "entrypoint": [repo_path / "llmc" / "__main__.py"],
        "gptq_config": [repo_path / "configs" / "quantization" / "methods" / "GPTQ" / "gptq_w_only.yml"],
        "smoothquant_config": [repo_path / "configs" / "quantization" / "methods" / "SmoothQuant" / "smoothquant_w_a.yml"],
        "awq_config": [
            repo_path / "configs" / "quantization" / "methods" / "Awq" / "awq_w_only.yml",
            repo_path / "configs" / "quantization" / "methods" / "AWQ" / "awq_w_only.yml",
        ],
        "omniquant_config": [
            repo_path / "configs" / "quantization" / "methods" / "OmniQuant" / "omniq_w_only.yml",
            repo_path / "configs" / "quantization" / "methods" / "Omniquant" / "omniq_w_only.yml",
        ],
    }
    resolved_required = {
        name: _first_existing_path(candidates) for name, candidates in required_candidates.items()
    }
    missing = [name for name, path in resolved_required.items() if path is None]
    return {
        "ok": not missing,
        "repo_path": str(repo_path),
        "repo_wsl_path": _to_wsl_path(repo_path),
        "required_files": {name: str(path) if path is not None else None for name, path in resolved_required.items()},
        "missing": missing,
    }


def validate_llmc_venv(llmc_venv: Optional[Path], llmc_repo: Optional[Path] = None) -> Dict[str, Any]:
    default_venv = _host_path(llmc_repo).resolve() / ".venv-llmc" if llmc_venv is None and llmc_repo is not None else None
    venv_path = _host_path(llmc_venv or default_venv or ".venv-llmc").resolve()
    python_path = venv_path / "bin" / "python"
    activate_path = venv_path / "bin" / "activate"
    missing = [name for name, path in {"python": python_path, "activate": activate_path}.items() if not path.exists()]
    return {
        "ok": not missing,
        "venv_path": str(venv_path),
        "venv_wsl_path": _to_wsl_path(venv_path),
        "python_path": str(python_path),
        "python_wsl_path": _to_wsl_path(python_path),
        "activate_path": str(activate_path),
        "activate_wsl_path": _to_wsl_path(activate_path),
        "missing": missing,
    }


def get_llmc_commit(llmc_repo: Path) -> Optional[str]:
    repo_path = _host_path(llmc_repo).resolve()
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return None


def render_config(
    *,
    template_path: Path,
    output_path: Path,
    model_type: str,
    model_path: str,
    torch_dtype: str,
    tokenizer_mode: str,
    calib_name: str,
    calib_download: bool,
    calib_n_samples: int,
    calib_bs: int,
    calib_seq_len: int,
    calib_preproc: str,
    calib_seed: int,
    eval_pos: List[str],
    eval_name: str,
    eval_download: bool,
    eval_bs: int,
    eval_seq_len: int,
    inference_per_block: bool,
    save_trans: bool,
    save_fake: bool,
    save_vllm: bool,
    save_path: str,
) -> Dict[str, Any]:
    with template_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    config.setdefault("model", {})
    config["model"]["type"] = model_type
    config["model"]["path"] = model_path
    config["model"]["torch_dtype"] = torch_dtype
    config["model"]["tokenizer_mode"] = tokenizer_mode

    config.setdefault("calib", {})
    config["calib"]["name"] = calib_name
    config["calib"]["download"] = bool(calib_download)
    config["calib"]["n_samples"] = int(calib_n_samples)
    config["calib"]["bs"] = int(calib_bs)
    config["calib"]["seq_len"] = int(calib_seq_len)
    config["calib"]["preproc"] = calib_preproc
    config["calib"]["seed"] = int(calib_seed)

    config.setdefault("eval", {})
    config["eval"]["eval_pos"] = list(eval_pos)
    config["eval"]["name"] = eval_name
    config["eval"]["download"] = bool(eval_download)
    config["eval"]["bs"] = int(eval_bs)
    config["eval"]["seq_len"] = int(eval_seq_len)
    config["eval"]["inference_per_block"] = bool(inference_per_block)

    config.setdefault("save", {})
    config["save"]["save_trans"] = bool(save_trans)
    config["save"]["save_fake"] = bool(save_fake)
    config["save"]["save_vllm"] = bool(save_vllm)
    config["save"]["save_path"] = save_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return config


def build_command(
    *,
    llmc_repo: Path,
    llmc_venv: Path,
    rendered_config_path: Path,
    task_id: str,
) -> Dict[str, Any]:
    repo_wsl = _to_wsl_path(_host_path(llmc_repo).resolve())
    activate_wsl = _to_wsl_path(_host_path(llmc_venv).resolve() / "bin" / "activate")
    config_wsl = _to_wsl_path(_host_path(rendered_config_path).resolve())

    shell_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(repo_wsl)}",
        f"source {shlex.quote(activate_wsl)}",
        f"export PYTHONPATH={shlex.quote(repo_wsl)}:${{PYTHONPATH:-}}",
        "torchrun --standalone --nproc_per_node=1 llmc/__main__.py \\",
        f"  --config {shlex.quote(config_wsl)} \\",
        f"  --task_id {shlex.quote(task_id)}",
    ]
    script_text = "\n".join(shell_lines) + "\n"
    inline_script = "\n".join(shell_lines[1:])
    if os.name == "nt":
        exec_command = ["wsl.exe", "bash", "-lc", inline_script]
    else:
        exec_command = ["bash", "-lc", inline_script]

    return {
        "script_text": script_text,
        "exec_command": exec_command,
        "display_command": inline_script,
    }


def parse_llmc_log(log_path: Path) -> Dict[str, Any]:
    ppl_pattern = re.compile(r"EVAL: ppl on (?P<dataset>\S+) is (?P<ppl>[0-9.eE+-]+)")
    duration_pattern = re.compile(r"llmc_duration_time:\s*(?P<duration>[0-9.eE+-]+)\s*s")
    finish_pattern = re.compile(r"--- llmc finished ---")

    ppl_matches: List[Dict[str, Any]] = []
    duration_value: Optional[float] = None
    finished = False

    if log_path.exists():
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                ppl_match = ppl_pattern.search(line)
                if ppl_match:
                    ppl_matches.append(
                        {
                            "dataset": ppl_match.group("dataset"),
                            "ppl": float(ppl_match.group("ppl")),
                        }
                    )
                duration_match = duration_pattern.search(line)
                if duration_match:
                    duration_value = float(duration_match.group("duration"))
                if finish_pattern.search(line):
                    finished = True

    notes: List[str] = []
    if len(ppl_matches) > 1:
        notes.append("multiple_ppl_lines_found_using_last")
    last_ppl = ppl_matches[-1] if ppl_matches else None
    return {
        "ppl": last_ppl["ppl"] if last_ppl else None,
        "ppl_dataset": last_ppl["dataset"] if last_ppl else None,
        "duration_sec": duration_value,
        "finished": finished,
        "notes": notes,
    }


def _discover_artifact(save_path: Path) -> Dict[str, Any]:
    if not save_path.exists():
        return {
            "artifact_path": None,
            "artifact_kind": None,
            "quant_disk_bytes": 0,
            "quant_disk_gb": 0.0,
            "hf_loadable_candidate": False,
        }

    artifact_dir: Optional[Path] = None
    artifact_kind: Optional[str] = None
    for dirname, kind in _ARTIFACT_DIRS:
        candidate = save_path / dirname
        if candidate.exists():
            artifact_dir = candidate
            artifact_kind = kind
            break

    if artifact_dir is None:
        artifact_dir = save_path
        artifact_kind = "save_path_root"

    quant_bytes = disk_usage_bytes(artifact_dir)
    tokenizer_markers = (
        artifact_dir / "tokenizer.json",
        artifact_dir / "tokenizer_config.json",
        artifact_dir / "vocab.json",
        artifact_dir / "merges.txt",
    )
    hf_loadable_candidate = (artifact_dir / "config.json").exists() and any(marker.exists() for marker in tokenizer_markers)
    return {
        "artifact_path": str(artifact_dir),
        "artifact_kind": artifact_kind,
        "quant_disk_bytes": quant_bytes,
        "quant_disk_gb": quant_bytes / float(1024 ** 3),
        "hf_loadable_candidate": hf_loadable_candidate,
    }


@dataclass
class LLMCRunSpec:
    method: str
    model: str
    task_id: str
    llmc_repo: Path
    llmc_venv: Optional[Path]
    template_path: Path
    method_run_dir: Path
    rendered_config_path: Path
    model_type: str
    torch_dtype: str
    tokenizer_mode: str
    calib_name: str
    calib_download: bool
    calib_n_samples: int
    calib_bs: int
    calib_seq_len: int
    calib_preproc: str
    calib_seed: int
    eval_pos: List[str]
    eval_name: str
    eval_download: bool
    eval_bs: int
    eval_seq_len: int
    inference_per_block: bool
    save_trans: bool
    save_fake: bool
    save_vllm: bool
    save_path_host: Path
    dry_run: bool
    log_filename: str = "combined.log"
    command_filename: str = "llmc_command.sh"
    adapter_summary_filename: str = "llmc_adapter_summary.json"


def run_llmc(spec: LLMCRunSpec) -> Dict[str, Any]:
    method_dir = spec.method_run_dir.resolve()
    logs_dir = method_dir / "logs"
    log_path = logs_dir / spec.log_filename
    command_path = method_dir / spec.command_filename
    adapter_summary_path = method_dir / spec.adapter_summary_filename
    method_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    repo_validation = validate_llmc_repo(spec.llmc_repo)
    venv_validation = validate_llmc_venv(spec.llmc_venv, spec.llmc_repo)
    llmc_commit = get_llmc_commit(spec.llmc_repo)
    save_path_host = spec.save_path_host.resolve()
    save_path_wsl = _to_wsl_path(save_path_host)

    notes: List[str] = []
    status = "dry_run" if spec.dry_run else "failed"
    returncode: Optional[int] = None
    reason: Optional[str] = None

    summary: Dict[str, Any] = {
        "method": spec.method,
        "model": spec.model,
        "task_id": spec.task_id,
        "llmc_repo": repo_validation["repo_wsl_path"],
        "llmc_repo_host": repo_validation["repo_path"],
        "llmc_commit": llmc_commit,
        "llmc_venv": venv_validation["venv_wsl_path"],
        "llmc_venv_host": venv_validation["venv_path"],
        "rendered_config_path": str(spec.rendered_config_path.resolve()),
        "save_path": save_path_wsl,
        "log_path": str(log_path.resolve()),
        "command": None,
        "dry_run": spec.dry_run,
        "returncode": None,
        "status": status,
        "reason": None,
        "ppl": None,
        "ppl_dataset": None,
        "duration_sec": None,
        "artifact_path": None,
        "artifact_kind": None,
        "quant_disk_bytes": 0,
        "quant_disk_gb": 0.0,
        "hf_loadable_candidate": False,
        "notes": notes,
        "timestamp": _utc_timestamp(),
        "repo_validation": repo_validation,
        "venv_validation": venv_validation,
    }

    if not repo_validation["ok"]:
        reason = (
            "LLMC repo validation failed: missing "
            + ", ".join(repo_validation["missing"])
            + ". Pass --llmc_repo /path/to/LightCompress."
        )
        summary["reason"] = reason
        write_json(adapter_summary_path, summary)
        return summary

    if not venv_validation["ok"]:
        reason = (
            "LLMC venv validation failed: missing "
            + ", ".join(venv_validation["missing"])
            + ". Pass --llmc_venv /path/to/.venv-llmc."
        )
        summary["reason"] = reason
        write_json(adapter_summary_path, summary)
        return summary

    render_config(
        template_path=spec.template_path,
        output_path=spec.rendered_config_path,
        model_type=spec.model_type,
        model_path=spec.model,
        torch_dtype=spec.torch_dtype,
        tokenizer_mode=spec.tokenizer_mode,
        calib_name=spec.calib_name,
        calib_download=spec.calib_download,
        calib_n_samples=spec.calib_n_samples,
        calib_bs=spec.calib_bs,
        calib_seq_len=spec.calib_seq_len,
        calib_preproc=spec.calib_preproc,
        calib_seed=spec.calib_seed,
        eval_pos=spec.eval_pos,
        eval_name=spec.eval_name,
        eval_download=spec.eval_download,
        eval_bs=spec.eval_bs,
        eval_seq_len=spec.eval_seq_len,
        inference_per_block=spec.inference_per_block,
        save_trans=spec.save_trans,
        save_fake=spec.save_fake,
        save_vllm=spec.save_vllm,
        save_path=save_path_wsl,
    )

    command_info = build_command(
        llmc_repo=spec.llmc_repo,
        llmc_venv=_host_path(spec.llmc_venv or _host_path(spec.llmc_repo) / ".venv-llmc"),
        rendered_config_path=spec.rendered_config_path,
        task_id=spec.task_id,
    )
    command_path.write_text(command_info["script_text"], encoding="utf-8")
    summary["command"] = command_info["display_command"]

    if spec.dry_run:
        status = "dry_run"
        summary["status"] = status
        summary["reason"] = "dry_run_requested"
        notes.append("command_rendered_but_not_executed")
        write_json(adapter_summary_path, summary)
        return summary

    if save_path_host.exists():
        if save_path_host.is_dir():
            shutil.rmtree(save_path_host)
        else:
            save_path_host.unlink()
        notes.append("removed_stale_generated_save_path")

    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command_info["exec_command"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
        returncode = process.wait()

    parsed = parse_llmc_log(log_path)
    summary.update(
        {
            "returncode": returncode,
            "ppl": parsed["ppl"],
            "ppl_dataset": parsed["ppl_dataset"],
            "duration_sec": parsed["duration_sec"],
        }
    )
    notes.extend(parsed["notes"])

    artifact_info = _discover_artifact(save_path_host)
    summary.update(artifact_info)

    if returncode == 0 and parsed["finished"]:
        status = "success"
    else:
        status = "failed"
        if returncode != 0:
            reason = f"llmc exited with return code {returncode}"
        elif not parsed["finished"]:
            reason = "llmc log did not contain finish marker"
    summary["status"] = status
    summary["reason"] = reason
    write_json(adapter_summary_path, summary)
    return summary
