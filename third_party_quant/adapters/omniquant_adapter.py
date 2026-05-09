#!/usr/bin/env python3
"""Thin SEQ-facing adapter for pinned upstream OpenGVLab OmniQuant.

This module intentionally invokes upstream ``main.py`` through a subprocess.
It does not port or reimplement OmniQuant quantization logic.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


UPSTREAM_REPO_URL = "https://github.com/OpenGVLab/OmniQuant"
UPSTREAM_COMMIT_SHA = "feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14"
ENVIRONMENT_NAME = "omniquant-upstream"

ADAPTER_DIR = Path(__file__).resolve().parent
THIRD_PARTY_DIR = ADAPTER_DIR.parent
DEFAULT_UPSTREAM_DIR = THIRD_PARTY_DIR / "OmniQuant"

OFFICIAL_MODEL_FAMILY_HINTS = (
    "llama",
    "llama-2",
    "opt",
    "falcon",
    "mixtral",
)


@dataclass
class OmniQuantRequest:
    model: str
    output_dir: Path
    python_executable: str = sys.executable
    upstream_dir: Path = DEFAULT_UPSTREAM_DIR
    cache_dir: Path = Path.home() / "seq-cache" / "omniquant"
    save_dir: Optional[Path] = None
    cuda_visible_devices: Optional[str] = None
    environment_name: str = ENVIRONMENT_NAME

    wbits: int = 4
    abits: int = 16
    group_size: Optional[int] = 128
    alpha: float = 0.5
    epochs: int = 20
    calib_dataset: str = "wikitext2"
    nsamples: int = 128
    batch_size: int = 1
    seed: int = 2
    tasks: str = ""
    num_fewshot: int = 0
    limit: int = -1
    attn_implementation: str = "eager"
    net: Optional[str] = None
    resume: Optional[Path] = None
    act_scales: Optional[Path] = None
    act_shifts: Optional[Path] = None

    eval_ppl: bool = True
    lwc: bool = True
    let: bool = False
    aug_loss: bool = False
    symmetric: bool = False
    disable_zero_point: bool = False
    real_quant: bool = False
    multigpu: bool = False
    deactive_amp: bool = False

    extra_args: List[str] = field(default_factory=list)


@dataclass
class OmniQuantResult:
    command: List[str]
    cwd: str
    output_dir: str
    save_dir: Optional[str]
    provenance_path: str
    stdout_path: str
    stderr_path: str
    returncode: Optional[int]
    dry_run: bool


def _path_str(path: Path) -> str:
    return str(path.expanduser().resolve())


def _append_flag(command: List[str], enabled: bool, flag: str) -> None:
    if enabled:
        command.append(flag)


def _append_value(command: List[str], flag: str, value: Optional[Any]) -> None:
    if value is None:
        return
    if isinstance(value, Path):
        value = _path_str(value)
    command.extend([flag, str(value)])


def model_support_status(model: str, net: Optional[str] = None) -> str:
    value = f"{model} {net or ''}".lower()
    if any(hint in value for hint in OFFICIAL_MODEL_FAMILY_HINTS):
        return "officially_supported"
    return "adapted_not_officially_supported"


def build_command(request: OmniQuantRequest) -> List[str]:
    main_py = request.upstream_dir / "main.py"
    command = [request.python_executable, str(main_py)]

    _append_value(command, "--model", request.model)
    _append_value(command, "--cache_dir", request.cache_dir)
    _append_value(command, "--output_dir", request.output_dir)
    _append_value(command, "--save_dir", request.save_dir)
    _append_value(command, "--resume", request.resume)
    _append_value(command, "--calib_dataset", request.calib_dataset)
    _append_value(command, "--nsamples", request.nsamples)
    _append_value(command, "--batch_size", request.batch_size)
    _append_value(command, "--seed", request.seed)
    _append_value(command, "--tasks", request.tasks)
    _append_value(command, "--num_fewshot", request.num_fewshot)
    _append_value(command, "--wbits", request.wbits)
    _append_value(command, "--abits", request.abits)
    _append_value(command, "--group_size", request.group_size)
    _append_value(command, "--alpha", request.alpha)
    _append_value(command, "--epochs", request.epochs)
    _append_value(command, "--limit", request.limit)
    _append_value(command, "--attn_implementation", request.attn_implementation)
    _append_value(command, "--net", request.net)
    _append_value(command, "--act-scales", request.act_scales)
    _append_value(command, "--act-shifts", request.act_shifts)

    _append_flag(command, request.eval_ppl, "--eval_ppl")
    _append_flag(command, request.lwc, "--lwc")
    _append_flag(command, request.let, "--let")
    _append_flag(command, request.aug_loss, "--aug_loss")
    _append_flag(command, request.symmetric, "--symmetric")
    _append_flag(command, request.disable_zero_point, "--disable_zero_point")
    _append_flag(command, request.real_quant, "--real_quant")
    _append_flag(command, request.multigpu, "--multigpu")
    _append_flag(command, request.deactive_amp, "--deactive_amp")

    command.extend(request.extra_args)
    return command


def _run_text(command: List[str], cwd: Optional[Path] = None) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            text=True,
            capture_output=True,
            check=False,
        )
        return {
            "command": command,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:
        return {"command": command, "error": str(exc)}


def _version_probe(python_executable: str) -> Dict[str, Any]:
    code = """
import importlib.metadata as metadata
import json
import platform
payload = {"python": platform.python_version()}
for name in ["torch", "transformers", "datasets", "auto-gptq", "auto_gptq", "omniquant"]:
    try:
        payload[name] = metadata.version(name)
    except Exception as exc:
        payload[name] = None
try:
    import torch
    payload["torch_cuda"] = torch.version.cuda
    payload["cuda_available"] = torch.cuda.is_available()
    payload["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
except Exception as exc:
    payload["torch_error"] = str(exc)
print(json.dumps(payload))
"""
    result = _run_text([python_executable, "-c", code])
    try:
        result["parsed"] = json.loads(result.get("stdout") or "{}")
    except Exception:
        result["parsed"] = None
    return result


def _git_commit(upstream_dir: Path) -> Optional[str]:
    result = _run_text(["git", "-C", str(upstream_dir), "rev-parse", "HEAD"])
    if result.get("returncode") == 0 and result.get("stdout"):
        return str(result["stdout"]).strip()
    return None


def build_provenance(request: OmniQuantRequest, command: List[str]) -> Dict[str, Any]:
    detected_commit = _git_commit(request.upstream_dir)
    version_probe = _version_probe(request.python_executable)
    return {
        "method": "omniquant_upstream",
        "upstream_repo_url": UPSTREAM_REPO_URL,
        "upstream_commit_sha": UPSTREAM_COMMIT_SHA,
        "detected_upstream_commit_sha": detected_commit,
        "upstream_commit_matches_pin": detected_commit == UPSTREAM_COMMIT_SHA,
        "environment_name": request.environment_name,
        "adapter_file": str(Path(__file__).resolve()),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "python_executable": request.python_executable,
        "version_probe": version_probe,
        "omniquant_arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(request).items()
            if key != "extra_args"
        },
        "extra_args": list(request.extra_args),
        "command": command,
        "calibration": {
            "source": request.calib_dataset,
            "nsamples": request.nsamples,
        },
        "pretrained_omniquant_parameters_used": request.resume is not None,
        "real_quant": request.real_quant,
        "model_family_support_status": model_support_status(request.model, request.net),
        "fidelity_note": (
            "Adapter invokes pinned upstream OmniQuant main.py. No OmniQuant math is "
            "implemented in SEQ adapter code."
        ),
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def run_omniquant(request: OmniQuantRequest, dry_run: bool = False) -> OmniQuantResult:
    request.output_dir.mkdir(parents=True, exist_ok=True)
    request.cache_dir.mkdir(parents=True, exist_ok=True)
    if request.save_dir is not None:
        request.save_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = request.output_dir / "logs"
    quant_dir = request.output_dir / "quant"
    logs_dir.mkdir(parents=True, exist_ok=True)
    quant_dir.mkdir(parents=True, exist_ok=True)

    command = build_command(request)
    provenance = build_provenance(request, command)
    provenance["dry_run"] = dry_run
    provenance_path = quant_dir / "upstream_provenance.json"
    write_json(provenance_path, provenance)

    stdout_path = logs_dir / "omniquant_stdout.log"
    stderr_path = logs_dir / "omniquant_stderr.log"
    returncode: Optional[int] = None

    if dry_run:
        stdout_path.write_text("dry run: upstream command was not executed\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
    else:
        env = os.environ.copy()
        if request.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = request.cuda_visible_devices
        proc = subprocess.run(
            command,
            cwd=str(request.upstream_dir),
            text=True,
            capture_output=True,
            check=False,
            env=env,
        )
        returncode = proc.returncode
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(
                f"Upstream OmniQuant failed with exit code {proc.returncode}. "
                f"See {stderr_path}"
            )

    result = OmniQuantResult(
        command=command,
        cwd=str(request.upstream_dir),
        output_dir=str(request.output_dir),
        save_dir=str(request.save_dir) if request.save_dir is not None else None,
        provenance_path=str(provenance_path),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        returncode=returncode,
        dry_run=dry_run,
    )
    write_json(request.output_dir / "omniquant_adapter_result.json", asdict(result))
    return result


def _parse_bool_flag(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=name.replace("-", "_"), action="store_true", help=help_text)
    group.add_argument(f"--no-{name}", dest=name.replace("-", "_"), action="store_false")
    parser.set_defaults(**{name.replace("-", "_"): default})


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch pinned upstream OmniQuant through a thin adapter.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--python_executable", default=sys.executable)
    parser.add_argument("--upstream_dir", default=str(DEFAULT_UPSTREAM_DIR))
    parser.add_argument("--cache_dir", default=str(Path.home() / "seq-cache" / "omniquant"))
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--cuda_visible_devices", default=None)
    parser.add_argument("--environment_name", default=ENVIRONMENT_NAME)
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--abits", type=int, default=16)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--calib_dataset", default="wikitext2")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--tasks", default="")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--net", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--act_scales", default=None)
    parser.add_argument("--act_shifts", default=None)
    parser.add_argument("--extra_arg", action="append", default=[], help="Additional raw upstream argument. Repeat as needed.")
    parser.add_argument("--dry_run", action="store_true")

    _parse_bool_flag(parser, "eval-ppl", True, "Pass --eval_ppl to upstream.")
    _parse_bool_flag(parser, "lwc", True, "Pass --lwc to upstream.")
    _parse_bool_flag(parser, "let", False, "Pass --let to upstream.")
    _parse_bool_flag(parser, "aug-loss", False, "Pass --aug_loss to upstream.")
    _parse_bool_flag(parser, "symmetric", False, "Pass --symmetric to upstream.")
    _parse_bool_flag(parser, "disable-zero-point", False, "Pass --disable_zero_point to upstream.")
    _parse_bool_flag(parser, "real-quant", False, "Pass --real_quant to upstream.")
    _parse_bool_flag(parser, "multigpu", False, "Pass --multigpu to upstream.")
    _parse_bool_flag(parser, "deactive-amp", False, "Pass --deactive_amp to upstream.")

    return parser.parse_args(argv)


def request_from_args(args: argparse.Namespace) -> OmniQuantRequest:
    return OmniQuantRequest(
        model=args.model,
        output_dir=Path(args.output_dir).expanduser().resolve(),
        python_executable=args.python_executable,
        upstream_dir=Path(args.upstream_dir).expanduser().resolve(),
        cache_dir=Path(args.cache_dir).expanduser().resolve(),
        save_dir=Path(args.save_dir).expanduser().resolve() if args.save_dir else None,
        cuda_visible_devices=args.cuda_visible_devices,
        environment_name=args.environment_name,
        wbits=args.wbits,
        abits=args.abits,
        group_size=args.group_size,
        alpha=args.alpha,
        epochs=args.epochs,
        calib_dataset=args.calib_dataset,
        nsamples=args.nsamples,
        batch_size=args.batch_size,
        seed=args.seed,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        attn_implementation=args.attn_implementation,
        net=args.net,
        resume=Path(args.resume).expanduser().resolve() if args.resume else None,
        act_scales=Path(args.act_scales).expanduser().resolve() if args.act_scales else None,
        act_shifts=Path(args.act_shifts).expanduser().resolve() if args.act_shifts else None,
        eval_ppl=args.eval_ppl,
        lwc=args.lwc,
        let=args.let,
        aug_loss=args.aug_loss,
        symmetric=args.symmetric,
        disable_zero_point=args.disable_zero_point,
        real_quant=args.real_quant,
        multigpu=args.multigpu,
        deactive_amp=args.deactive_amp,
        extra_args=list(args.extra_arg),
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    result = run_omniquant(request_from_args(args), dry_run=args.dry_run)
    print(json.dumps(asdict(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
