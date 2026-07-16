#!/usr/bin/env python3
"""Run comparable LLMC AWQ/GPTQ W4A16 g128 perplexity baselines.

The quantization algorithms and their method-specific calibration settings come
from the installed LightCompress (LLMC) checkout.  This script only renders a
model-specific config, runs LLMC, parses its WikiText-2 perplexity, and writes a
baseline file consumable by ``analysis/build_comparison.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


MODEL = "meta-llama/Llama-3.2-1B"
METHOD_CONFIGS = {
    "awq": Path("configs/quantization/methods/Awq/awq_w_only.yml"),
    "gptq": Path("configs/quantization/methods/GPTQ/gptq_w_only.yml"),
}
METHOD_LABELS = {"awq": "AWQ-4 g128", "gptq": "GPTQ-4 g128"}
PPL_RE = re.compile(r"EVAL:\s+ppl\s+on\s+(?P<dataset>\S+)\s+is\s+(?P<ppl>[0-9.eE+-]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--model-type", default="Llama", help="LLMC model registry name")
    parser.add_argument(
        "--llmc-repo",
        type=Path,
        default=Path(os.environ.get("LLMC_REPO", "/mnt/d/LightCompress")),
    )
    parser.add_argument("--llmc-venv", type=Path, default=None)
    parser.add_argument("--methods", default="awq,gptq", help="Comma-separated: awq,gptq")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/llmc_w4_baselines/Llama-3.2-1B"),
    )
    parser.add_argument("--eval-seq-len", type=int, default=2048)
    parser.add_argument("--fp16-ppl", type=float, default=None)
    parser.add_argument("--hqq-ppl", type=float, default=None)
    parser.add_argument("--force", action="store_true", help="Rerun methods with completed summaries")
    parser.add_argument("--dry-run", action="store_true", help="Render configs without launching LLMC")
    return parser.parse_args()


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def _git_commit(repo: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None if result.returncode == 0 else None


def render_config(
    *, method: str, model: str, model_type: str, llmc_repo: Path, out_path: Path, eval_seq_len: int
) -> dict[str, Any]:
    source_path = llmc_repo / METHOD_CONFIGS[method]
    if not source_path.is_file():
        raise FileNotFoundError(f"LLMC native {method.upper()} config not found: {source_path}")
    config = _read_yaml(source_path)

    weight = config.get("quant", {}).get("weight", {})
    expected_method = {"awq": "Awq", "gptq": "GPTQ"}[method]
    if config.get("quant", {}).get("method") != expected_method:
        raise ValueError(f"Unexpected method in {source_path}: {config.get('quant', {}).get('method')}")
    if weight.get("bit") != 4 or weight.get("group_size") != 128:
        raise ValueError(f"Expected native W4 g128 config, found bit={weight.get('bit')} g={weight.get('group_size')}")

    config.setdefault("base", {})["seed"] = 42
    config.setdefault("model", {}).update(
        {"type": model_type, "path": model, "torch_dtype": "auto", "tokenizer_mode": "slow"}
    )
    # Retain native calibration dataset, sample count, sequence length, batch
    # size, and preprocessor. Only switch from a pre-downloaded path to the
    # datasets-backed loader so this run is self-contained.
    config.setdefault("calib", {})["download"] = True
    config["calib"].pop("path", None)
    config.setdefault("eval", {}).update(
        {
            "eval_pos": ["fake_quant"],
            "name": "wikitext2",
            "download": True,
            "bs": 1,
            "seq_len": eval_seq_len,
            "inference_per_block": False,
        }
    )
    config["eval"].pop("path", None)
    config["save"] = {
        "save_trans": False,
        "save_fake": False,
        "save_vllm": False,
        "save_path": str((out_path.parent / "artifacts").resolve()),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return config


def parse_ppl(log_path: Path) -> tuple[str | None, float | None]:
    matches = list(PPL_RE.finditer(log_path.read_text(encoding="utf-8", errors="replace")))
    if not matches:
        return None, None
    match = matches[-1]
    return match.group("dataset"), float(match.group("ppl"))


def run_method(
    *, method: str, args: argparse.Namespace, llmc_venv: Path, llmc_commit: str | None
) -> dict[str, Any]:
    method_dir = args.out_dir / method
    summary_path = method_dir / "summary.json"
    if summary_path.is_file() and not args.force:
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        if prior.get("status") == "completed" and prior.get("ppl") is not None:
            print(f"Reusing completed {method.upper()} result: PPL {prior['ppl']}", flush=True)
            return prior

    config_path = method_dir / "config.yml"
    config = render_config(
        method=method,
        model=args.model,
        model_type=args.model_type,
        llmc_repo=args.llmc_repo,
        out_path=config_path,
        eval_seq_len=args.eval_seq_len,
    )
    torchrun = llmc_venv / "bin" / "torchrun"
    if not torchrun.is_file():
        raise FileNotFoundError(f"LLMC torchrun not found: {torchrun}")
    command = [
        str(torchrun),
        "--standalone",
        "--nproc_per_node=1",
        # Module mode avoids putting ``llmc/`` itself on sys.path.  That is
        # required on Python 3.14, whose stdlib ``gzip`` imports a top-level
        # ``compression`` package that would otherwise collide with
        # ``llmc/compression``.
        "--module",
        "scripts.llmc_ppl_entrypoint",
        "--config",
        str(config_path.resolve()),
        "--task_id",
        f"{method}_w4_g128_llama32_1b",
    ]
    method_dir.mkdir(parents=True, exist_ok=True)
    (method_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")

    calib = config["calib"]
    result: dict[str, Any] = {
        "model": args.model,
        "method": METHOD_LABELS[method],
        "bits": 4.0,
        "status": "dry_run" if args.dry_run else "running",
        "ppl": None,
        "ppl_dataset": "wikitext2",
        "eval_seq_len": args.eval_seq_len,
        "calibration": {
            key: calib.get(key) for key in ("name", "n_samples", "bs", "seq_len", "preproc", "seed")
        },
        "llmc_config_source": str(METHOD_CONFIGS[method]),
        "rendered_config": str(config_path),
        "llmc_repo": str(args.llmc_repo),
        "llmc_commit": llmc_commit,
        "command": command,
    }
    _write_json(summary_path, result)
    if args.dry_run:
        print(f"Rendered {method.upper()}: {config_path}", flush=True)
        return result

    log_path = method_dir / "llmc.log"
    env = os.environ.copy()
    workspace = Path(__file__).resolve().parent.parent
    env["PYTHONPATH"] = os.pathsep.join(
        [str(workspace), str(args.llmc_repo), env.get("PYTHONPATH", "")]
    )
    started = time.time()
    print(f"Running native LLMC {method.upper()} W4 g128...", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=args.llmc_repo,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        returncode = process.wait()

    dataset, ppl = parse_ppl(log_path)
    result.update(
        {
            "status": "completed" if returncode == 0 and ppl is not None else "failed",
            "ppl": ppl,
            "ppl_dataset": dataset or "wikitext2",
            "returncode": returncode,
            "duration_sec": round(time.time() - started, 3),
            "log": str(log_path),
        }
    )
    if returncode != 0:
        result["reason"] = f"LLMC exited with status {returncode}"
    elif ppl is None:
        result["reason"] = "LLMC completed but no perplexity was found in its log"
    _write_json(summary_path, result)
    return result


def write_outputs(args: argparse.Namespace, results: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    if args.fp16_ppl is not None:
        rows.append({"method": "FP16", "bits": 16.0, "ppl": args.fp16_ppl})
    if args.hqq_ppl is not None:
        rows.append({"method": "HQQ-4 uniform", "bits": 4.0, "ppl": args.hqq_ppl})
    rows.extend(
        {"method": result["method"], "bits": 4.0, "ppl": result["ppl"]}
        for result in results
        if result.get("status") == "completed" and result.get("ppl") is not None
    )
    _write_json(args.out_dir / "baselines.json", {args.model: rows})
    _write_json(args.out_dir / "run_summary.json", {"model": args.model, "results": results})

    lines = [f"# LLMC W4A16 g128 baselines — {args.model}", "", "| method | bits | PPL | status |", "|---|---:|---:|---|"]
    for row in rows:
        lines.append(f"| {row['method']} | {row['bits']:.1f} | {row['ppl']:.4f} | completed |")
    for result in results:
        if result.get("status") != "completed":
            lines.append(f"| {result['method']} | 4.0 | — | {result['status']} |")
    lines.extend(
        [
            "",
            f"Evaluation: full WikiText-2 test corpus in non-overlapping {args.eval_seq_len}-token chunks.",
            "AWQ and GPTQ algorithm/calibration settings are taken from the installed LLMC native configs.",
            "",
        ]
    )
    (args.out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    methods = [method.strip().lower() for method in args.methods.split(",") if method.strip()]
    unknown = sorted(set(methods) - set(METHOD_CONFIGS))
    if unknown:
        raise SystemExit(f"Unsupported methods: {', '.join(unknown)}")
    args.llmc_repo = args.llmc_repo.resolve()
    llmc_venv = (args.llmc_venv or args.llmc_repo / ".venv-llmc").resolve()
    if not (args.llmc_repo / "llmc" / "__main__.py").is_file():
        raise SystemExit(f"Not an LLMC checkout: {args.llmc_repo}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    commit = _git_commit(args.llmc_repo)
    results = [run_method(method=method, args=args, llmc_venv=llmc_venv, llmc_commit=commit) for method in methods]
    write_outputs(args, results)
    return 0 if all(result.get("status") in {"completed", "dry_run"} for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
