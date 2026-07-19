#!/usr/bin/env python3
"""Fail-closed environment preflight and reproducibility manifest writer."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


PINNED_LLMC_COMMIT = "86f564ddb1d6548b228c67a10509a4ed7264345c"


def _run(command: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{result.stderr}")
    return result.stdout.strip()


def _freeze(python: Path | str) -> str:
    """Capture an installed-package freeze even when a uv venv has no pip module."""
    code = (
        "from importlib.metadata import distributions; "
        "rows=sorted((d.metadata.get('Name') or '').lower()+'=='+d.version "
        "for d in distributions() if d.metadata.get('Name')); print('\\n'.join(rows))"
    )
    return _run([str(python), "-c", code])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--llmc-repo", type=Path, required=True)
    parser.add_argument("--llmc-venv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-hf", action="store_true", help="test-only: skip authenticated model lookup")
    args = parser.parse_args()
    root, llmc_repo, llmc_venv = args.root.resolve(), args.llmc_repo.resolve(), args.llmc_venv.resolve()
    for label, path in (("LightCompress checkout", llmc_repo), ("LightCompress venv", llmc_venv)):
        if any(char.isspace() for char in str(path)):
            raise SystemExit(f"{label} path must not contain whitespace: {path}")
    if not (llmc_repo / "llmc" / "__main__.py").is_file():
        raise SystemExit(f"invalid LightCompress checkout: {llmc_repo}")
    llmc_python = llmc_venv / "bin" / "python"
    if not llmc_python.is_file():
        raise SystemExit(f"LightCompress Python is missing: {llmc_python}")
    missing_modules = [name for name in ("hqq", "pytest") if importlib.util.find_spec(name) is None]
    if missing_modules:
        raise SystemExit("SEQ environment is missing required modules: " + ", ".join(missing_modules))
    llmc_commit = _run(["git", "rev-parse", "HEAD"], llmc_repo)
    if llmc_commit != PINNED_LLMC_COMMIT:
        raise SystemExit(f"LightCompress commit mismatch: expected {PINNED_LLMC_COMMIT}, found {llmc_commit}")
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    model_revisions: dict[str, str] = {}
    hf_auth_source = "skipped"
    if not args.skip_hf:
        from huggingface_hub import HfApi, get_token

        token = os.environ.get("HF_TOKEN") or get_token()
        if not token:
            raise SystemExit(
                "Hugging Face authentication is required: export HF_TOKEN or run `hf auth login`"
            )
        hf_auth_source = "environment" if os.environ.get("HF_TOKEN") else "cached_login"
        api = HfApi(token=token)
        for model in matrix["models"]:
            model_revisions[model] = api.model_info(model).sha
    try:
        import torch
        cuda = {"available": bool(torch.cuda.is_available()), "runtime": torch.version.cuda,
                "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]}
    except Exception as exc:  # noqa: BLE001
        cuda = {"available": False, "error": str(exc)}
    if not cuda.get("available"):
        raise SystemExit("CUDA is unavailable in the SEQ environment")
    report = {
        "status": "PASS",
        "root": str(root),
        "seq_git_commit": _run(["git", "rev-parse", "HEAD"], root),
        "seq_git_status": _run(["git", "status", "--short"], root),
        "seq_python": sys.executable,
        "seq_freeze": _freeze(sys.executable),
        "llmc_repo": str(llmc_repo),
        "llmc_venv": str(llmc_venv),
        "llmc_commit": llmc_commit,
        "llmc_freeze": _freeze(llmc_python),
        "matrix": matrix,
        "matrix_sha256": hashlib.sha256(args.matrix.read_bytes()).hexdigest(),
        "huggingface_auth_source": hf_auth_source,
        "model_revisions": model_revisions,
        "cuda": cuda,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
