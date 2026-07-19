#!/usr/bin/env python3
"""Audit the environment used by the final SEQ pipeline.

The report is JSON by default and is intentionally useful on both a local CPU
checkout and the GPU/LightCompress machine.  Core packages are required;
quantizer backends and lm-eval are reported as missing so a dry-run can still
be performed without pretending that model experiments ran.
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def _version(name: str):
    try:
        return metadata.version(name)
    except Exception:
        return None


def _git(root: Path):
    def run(args):
        p = subprocess.run(["git", *args], cwd=root, text=True,
                           capture_output=True, check=False)
        return p.stdout.strip() if p.returncode == 0 else None
    return {"branch": run(["branch", "--show-current"]), "commit": run(["rev-parse", "HEAD"]),
            "status": run(["status", "--short"])}


def _path_info(path: Path):
    return {"path": str(path), "exists": path.exists(), "writable": os.access(path, os.W_OK)
            if path.exists() else path.parent.exists() and os.access(path.parent, os.W_OK)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=Path(__file__).resolve().parents[1], type=Path)
    ap.add_argument("--llmc-repo", default=os.environ.get("LLMC_REPO", ""))
    ap.add_argument("--llmc-venv", default=os.environ.get("LLMC_VENV", ""))
    ap.add_argument("--output", default="")
    args = ap.parse_args()
    root = args.root.resolve()
    packages = ["torch", "transformers", "accelerate", "pyyaml", "datasets",
                "bitsandbytes", "hqq", "lm-eval", "pytest"]
    modules = {"torch": "torch", "transformers": "transformers", "accelerate": "accelerate",
               "datasets": "datasets", "bitsandbytes": "bitsandbytes", "hqq": "hqq",
               "lm_eval": "lm_eval", "yaml": "yaml", "pytest": "pytest"}
    report = {
        "python": sys.version,
        "python_executable": sys.executable,
        "packages": {p: _version(p) for p in packages},
        "modules_available": {k: bool(importlib.util.find_spec(v)) for k, v in modules.items()},
        "cuda": {}, "gpu": [],
        "paths": {"root": _path_info(root), "runs": _path_info(root / "runs"),
                  "results": _path_info(root / "results"), "docs": _path_info(root / "docs")},
        "lightcompress": {"repo": _path_info(Path(args.llmc_repo).resolve()) if args.llmc_repo else None,
                          "venv": _path_info(Path(args.llmc_venv).resolve()) if args.llmc_venv else None},
        "huggingface_cache": os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE"),
        "git": _git(root),
    }
    try:
        import torch
        report["cuda"] = {"available": bool(torch.cuda.is_available()),
                           "runtime": torch.version.cuda,
                           "device_count": int(torch.cuda.device_count())}
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            report["gpu"].append({"index": i, "name": props.name,
                                  "memory_gb": round(props.total_memory / 2**30, 2)})
    except Exception as exc:
        report["cuda"] = {"available": False, "error": str(exc)}
    required = ["torch", "transformers", "accelerate", "yaml"]
    missing = [m for m in required if not report["modules_available"].get(m, False)]
    report["required_missing"] = missing
    report["optional_missing"] = [m for m, ok in report["modules_available"].items()
                                   if not ok and m not in required]
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
