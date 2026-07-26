#!/usr/bin/env python3
"""Record the environment that produced the committed results.

Writes `run_manifest.json`: package versions, Python/CUDA/GPU, the git commit, and the
resolved revision of every model used. Run after a results-producing session so each
number in the paper can be traced to an exact environment.

    python scripts/write_run_manifest.py [--out run_manifest.json]
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

PACKAGES = ["torch", "transformers", "tokenizers", "huggingface-hub", "datasets",
            "accelerate", "safetensors", "hqq", "lm_eval", "matplotlib", "numpy"]
MODELS = ["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B",
          "meta-llama/Llama-2-7b-hf", "Qwen/Qwen2.5-3B"]


def _sh(*cmd: str) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("run_manifest.json"))
    ap.add_argument("--models", default=",".join(MODELS))
    args = ap.parse_args()

    import importlib.metadata as md
    packages = {}
    for p in PACKAGES:
        try:
            packages[p] = md.version(p)
        except Exception:
            packages[p] = None

    hardware: dict = {"platform": platform.platform(), "python": sys.version.split()[0]}
    try:
        import torch
        hardware["torch_cuda"] = torch.version.cuda
        hardware["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            hardware["gpus"] = [torch.cuda.get_device_name(i)
                                for i in range(torch.cuda.device_count())]
    except Exception as exc:  # noqa: BLE001
        hardware["torch_error"] = str(exc)

    models = {}
    for name in [m for m in args.models.split(",") if m.strip()]:
        entry: dict = {}
        try:  # resolved commit hash of the checkpoint actually used
            from huggingface_hub import HfApi
            entry["revision"] = HfApi().model_info(name).sha
        except Exception as exc:  # noqa: BLE001
            entry["revision"] = None
            entry["error"] = str(exc)
        models[name.strip()] = entry

    manifest = {
        "git_commit": _sh("git", "rev-parse", "HEAD"),
        "git_dirty": bool(_sh("git", "status", "--porcelain")),
        "hardware": hardware,
        "packages": packages,
        "models": models,
        "evaluation": {"dataset": "wikitext2", "split": "test", "seq_len": 2048,
                       "ppl_mode": "canonical", "chunking": "non_overlapping"},
        "scope": {"skip_lm_head": True, "storage": "theoretical weight-only bits/param",
                  "gptq_base_group_size": 128},
    }
    args.out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
