from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from third_party_quant.adapters.llmc_adapter import (
    LLMCRunSpec,
    get_llmc_commit,
    run_llmc,
    write_json,
)


THIRD_PARTY_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = THIRD_PARTY_DIR / "llmc_templates"
SUPPORTED_LLMC_METHODS = {
    "gptq_llmc",
    "smoothquant_llmc",
    "awq_llmc",
    "rtn_llmc",
    "llm_int8_llmc",
    "spinquant_llmc",
    "omniquant_llmc",
}
PHASE2_METHODS: set[str] = set()
OMNIQUANT_DISABLED_REASON = "LLMC OmniQuant upstream flow not safely resolved."
SPINQUANT_DISABLED_REASON = "LLMC SpinQuant is not yet safely enabled in the SEQ integration."


def _infer_model_type(model: str) -> Optional[str]:
    lowered = model.strip().lower()
    if "facebook/opt-" in lowered or lowered.startswith("opt") or "/opt-" in lowered:
        return "Opt"
    if lowered.startswith("qwen/") or lowered.startswith("qwen2") or "qwen2" in lowered:
        return "Qwen2"
    if lowered.startswith("meta-llama/llama") or "tinyllama" in lowered or "llama" in lowered:
        return "Llama"
    return None


def _method_spec(method: str) -> Dict[str, Any]:
    if method == "gptq_llmc":
        return {
            "template_path": TEMPLATE_DIR / "gptq_w4a16.yml",
            "calib_preproc": "wikitext2_gptq",
        }
    if method == "smoothquant_llmc":
        return {
            "template_path": TEMPLATE_DIR / "smoothquant_w8a8.yml",
            "calib_preproc": "txt_general_preproc",
        }
    if method == "awq_llmc":
        return {
            "template_path": TEMPLATE_DIR / "awq_w4a16.yml",
            "calib_preproc": "txt_general_preproc",
        }
    if method == "rtn_llmc":
        return {
            "template_path": TEMPLATE_DIR / "rtn_w8a16.yml",
            "calib_preproc": "txt_general_preproc",
            "notes": [
                "RTN template follows the upstream W8A16 recommendation for a simple no-calibration baseline.",
            ],
        }
    if method == "llm_int8_llmc":
        return {
            "template_path": TEMPLATE_DIR / "llm_int8_w8a8.yml",
            "calib_preproc": "txt_general_preproc",
            "notes": [
                "LlmInt8 template preserves the upstream LlmInt8 method name and threshold setting.",
            ],
        }
    if method == "omniquant_llmc":
        return {
            "mode": "two_step_awq_omniquant",
            "step1_template_path": TEMPLATE_DIR / "omniquant_step1_awq_w4a16.yml",
            "step2_template_path": TEMPLATE_DIR / "omniquant_step2_omniquant_w4a16.yml",
            "notes": [
                "OmniQuant uses the upstream AWQ + OmniQuant two-step weight-only flow.",
                "Final PPL comes from step 2 (OmniQuant).",
            ],
        }
    raise ValueError(f"Unsupported LLMC method: {method}")


def _safe_name(text: str, max_len: int = 64) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return safe[:max_len] if len(safe) > max_len else safe


def _save_flags(save_mode: str) -> Dict[str, Any]:
    if save_mode == "none":
        return {"save_trans": False, "save_fake": False, "save_vllm": False, "backend": "llmc_none"}
    if save_mode == "fake":
        return {"save_trans": False, "save_fake": True, "save_vllm": False, "backend": "llmc_fake"}
    if save_mode == "trans":
        return {"save_trans": True, "save_fake": False, "save_vllm": False, "backend": "llmc_transformed"}
    raise ValueError(f"Unsupported LLMC save mode: {save_mode}")


def _lm_eval_skipped(tasks: List[str]) -> Dict[str, Any]:
    return {
        "status": "skipped",
        "reason": "phase_1_llmc_log_ppl_only",
        "backend": "hf",
        "tasks": tasks,
        "num_fewshot": 0,
        "limit": None,
        "requested": True,
        "results": {},
        "fail_policy": "warn",
        "flat": {
            "lm_eval__status": "skipped",
            "lm_eval__tasks": ",".join(tasks),
        },
    }


def _write_terminal_summary(method_dir: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    write_json(method_dir / "summary.json", payload)
    return payload


def _disabled_method_summary(
    *,
    method_dir: Path,
    model: str,
    method: str,
    reason: str,
    notes: List[str],
    eval_seq_len: int,
    llmc_repo: Path,
    llmc_venv: Optional[Path],
    dry_run: bool,
    lm_eval_tasks: Optional[List[str]],
) -> Dict[str, Any]:
    llmc_commit = get_llmc_commit(llmc_repo)
    return _write_terminal_summary(
        method_dir,
        {
            "model": model,
            "method": method,
            "status": "not_implemented",
            "compare_status": "not_implemented",
            "reason": reason,
            "ppl": None,
            "ppl_source": None,
            "ppl_dataset": None,
            "ppl_seq_len": eval_seq_len,
            "duration_sec": None,
            "quant_disk_bytes": 0,
            "quant_disk_gb": 0.0,
            "artifact_kind": None,
            "artifact_path": None,
            "backend": "llmc_none",
            "run_dir": str(method_dir),
            "llmc_config_path": None,
            "llmc_log_path": None,
            "llmc_save_path": None,
            "llmc_task_id": None,
            "llmc_returncode": None,
            "model_path": None,
            "notes": notes,
            "lm_eval": _lm_eval_skipped(lm_eval_tasks or []),
            "adapter_summary_path": None,
            "llmc_repo": str(llmc_repo),
            "llmc_commit": llmc_commit,
            "llmc_venv": str(llmc_venv) if llmc_venv is not None else None,
            "dry_run": dry_run,
        },
    )


def _omniquant_not_implemented_summary(
    *,
    method_dir: Path,
    model: str,
    method: str,
    eval_seq_len: int,
    llmc_repo: Path,
    llmc_venv: Optional[Path],
    dry_run: bool,
    lm_eval_tasks: Optional[List[str]],
    requested_model_type: Optional[str],
) -> Dict[str, Any]:
    notes = [
        "Disabled after recovery audit on 2026-05-12.",
        "Upstream LLMC documents OmniQuant as a two-step AWQ -> OmniQuant flow.",
    ]
    if requested_model_type == "Opt":
        notes.extend(
            [
                "The audited OPT smoke completed AWQ step 1 and saved transformed_model successfully.",
                (
                    "The audited OmniQuant step 2 then failed on LightCompress commit "
                    f"{llmc_commit or 'unknown'} with "
                    "ValueError: not enough values to unpack (expected 3, got 2) "
                    "from transformers.models.opt.modeling_opt while processing OPTDecoderLayer."
                ),
            ]
        )
    elif requested_model_type == "Llama":
        notes.extend(
            [
                "The audited TinyLlama smoke completed AWQ step 1 and saved transformed_model successfully.",
                (
                    "The audited OmniQuant step 2 then failed on LightCompress commit "
                    f"{llmc_commit or 'unknown'} with "
                    "TypeError: cannot unpack non-iterable NoneType object "
                    "from transformers.models.llama.modeling_llama when Llama attention "
                    "expected position_embeddings."
                ),
            ]
        )
    else:
        notes.extend(
            [
                "Known upstream failures were reproduced on both OPT and Llama-family smoke tests.",
                (
                    "OPT failed in transformers.models.opt.modeling_opt with "
                    "ValueError: not enough values to unpack (expected 3, got 2)."
                ),
                (
                    "TinyLlama failed in transformers.models.llama.modeling_llama with "
                    "TypeError: cannot unpack non-iterable NoneType object."
                ),
            ]
        )
    notes.extend(
        [
            "No safe SEQ-side fix was identified without patching upstream LLMC behavior.",
            "Re-enable only after an upstream fix or a separately validated LLMC patch lands.",
        ]
    )
    return _disabled_method_summary(
        method_dir=method_dir,
        model=model,
        method=method,
        reason=OMNIQUANT_DISABLED_REASON,
        notes=notes,
        eval_seq_len=eval_seq_len,
        llmc_repo=llmc_repo,
        llmc_venv=llmc_venv,
        dry_run=dry_run,
        lm_eval_tasks=lm_eval_tasks,
    )


def _spinquant_not_implemented_summary(
    *,
    method_dir: Path,
    model: str,
    method: str,
    eval_seq_len: int,
    llmc_repo: Path,
    llmc_venv: Optional[Path],
    dry_run: bool,
    lm_eval_tasks: Optional[List[str]],
) -> Dict[str, Any]:
    notes = [
        "Audited on 2026-05-12 while extending LLMC external baselines.",
        "The stable LightCompress main checkout used by SEQ does not provide a SpinQuant method config or SpinQuant algorithm class.",
        "A separate experimental LightCompress dev_spinquant branch was inspected in isolation.",
        "That experimental branch adds configs/quantization/SpinQuant/spinquant_w4a4.yml and llmc/compression/quantization/spinquant.py.",
        "The experimental branch requires fast_hadamard_transform for its Hadamard rotation path.",
        "After switching the isolated experimental venv to torch 2.11.0+cu126, fast_hadamard_transform installed and imported successfully.",
        "A direct TinyLlama smoke on the experimental branch then reached model loading and calibration setup, but failed on first CUDA execution with torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device.",
        "PyTorch warned that the RTX 5090 SM120 device is not supported by the current cu126 build, which only lists support through sm_90 and recommends newer 12.8 or 13.0 builds.",
        "A prior direct Opt smoke on the experimental branch also failed in SpinQuant setup with AttributeError: OPTConfig has no attribute intermediate_size.",
        "Keeping spinquant_llmc recognized but disabled avoids routing users into an unvalidated branch-specific path.",
        "Revisit only after a separate SpinQuant LLMC checkout and venv are validated end to end for the target model family.",
    ]
    return _disabled_method_summary(
        method_dir=method_dir,
        model=model,
        method=method,
        reason=SPINQUANT_DISABLED_REASON,
        notes=notes,
        eval_seq_len=eval_seq_len,
        llmc_repo=llmc_repo,
        llmc_venv=llmc_venv,
        dry_run=dry_run,
        lm_eval_tasks=lm_eval_tasks,
    )


def _summary_from_adapter(
    *,
    method_dir: Path,
    model: str,
    method: str,
    adapter_summary: Dict[str, Any],
    eval_seq_len: int,
    backend: str,
    task_id: str,
    rendered_config_path: Path,
    log_filename: str = "combined.log",
    notes: Optional[List[str]] = None,
    lm_eval_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "method": method,
        "status": adapter_summary.get("status"),
        "compare_status": adapter_summary.get("status"),
        "reason": adapter_summary.get("reason"),
        "ppl": adapter_summary.get("ppl"),
        "ppl_source": "llmc_log" if adapter_summary.get("ppl") is not None else None,
        "ppl_dataset": adapter_summary.get("ppl_dataset"),
        "ppl_seq_len": eval_seq_len,
        "duration_sec": adapter_summary.get("duration_sec"),
        "quant_disk_bytes": adapter_summary.get("quant_disk_bytes"),
        "quant_disk_gb": adapter_summary.get("quant_disk_gb"),
        "artifact_kind": adapter_summary.get("artifact_kind"),
        "artifact_path": adapter_summary.get("artifact_path"),
        "backend": backend,
        "run_dir": str(method_dir),
        "llmc_config_path": str(rendered_config_path),
        "llmc_log_path": str(method_dir / "logs" / log_filename),
        "llmc_save_path": adapter_summary.get("save_path"),
        "llmc_task_id": task_id,
        "llmc_returncode": adapter_summary.get("returncode"),
        "model_path": adapter_summary.get("artifact_path") if adapter_summary.get("hf_loadable_candidate") else None,
        "notes": list(notes or []),
        "lm_eval": lm_eval_summary or _lm_eval_skipped([]),
        "adapter_summary_path": str(method_dir / "llmc_adapter_summary.json"),
        "llmc_repo": adapter_summary.get("llmc_repo"),
        "llmc_commit": adapter_summary.get("llmc_commit"),
        "llmc_venv": adapter_summary.get("llmc_venv"),
        "dry_run": adapter_summary.get("dry_run"),
    }
    return _write_terminal_summary(method_dir, payload)


def run_llmc_baseline(
    *,
    method: str,
    model: str,
    run_dir: Path,
    llmc_repo: Path,
    llmc_venv: Optional[Path],
    dry_run: bool,
    save_mode: str,
    calib_samples: int,
    calib_seq_len: int,
    eval_seq_len: int,
    eval_dataset: str,
    calib_dataset: str,
    seed: int,
    model_type: Optional[str],
    tokenizer_mode: str,
    inference_per_block: bool,
    lm_eval_tasks: Optional[List[str]] = None,
    torch_dtype: str = "auto",
) -> Dict[str, Any]:
    method_dir = run_dir.resolve()
    method_dir.mkdir(parents=True, exist_ok=True)
    lm_eval_summary = _lm_eval_skipped(lm_eval_tasks or [])

    if method in PHASE2_METHODS:
        return _write_terminal_summary(
            method_dir,
            {
                "model": model,
                "method": method,
                "status": "not_implemented",
                "compare_status": "not_implemented",
                "reason": f"{method} is reserved for phase 2.",
                "ppl": None,
                "ppl_source": None,
                "ppl_dataset": None,
                "ppl_seq_len": eval_seq_len,
                "duration_sec": None,
                "quant_disk_bytes": 0,
                "quant_disk_gb": 0.0,
                "artifact_kind": None,
                "artifact_path": None,
                "backend": "llmc_none",
                "run_dir": str(method_dir),
                "llmc_config_path": None,
                "llmc_log_path": None,
                "llmc_save_path": None,
                "llmc_task_id": None,
                "llmc_returncode": None,
                "model_path": None,
                "notes": ["phase_2_method_not_enabled"],
                "lm_eval": lm_eval_summary,
            },
        )

    if method not in SUPPORTED_LLMC_METHODS:
        raise ValueError(f"Unsupported LLMC method: {method}")

    if method == "omniquant_llmc":
        requested_model_type = model_type or _infer_model_type(model)
        return _omniquant_not_implemented_summary(
            method_dir=method_dir,
            model=model,
            method=method,
            eval_seq_len=eval_seq_len,
            llmc_repo=llmc_repo,
            llmc_venv=llmc_venv,
            dry_run=dry_run,
            lm_eval_tasks=lm_eval_tasks,
            requested_model_type=requested_model_type,
        )
    if method == "spinquant_llmc":
        return _spinquant_not_implemented_summary(
            method_dir=method_dir,
            model=model,
            method=method,
            eval_seq_len=eval_seq_len,
            llmc_repo=llmc_repo,
            llmc_venv=llmc_venv,
            dry_run=dry_run,
            lm_eval_tasks=lm_eval_tasks,
        )

    inferred_model_type = model_type or _infer_model_type(model)
    if not inferred_model_type:
        return _write_terminal_summary(
            method_dir,
            {
                "model": model,
                "method": method,
                "status": "failed",
                "compare_status": "failed",
                "reason": f"Could not infer LLMC model.type for {model}. Pass --llmc_model_type.",
                "ppl": None,
                "ppl_source": None,
                "ppl_dataset": None,
                "ppl_seq_len": eval_seq_len,
                "duration_sec": None,
                "quant_disk_bytes": 0,
                "quant_disk_gb": 0.0,
                "artifact_kind": None,
                "artifact_path": None,
                "backend": "llmc_none",
                "run_dir": str(method_dir),
                "llmc_config_path": None,
                "llmc_log_path": None,
                "llmc_save_path": None,
                "llmc_task_id": None,
                "llmc_returncode": None,
                "model_path": None,
                "notes": ["model_type_inference_failed"],
                "lm_eval": lm_eval_summary,
            },
        )

    method_cfg = _method_spec(method)
    method_mode = method_cfg.get("mode", "single_step")
    save_cfg = _save_flags(save_mode)
    task_id = f"{_safe_name(method)}__{_safe_name(model.replace('/', '_'))}"
    rendered_config_path = method_dir / "rendered_config.yml"
    save_path_host = method_dir / f"llmc_save_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    if method_mode == "two_step_awq_omniquant":
        step1_rendered_config_path = method_dir / "rendered_config_step1_awq.yml"
        step2_rendered_config_path = method_dir / "rendered_config_step2_omniquant.yml"
        step1_save_path_host = method_dir / f"llmc_save_step1_awq_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        step2_save_path_host = method_dir / f"llmc_save_step2_omniquant_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        step1_summary = run_llmc(
            LLMCRunSpec(
                method=method,
                model=model,
                task_id=f"{task_id}__step1_awq",
                llmc_repo=llmc_repo,
                llmc_venv=llmc_venv,
                template_path=method_cfg["step1_template_path"],
                method_run_dir=method_dir,
                rendered_config_path=step1_rendered_config_path,
                model_type=inferred_model_type,
                torch_dtype=torch_dtype,
                tokenizer_mode=tokenizer_mode,
                calib_name=calib_dataset,
                calib_download=True,
                calib_n_samples=calib_samples,
                calib_bs=1,
                calib_seq_len=calib_seq_len,
                calib_preproc="txt_general_preproc",
                calib_seed=seed,
                eval_pos=["fake_quant"],
                eval_name=eval_dataset,
                eval_download=True,
                eval_bs=1,
                eval_seq_len=eval_seq_len,
                inference_per_block=inference_per_block,
                save_trans=True,
                save_fake=False,
                save_vllm=False,
                save_path_host=step1_save_path_host,
                dry_run=dry_run,
                log_filename="combined_step1_awq.log",
                command_filename="llmc_command_step1_awq.sh",
                adapter_summary_filename="llmc_adapter_summary_step1_awq.json",
            )
        )

        step_summaries: Dict[str, Any] = {"step1_awq": step1_summary}
        notes = list(step1_summary.get("notes") or [])
        notes.extend(method_cfg.get("notes") or [])
        notes.append("PPL from LLMC log; runtime metrics not comparable in phase 1")

        if step1_summary.get("status") == "failed":
            write_json(method_dir / "llmc_adapter_summary.json", {"steps": step_summaries})
            return _write_terminal_summary(
                method_dir,
                {
                    "model": model,
                    "method": method,
                    "status": "failed",
                    "compare_status": "failed",
                    "reason": "OmniQuant step 1 (AWQ transform) failed.",
                    "ppl": None,
                    "ppl_source": None,
                    "ppl_dataset": None,
                    "ppl_seq_len": eval_seq_len,
                    "duration_sec": None,
                    "quant_disk_bytes": 0,
                    "quant_disk_gb": 0.0,
                    "artifact_kind": None,
                    "artifact_path": None,
                    "backend": save_cfg["backend"],
                    "run_dir": str(method_dir),
                    "llmc_config_path": str(step1_rendered_config_path),
                    "llmc_log_path": str(method_dir / "logs" / "combined_step1_awq.log"),
                    "llmc_save_path": step1_summary.get("save_path"),
                    "llmc_task_id": f"{task_id}__step1_awq",
                    "llmc_returncode": step1_summary.get("returncode"),
                    "model_path": None,
                    "notes": notes,
                    "lm_eval": lm_eval_summary,
                    "adapter_summary_path": str(method_dir / "llmc_adapter_summary.json"),
                    "llmc_repo": step1_summary.get("llmc_repo"),
                    "llmc_commit": step1_summary.get("llmc_commit"),
                    "llmc_venv": step1_summary.get("llmc_venv"),
                    "dry_run": step1_summary.get("dry_run"),
                    "step_summaries": step_summaries,
                },
            )

        step2_model_path = (
            str(Path(step1_summary["save_path"]) / "transformed_model")
            if step1_summary.get("save_path")
            else model
        )
        step2_summary = run_llmc(
            LLMCRunSpec(
                method=method,
                model=step2_model_path,
                task_id=f"{task_id}__step2_omniquant",
                llmc_repo=llmc_repo,
                llmc_venv=llmc_venv,
                template_path=method_cfg["step2_template_path"],
                method_run_dir=method_dir,
                rendered_config_path=step2_rendered_config_path,
                model_type=inferred_model_type,
                torch_dtype=torch_dtype,
                tokenizer_mode=tokenizer_mode,
                calib_name=calib_dataset,
                calib_download=True,
                calib_n_samples=calib_samples,
                calib_bs=1,
                calib_seq_len=calib_seq_len,
                calib_preproc="wikitext2_gptq",
                calib_seed=seed,
                eval_pos=["fake_quant"],
                eval_name=eval_dataset,
                eval_download=True,
                eval_bs=1,
                eval_seq_len=eval_seq_len,
                inference_per_block=inference_per_block,
                save_trans=save_cfg["save_trans"],
                save_fake=save_cfg["save_fake"],
                save_vllm=save_cfg["save_vllm"],
                save_path_host=step2_save_path_host,
                dry_run=dry_run,
                log_filename="combined_step2_omniquant.log",
                command_filename="llmc_command_step2_omniquant.sh",
                adapter_summary_filename="llmc_adapter_summary_step2_omniquant.json",
            )
        )
        step_summaries["step2_omniquant"] = step2_summary
        notes.extend(step2_summary.get("notes") or [])
        write_json(method_dir / "llmc_adapter_summary.json", {"steps": step_summaries})

        return _write_terminal_summary(
            method_dir,
            {
                "model": model,
                "method": method,
                "status": step2_summary.get("status"),
                "compare_status": step2_summary.get("status"),
                "reason": step2_summary.get("reason"),
                "ppl": step2_summary.get("ppl"),
                "ppl_source": "llmc_log" if step2_summary.get("ppl") is not None else None,
                "ppl_dataset": step2_summary.get("ppl_dataset"),
                "ppl_seq_len": eval_seq_len,
                "duration_sec": step2_summary.get("duration_sec"),
                "quant_disk_bytes": step2_summary.get("quant_disk_bytes"),
                "quant_disk_gb": step2_summary.get("quant_disk_gb"),
                "artifact_kind": step2_summary.get("artifact_kind"),
                "artifact_path": step2_summary.get("artifact_path"),
                "backend": save_cfg["backend"],
                "run_dir": str(method_dir),
                "llmc_config_path": str(step2_rendered_config_path),
                "llmc_log_path": str(method_dir / "logs" / "combined_step2_omniquant.log"),
                "llmc_save_path": step2_summary.get("save_path"),
                "llmc_task_id": f"{task_id}__step2_omniquant",
                "llmc_returncode": step2_summary.get("returncode"),
                "model_path": step2_summary.get("artifact_path") if step2_summary.get("hf_loadable_candidate") else None,
                "notes": notes,
                "lm_eval": lm_eval_summary,
                "adapter_summary_path": str(method_dir / "llmc_adapter_summary.json"),
                "llmc_repo": step2_summary.get("llmc_repo"),
                "llmc_commit": step2_summary.get("llmc_commit"),
                "llmc_venv": step2_summary.get("llmc_venv"),
                "dry_run": step2_summary.get("dry_run"),
                "step_summaries": step_summaries,
                "intermediate_model_path": step2_model_path,
                "intermediate_save_path": step1_summary.get("save_path"),
            },
        )

    adapter_summary = run_llmc(
        LLMCRunSpec(
            method=method,
            model=model,
            task_id=task_id,
            llmc_repo=llmc_repo,
            llmc_venv=llmc_venv,
            template_path=method_cfg["template_path"],
            method_run_dir=method_dir,
            rendered_config_path=rendered_config_path,
            model_type=inferred_model_type,
            torch_dtype=torch_dtype,
            tokenizer_mode=tokenizer_mode,
            calib_name=calib_dataset,
            calib_download=True,
            calib_n_samples=calib_samples,
            calib_bs=1,
            calib_seq_len=calib_seq_len,
            calib_preproc=method_cfg["calib_preproc"],
            calib_seed=seed,
            eval_pos=["fake_quant"],
            eval_name=eval_dataset,
            eval_download=True,
            eval_bs=1,
            eval_seq_len=eval_seq_len,
            inference_per_block=inference_per_block,
            save_trans=save_cfg["save_trans"],
            save_fake=save_cfg["save_fake"],
            save_vllm=save_cfg["save_vllm"],
            save_path_host=save_path_host,
            dry_run=dry_run,
        )
    )

    notes = list(adapter_summary.get("notes") or [])
    notes.extend(method_cfg.get("notes") or [])
    notes.append("PPL from LLMC log; runtime metrics not comparable in phase 1")
    if adapter_summary.get("dry_run"):
        notes.append("dry_run_no_llmc_execution")

    return _summary_from_adapter(
        method_dir=method_dir,
        model=model,
        method=method,
        adapter_summary=adapter_summary,
        eval_seq_len=eval_seq_len,
        backend=save_cfg["backend"],
        task_id=task_id,
        rendered_config_path=rendered_config_path,
        notes=notes,
        lm_eval_summary=lm_eval_summary,
    )
