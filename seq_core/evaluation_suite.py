#!/usr/bin/env python3
import gc
import json
import logging
import os
import random
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .benchmarks import compute_ppl
from .eval_config import resolve_metric_plan
from .metrics_utils import dir_size_bytes, estimate_fp16_weight_bytes
from .multiple_choice_eval import run_mmlu_eval, run_zero_shot_suite
from .seq_lm_eval import run_lm_eval_suite

LOGGER = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _token_repetition_ratio(token_ids: List[int], n: int = 4) -> float:
    if len(token_ids) < n:
        return 0.0
    ngrams = [tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)]
    total = len(ngrams)
    unique = len(set(ngrams))
    if total == 0:
        return 0.0
    return float(1.0 - (unique / total))


def _decode_output(tokenizer, output_ids: torch.Tensor) -> str:
    return tokenizer.decode(output_ids, skip_special_tokens=True)


def _generate(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    device: str,
) -> Tuple[str, int, int]:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "remove_invalid_values": True,
            "renormalize_logits": True,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        output = model.generate(**inputs, **gen_kwargs)
    output_ids = output[0]
    gen_len = int(output_ids.shape[0] - input_len)
    gen_ids = output_ids[input_len:]
    return _decode_output(tokenizer, gen_ids), input_len, gen_len


def tail_risk_test(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[Dict[str, Any]],
    out_path: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
    repetition_threshold: float = 0.2,
) -> Dict[str, Any]:
    results = []
    matches = 0
    match_total = 0
    trunc_hits = 0
    rep_hits = 0

    for entry in prompts:
        prompt = entry["prompt"]
        expected = entry.get("expected")
        output, input_len, gen_len = _generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            device=device,
        )
        tokens = tokenizer(output, add_special_tokens=False)["input_ids"]
        rep_ratio = _token_repetition_ratio(tokens, n=4)
        is_trunc = gen_len >= max_new_tokens
        if is_trunc:
            trunc_hits += 1
        if rep_ratio >= repetition_threshold:
            rep_hits += 1

        exact_match = None
        if expected is not None:
            match_total += 1
            exact_match = expected in output
            if exact_match:
                matches += 1

        results.append(
            {
                "prompt": prompt,
                "expected": expected,
                "output": output,
                "gen_len": gen_len,
                "repetition_ratio": rep_ratio,
                "truncated": is_trunc,
                "exact_span_match": exact_match,
            }
        )

    summary = {
        "exact_span_match_rate": (matches / match_total) if match_total else None,
        "repetition_rate": rep_hits / max(len(results), 1),
        "truncation_rate": trunc_hits / max(len(results), 1),
        "repetition_threshold": repetition_threshold,
        "num_prompts": len(results),
    }
    payload = {"summary": summary, "results": results}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return summary


def json_stress_test(
    model: torch.nn.Module,
    tokenizer,
    out_path: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Dict[str, Any]:
    prompt = (
        "Return ONLY valid JSON matching this schema: "
        "{\"id\": int, \"label\": string, \"score\": float, \"items\": [string, ...]} "
        "No extra text."
    )
    inputs = [prompt] * 5
    failures = {"invalid_json": 0, "trailing_text": 0, "missing_keys": 0, "wrong_types": 0}
    success = 0
    failure_examples = []

    for text in inputs:
        output, _, _ = _generate(
            model,
            tokenizer,
            text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            device=device,
        )
        parsed = None
        error_type = None
        try:
            parsed = json.loads(output)
        except Exception:
            try:
                decoder = json.JSONDecoder()
                obj, end = decoder.raw_decode(output)
                if output[end:].strip():
                    error_type = "trailing_text"
                else:
                    parsed = obj
            except Exception:
                error_type = "invalid_json"

        if parsed is None:
            failures[error_type] += 1
            failure_examples.append({"error": error_type, "output": output[:200]})
            continue

        required = {"id": int, "label": str, "score": (int, float), "items": list}
        missing = [k for k in required if k not in parsed]
        if missing:
            failures["missing_keys"] += 1
            failure_examples.append({"error": "missing_keys", "output": output[:200]})
            continue

        wrong = False
        for key, expected_type in required.items():
            if not isinstance(parsed[key], expected_type):
                wrong = True
                break
        if wrong:
            failures["wrong_types"] += 1
            failure_examples.append({"error": "wrong_types", "output": output[:200]})
            continue

        success += 1

    summary = {
        "success_rate": success / len(inputs),
        "failures": failures,
        "num_samples": len(inputs),
    }
    payload = {"summary": summary, "failure_examples": failure_examples}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return summary


def temperature_sweep(
    model: torch.nn.Module,
    tokenizer,
    out_path: str,
    device: str,
    max_new_tokens: int,
    top_p: float,
    seed: int,
    temps: Optional[List[float]] = None,
) -> Dict[str, Any]:
    prompt = "Write a concise one-sentence summary of entropy in neural nets."
    if temps is None:
        temps = [0.0, 0.2, 0.7, 1.0]
    results = []

    for temp in temps:
        _set_seed(seed)
        do_sample = temp > 0.0
        output, _, _ = _generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temp,
            top_p=top_p,
            do_sample=do_sample,
            device=device,
        )
        token_ids = tokenizer(output, add_special_tokens=False)["input_ids"]
        rep_ratio = _token_repetition_ratio(token_ids, n=4)
        length = len(token_ids)
        unique_ratio = (len(set(token_ids)) / length) if length else 0.0
        results.append(
            {
                "temperature": temp,
                "output": output,
                "length": length,
                "repetition_ratio": rep_ratio,
                "unique_token_ratio": unique_ratio,
            }
        )

    payload = {"results": results}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return {"results": results}


def _build_long_context_inputs(tokenizer, target_len: int, needle: str) -> torch.Tensor:
    needle_ids = tokenizer.encode(needle, add_special_tokens=False)
    prompt_ids = tokenizer.encode("\nReturn the needle exactly.", add_special_tokens=False)
    filler_ids = tokenizer.encode("lorem ipsum dolor sit amet ", add_special_tokens=False)

    context_len = max(target_len - len(prompt_ids), len(needle_ids))
    filler_len = max(context_len - len(needle_ids), 0)
    reps = (filler_len // max(len(filler_ids), 1)) + 1
    filler = (filler_ids * reps)[:filler_len]
    insert_pos = len(filler) // 2
    context_ids = filler[:insert_pos] + needle_ids + filler[insert_pos:]
    input_ids = context_ids + prompt_ids
    return torch.tensor([input_ids], dtype=torch.long)


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            cost = 0 if ca == cb else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[-1]


def _is_oom_error(exc: Exception) -> bool:
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def long_context_test(
    model: torch.nn.Module,
    tokenizer,
    out_path: str,
    device: str,
    max_new_tokens: int,
    seed: int,
    lengths: Optional[List[int]] = None,
) -> Dict[str, Any]:
    _set_seed(seed)
    max_len = getattr(tokenizer, "model_max_length", 2048)
    max_positions = getattr(model.config, "max_position_embeddings", max_len)
    effective_max = max(1, int(min(max_len, max_positions) - 1))
    if lengths is None:
        lengths = [512, 1024, 1536, 2048]
    else:
        lengths = [int(l) for l in lengths if l is not None]
    lengths = [l for l in lengths if l <= effective_max]
    lengths = sorted(set(lengths), reverse=True)

    rng = random.Random(seed)
    needle = f"NEEDLE_{rng.getrandbits(32):08x}"

    results = []
    warnings = []
    success_any = False
    smallest_failed = False
    orig_use_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = False
    fixed_max_new = min(int(max_new_tokens), 32)
    try:
        for length in lengths:
            attempt_warnings: List[str] = []
            if device == "cuda":
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            input_ids = _build_long_context_inputs(tokenizer, length, needle)
            input_len = int(input_ids.shape[1])
            input_ids = input_ids.to(device)
            max_new = min(fixed_max_new, max(1, effective_max - input_len))
            attention_mask = torch.ones_like(input_ids)
            peak_mem = None
            try:
                with torch.no_grad():
                    output = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new,
                        do_sample=False,
                        remove_invalid_values=True,
                        renormalize_logits=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                if device == "cuda":
                    peak_mem = int(torch.cuda.max_memory_allocated())
                gen_ids = output[0][input_len:]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                found = needle in text
                candidate = text.strip()
                dist = _edit_distance(candidate[: len(needle)], needle)
                results.append(
                    {
                        "context_length_tokens": input_len,
                        "success": True,
                        "needle": needle,
                        "exact_match": candidate == needle,
                        "needle_found": found,
                        "edit_distance": dist,
                        "extracted_output_excerpt": text[-200:],
                        "warnings": attempt_warnings,
                        "peak_memory_allocated_bytes": peak_mem,
                    }
                )
                success_any = True
            except Exception as exc:
                if _is_oom_error(exc):
                    attempt_warnings.append("oom")
                    warnings.append(f"long_context_oom_length_{length}")
                    if device == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    results.append(
                        {
                            "context_length_tokens": input_len,
                            "success": False,
                            "needle": needle,
                            "exact_match": None,
                            "needle_found": None,
                            "edit_distance": None,
                            "extracted_output_excerpt": None,
                            "warnings": attempt_warnings,
                            "error": "oom",
                            "peak_memory_allocated_bytes": peak_mem,
                        }
                    )
                    if length == min(lengths):
                        smallest_failed = True
                else:
                    attempt_warnings.append(str(exc))
                    warnings.append(f"long_context_error_length_{length}: {exc}")
                    results.append(
                        {
                            "context_length_tokens": input_len,
                            "success": False,
                            "needle": needle,
                            "exact_match": None,
                            "needle_found": None,
                            "edit_distance": None,
                            "extracted_output_excerpt": None,
                            "warnings": attempt_warnings,
                            "error": str(exc),
                            "peak_memory_allocated_bytes": peak_mem,
                        }
                    )
    finally:
        model.config.use_cache = orig_use_cache

    if smallest_failed or not success_any:
        warnings.append("long_context_skipped_due_to_memory")
        results = [
            {
                "context_length_tokens": int(length),
                "success": None,
                "needle": needle,
                "exact_match": None,
                "needle_found": None,
                "edit_distance": None,
                "extracted_output_excerpt": None,
                "warnings": ["long_context_skipped_due_to_memory"],
                "error": "oom",
                "peak_memory_allocated_bytes": None,
            }
            for length in sorted(set(lengths))
        ]

    payload = {"results": results, "warnings": warnings}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return {"results": results, "warnings": warnings}


def latency_memory_test(
    model: torch.nn.Module,
    tokenizer,
    device: str,
    prompt: str,
    max_new_tokens: int,
    warmup_runs: int,
    measured_runs: int,
) -> Dict[str, Any]:
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    prompt_length = int(inputs["input_ids"].shape[1])

    def _sync():
        if device == "cuda":
            torch.cuda.synchronize()

    prefill_times = []
    decode_times = []
    tokens_per_sec = []
    resident_mem = None
    resident_reserved = None

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                remove_invalid_values=True,
                renormalize_logits=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        for _ in range(warmup_runs):
            _ = model(**inputs)

        if device == "cuda":
            _sync()
            resident_mem = int(torch.cuda.memory_allocated())
            resident_reserved = int(torch.cuda.memory_reserved())
            torch.cuda.reset_peak_memory_stats()

        for _ in range(measured_runs):
            _sync()
            start = time.perf_counter()
            _ = model(**inputs)
            _sync()
            prefill = time.perf_counter() - start
            _sync()
            start = time.perf_counter()
            _ = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                remove_invalid_values=True,
                renormalize_logits=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            _sync()
            total = time.perf_counter() - start
            decode = max(total - prefill, 0.0)
            prefill_times.append(prefill)
            decode_times.append(decode)
            tokens_per_sec.append(max_new_tokens / max(decode, 1e-8))

    def _stats(vals: List[float]) -> Dict[str, float]:
        arr = np.array(vals, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
        }

    stats_prefill = _stats(prefill_times) if prefill_times else None
    stats_decode = _stats(decode_times) if decode_times else None
    tps = np.array(tokens_per_sec, dtype=np.float64)
    tps_stats = {
        "mean": float(np.mean(tps)),
        "std": float(np.std(tps)),
        "p50": float(np.percentile(tps, 50)),
        "p90": float(np.percentile(tps, 90)),
    }

    peak_mem = None
    extra_peak_over_resident = None
    if device == "cuda":
        _sync()
        peak_mem = int(torch.cuda.max_memory_allocated())
        extra_peak_over_resident = max(int(peak_mem) - int(resident_mem), 0)

    payload = {
        "prompt_length": prompt_length,
        "max_new_tokens": max_new_tokens,
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "prefill_sec": stats_prefill,
        "decode_sec": stats_decode,
        "tokens_per_sec": tps_stats,
        "peak_allocated_bytes": peak_mem,
        # Captured after warmups and immediately before the measured section begins.
        "resident_allocated_bytes": resident_mem,
        "resident_reserved_bytes": resident_reserved,
        "extra_peak_over_resident_bytes": extra_peak_over_resident,
    }
    return payload


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _safe_write_error(path: str, summary: Dict[str, Any], error: Exception) -> None:
    payload = {"summary": summary, "error": str(error)}
    _write_json(path, payload)


def run_full_suite(
    model: torch.nn.Module,
    tokenizer,
    *,
    out_dir: str,
    config: Dict[str, Any],
    device: str,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    bench_dir = config.get("bench_dir")
    if bench_dir:
        os.makedirs(bench_dir, exist_ok=True)

    seed = int(config.get("seed", 1234))
    _set_seed(seed)
    model.eval()

    metric_plan = config.get("metric_plan")
    if not isinstance(metric_plan, dict):
        metric_plan = resolve_metric_plan(config)

    eval_prompts = config.get("eval_prompts", [])
    max_new_tokens = int(config.get("max_new_tokens", 128))
    temperature = float(config.get("temperature", 0.7))
    top_p = float(config.get("top_p", 0.95))
    ppl_enabled = bool(config.get("ppl_enabled", True)) and bool(metric_plan.get("run_seq_ppl", True))
    ppl_dataset = config.get("ppl_dataset", "wikitext2")
    ppl_mode = str(config.get("ppl_mode", "proxy")).strip().lower()
    ppl_split = config.get("ppl_split", "test" if ppl_mode == "canonical" else "validation")
    ppl_seq_len = int(config.get("ppl_seq_len", 2048 if ppl_mode == "canonical" else 256))
    ppl_max_examples = config.get("ppl_max_examples", None if ppl_mode == "canonical" else 128)
    ppl_stride = config.get("ppl_stride", None)
    ppl_full_corpus = bool(config.get("ppl_full_corpus", ppl_mode == "canonical"))
    latency_prompt = config.get("latency_prompt", "Explain quantization in one sentence.")
    warmup_runs = int(config.get("warmup_runs", 5))
    measured_runs = int(config.get("measured_runs", 20))
    temps = config.get("temperature_sweep_temps")
    long_lengths = config.get("long_context_lengths")
    skip_long_context = bool(config.get("skip_long_context", False)) or not bool(metric_plan.get("run_long_context", True))
    tail_risk_enabled = bool(config.get("tail_risk_enabled", True)) and bool(metric_plan.get("run_tail_risk", True))
    json_stress_enabled = bool(config.get("json_stress_enabled", True)) and bool(metric_plan.get("run_json_stress", True))
    temperature_sweep_enabled = bool(config.get("temperature_sweep_enabled", True)) and bool(
        metric_plan.get("run_temperature_sweep", True)
    )
    latency_memory_enabled = bool(metric_plan.get("run_latency_memory", True)) and bool(
        config.get("latency_memory_enabled", True)
    )
    size_enabled = bool(metric_plan.get("run_size", True)) and bool(config.get("size_enabled", True))
    lm_eval_enabled = bool(metric_plan.get("run_lm_eval", False))

    warnings: List[str] = []

    tail_path = f"{out_dir}/tail_risk.json"
    json_path = f"{out_dir}/json_stress.json"
    temp_path = f"{out_dir}/temperature_sweep.json"
    long_path = f"{out_dir}/long_context.json"
    ppl_path = f"{out_dir}/perplexity.json"
    mmlu_path = f"{out_dir}/mmlu.json"
    zero_shot_path = f"{out_dir}/zero_shot.json"
    latency_path = f"{out_dir}/latency.json"
    memory_path = f"{out_dir}/memory.json"

    tail_summary = {"exact_span_match_rate": None, "repetition_rate": None, "truncation_rate": None}
    if tail_risk_enabled:
        try:
            tail_summary = tail_risk_test(
                model,
                tokenizer,
                eval_prompts,
                tail_path,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                device=device,
            )
        except Exception as exc:
            warnings.append(f"tail_risk_error: {exc}")
            _safe_write_error(tail_path, tail_summary, exc)
    else:
        tail_summary["enabled"] = False
        tail_summary["status"] = "skipped"
        tail_summary["reason"] = "disabled_by_config"
        tail_summary["skip_reason"] = "disabled_by_config"
        _write_json(tail_path, tail_summary)

    json_summary = {"success_rate": None, "failures": {}, "num_samples": 0}
    if json_stress_enabled:
        try:
            json_summary = json_stress_test(
                model,
                tokenizer,
                json_path,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        except Exception as exc:
            warnings.append(f"json_stress_error: {exc}")
            _safe_write_error(json_path, json_summary, exc)
    else:
        json_summary["enabled"] = False
        json_summary["status"] = "skipped"
        json_summary["reason"] = "disabled_by_config"
        json_summary["skip_reason"] = "disabled_by_config"
        _write_json(json_path, json_summary)

    temp_results = {"results": []}
    if temperature_sweep_enabled:
        try:
            temp_results = temperature_sweep(
                model,
                tokenizer,
                temp_path,
                device=device,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                seed=seed,
                temps=temps,
            )
        except Exception as exc:
            warnings.append(f"temperature_sweep_error: {exc}")
            _safe_write_error(temp_path, {"results": []}, exc)
    else:
        temp_results = {
            "results": [],
            "enabled": False,
            "status": "skipped",
            "reason": "disabled_by_config",
            "skip_reason": "disabled_by_config",
        }
        _write_json(temp_path, temp_results)

    long_results = {"results": []}
    if skip_long_context:
        long_results = {
            "results": [],
            "warnings": [],
            "skipped": True,
            "status": "skipped",
            "reason": "disabled_by_config",
            "skip_reason": "disabled_by_config",
        }
        _write_json(long_path, long_results)
    else:
        try:
            long_results = long_context_test(
                model,
                tokenizer,
                long_path,
                device=device,
                max_new_tokens=max_new_tokens,
                seed=seed,
                lengths=long_lengths,
            )
            if isinstance(long_results, dict) and long_results.get("warnings"):
                warnings.extend(long_results.get("warnings", []))
        except Exception as exc:
            warnings.append(f"long_context_error: {exc}")
            _safe_write_error(long_path, {"results": []}, exc)

    perplexity_summary = {
        "loss": None,
        "avg_loss": None,
        "ppl": None,
        "num_tokens": 0,
        "num_supervised_tokens": 0,
        "num_sequences": 0,
        "num_full_chunks": 0,
        "num_examples": 0,
        "num_dataset_rows": 0,
        "num_corpus_tokens": 0,
        "seq_len": ppl_seq_len,
        "dataset_name": ppl_dataset,
        "split": ppl_split,
        "max_examples": ppl_max_examples,
        "stride": ppl_stride,
        "mode": ppl_mode,
        "full_corpus": ppl_full_corpus,
        "stream_mode": "continuous_stream" if ppl_mode == "canonical" else "row_wise_eos_joined",
        "chunking": "non_overlapping",
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__),
        "add_special_tokens": False,
        "loss_impl": "hf_internal_shift",
        "tail_dropped": False,
        "tail_tokens_dropped": 0,
        "sample_indices": [],
        "method": "canonical_full_corpus_continuous_stream" if ppl_mode == "canonical" else "proxy_first_n_nonempty_rowwise",
        "status": "pending",
        "error": None,
        "notes": "canonical_full_corpus" if ppl_mode == "canonical" else "fast_proxy_rowwise",
    }
    try:
        if ppl_enabled:
            ppl_info = compute_ppl(
                model,
                tokenizer,
                dataset_name=ppl_dataset,
                split=ppl_split,
                seq_len=ppl_seq_len,
                max_examples=ppl_max_examples,
                stride=ppl_stride,
                device=device,
                dtype=dtype,
                seed=seed,
                mode=ppl_mode,
                full_corpus=ppl_full_corpus,
            )
            perplexity_summary = {
                "loss": ppl_info.get("avg_loss"),
                "avg_loss": ppl_info.get("avg_loss"),
                "ppl": ppl_info.get("ppl"),
                "num_tokens": ppl_info.get("num_tokens", 0),
                "num_supervised_tokens": ppl_info.get("num_supervised_tokens", 0),
                "num_sequences": ppl_info.get("num_sequences", 0),
                "num_full_chunks": ppl_info.get("num_full_chunks", ppl_info.get("num_sequences", 0)),
                "num_examples": ppl_info.get("num_examples", 0),
                "num_dataset_rows": ppl_info.get("num_dataset_rows", 0),
                "num_corpus_tokens": ppl_info.get("num_corpus_tokens", 0),
                "seq_len": ppl_info.get("seq_len", ppl_seq_len),
                "dataset_name": ppl_info.get("dataset_name", ppl_dataset),
                "split": ppl_info.get("split", ppl_split),
                "max_examples": ppl_info.get("max_examples", ppl_max_examples),
                "stride": ppl_info.get("stride", ppl_stride),
                "mode": ppl_info.get("mode", ppl_mode),
                "full_corpus": ppl_info.get("full_corpus", ppl_full_corpus),
                "stream_mode": ppl_info.get("stream_mode"),
                "chunking": ppl_info.get("chunking"),
                "tokenizer_name_or_path": ppl_info.get("tokenizer_name_or_path"),
                "add_special_tokens": ppl_info.get("add_special_tokens"),
                "loss_impl": ppl_info.get("loss_impl"),
                "tail_dropped": ppl_info.get("tail_dropped"),
                "tail_tokens_dropped": ppl_info.get("tail_tokens_dropped", 0),
                "sample_indices": ppl_info.get("sample_indices", []),
                "method": ppl_info.get("method"),
                "status": "error" if ppl_info.get("error") else "ok",
                "error": ppl_info.get("error"),
                "notes": ppl_info.get("notes"),
            }
            if ppl_info.get("error"):
                warnings.append(f"perplexity_error: {ppl_info.get('error')}")
        else:
            perplexity_summary["error"] = "disabled"
            perplexity_summary["status"] = "skipped"
            perplexity_summary["skip_reason"] = "disabled_by_config"
    except Exception as exc:
        warnings.append(f"perplexity_error: {exc}")
        perplexity_summary["error"] = str(exc)
        perplexity_summary["status"] = "error"
    _write_json(ppl_path, perplexity_summary)

    mmlu_cfg = config.get("mmlu") or {}
    mmlu_enabled = bool(mmlu_cfg.get("enabled", True)) and bool(metric_plan.get("run_mmlu", True))
    mmlu_summary = {
        "enabled": mmlu_enabled,
        "accuracy": None,
        "num_examples": 0,
        "num_subjects": 0,
        "dataset_name": None,
        "split": mmlu_cfg.get("split", "test"),
        "error": None,
        "status": "pending" if mmlu_enabled else "skipped",
    }
    if mmlu_enabled:
        try:
            mmlu_payload = run_mmlu_eval(
                model,
                tokenizer,
                mmlu_path,
                device=device,
                split=str(mmlu_cfg.get("split", "test")),
                max_subjects=mmlu_cfg.get("max_subjects", 8),
                max_examples_per_subject=mmlu_cfg.get("max_examples_per_subject", 4),
                dataset_candidates=mmlu_cfg.get("dataset_candidates", ["cais/mmlu", "hendrycks_test"]),
            )
            mmlu_summary = {
                "enabled": True,
                "accuracy": mmlu_payload.get("accuracy"),
                "num_examples": mmlu_payload.get("num_examples", 0),
                "num_subjects": mmlu_payload.get("num_subjects", 0),
                "dataset_name": mmlu_payload.get("dataset_name"),
                "split": mmlu_payload.get("split"),
                "error": mmlu_payload.get("error"),
                "status": "error" if mmlu_payload.get("error") else "ok",
            }
            if mmlu_payload.get("error"):
                warnings.append(f"mmlu_error: {mmlu_payload.get('error')}")
        except Exception as exc:
            warnings.append(f"mmlu_error: {exc}")
            mmlu_summary["error"] = str(exc)
            mmlu_summary["status"] = "error"
            _write_json(mmlu_path, {"error": str(exc), "accuracy": None, "num_examples": 0, "num_subjects": 0})
    else:
        mmlu_summary["enabled"] = False
        mmlu_summary["skip_reason"] = "disabled_by_config"
        _write_json(mmlu_path, {"enabled": False, "reason": "disabled_by_config"})

    zero_shot_cfg = config.get("zero_shot") or {}
    zero_shot_enabled = bool(zero_shot_cfg.get("enabled", True)) and bool(metric_plan.get("run_zero_shot", True))
    default_zero_shot_tasks = [{"name": "arc_easy", "split": "validation", "max_examples": 64}]
    zero_shot_tasks = zero_shot_cfg.get("tasks") or default_zero_shot_tasks
    zero_shot_summary = {
        "enabled": zero_shot_enabled,
        "mean_accuracy": None,
        "micro_accuracy": None,
        "num_tasks": 0,
        "num_examples": 0,
        "tasks": [],
        "error": None,
        "status": "pending" if zero_shot_enabled else "skipped",
    }
    if zero_shot_enabled:
        try:
            zero_payload = run_zero_shot_suite(
                model,
                tokenizer,
                zero_shot_path,
                device=device,
                tasks=zero_shot_tasks,
                default_max_examples=zero_shot_cfg.get("max_examples_per_task", 64),
                default_split=str(zero_shot_cfg.get("default_split", "validation")),
            )
            zero_shot_summary = {
                "enabled": True,
                "mean_accuracy": zero_payload.get("mean_accuracy"),
                "micro_accuracy": zero_payload.get("micro_accuracy"),
                "num_tasks": zero_payload.get("num_tasks", 0),
                "num_examples": zero_payload.get("num_examples", 0),
                "tasks": [
                    {
                        "task": task.get("task"),
                        "accuracy": task.get("accuracy"),
                        "num_examples": task.get("num_examples", 0),
                        "dataset_name": task.get("dataset_name"),
                        "error": task.get("error"),
                    }
                    for task in zero_payload.get("tasks", [])
                ],
                "error": zero_payload.get("error"),
                "status": "error" if zero_payload.get("error") else "ok",
            }
            if zero_payload.get("error"):
                warnings.append(f"zero_shot_error: {zero_payload.get('error')}")
            for task in zero_payload.get("tasks", []):
                if task.get("error"):
                    warnings.append(f"zero_shot_{task.get('task')}_error: {task.get('error')}")
        except Exception as exc:
            warnings.append(f"zero_shot_error: {exc}")
            zero_shot_summary["error"] = str(exc)
            zero_shot_summary["status"] = "error"
            _write_json(zero_shot_path, {"error": str(exc), "tasks": []})
    else:
        zero_shot_summary["enabled"] = False
        zero_shot_summary["skip_reason"] = "disabled_by_config"
        _write_json(zero_shot_path, {"enabled": False, "reason": "disabled_by_config", "tasks": zero_shot_tasks})

    size_summary = {
        "quant_model_dir_bytes": None,
        "fp16_weight_est_bytes": None,
        "notes": "bitsandbytes runtime quant; disk != VRAM",
        "status": "pending" if size_enabled else "skipped",
    }
    if size_enabled:
        try:
            size_summary["fp16_weight_est_bytes"] = estimate_fp16_weight_bytes(model)
            quant_dir = config.get("quant_model_dir")
            if quant_dir:
                size_summary["quant_model_dir_bytes"] = dir_size_bytes(str(quant_dir))
            size_summary["status"] = "ok"
        except Exception as exc:
            warnings.append(f"size_error: {exc}")
            size_summary["error"] = str(exc)
            size_summary["status"] = "error"
    else:
        size_summary["skip_reason"] = "disabled_by_config"

    latency_summary = {
        "prefill_sec": None,
        "decode_sec": None,
        "tokens_per_sec": None,
        "peak_allocated_bytes": None,
        "resident_allocated_bytes": None,
        "resident_reserved_bytes": None,
        "extra_peak_over_resident_bytes": None,
        "prompt_length": None,
        "max_new_tokens": None,
        "warmup_runs": None,
        "measured_runs": None,
        "status": "pending" if latency_memory_enabled else "skipped",
    }
    if latency_memory_enabled:
        try:
            latency_summary = latency_memory_test(
                model,
                tokenizer,
                device=device,
                prompt=latency_prompt,
                max_new_tokens=max_new_tokens,
                warmup_runs=warmup_runs,
                measured_runs=measured_runs,
            )
            latency_summary["status"] = "ok"
            _write_json(latency_path, latency_summary)
            peak_bytes = latency_summary.get("peak_allocated_bytes")
            resident_bytes = latency_summary.get("resident_allocated_bytes")
            resident_reserved_bytes = latency_summary.get("resident_reserved_bytes")
            extra_bytes = latency_summary.get("extra_peak_over_resident_bytes")
            peak_gb = None
            resident_gb = None
            resident_reserved_gb = None
            extra_gb = None
            if peak_bytes is not None:
                peak_gb = float(peak_bytes) / (1024 ** 3)
            if resident_bytes is not None:
                resident_gb = float(resident_bytes) / (1024 ** 3)
            if resident_reserved_bytes is not None:
                resident_reserved_gb = float(resident_reserved_bytes) / (1024 ** 3)
            if extra_bytes is not None:
                extra_gb = float(extra_bytes) / (1024 ** 3)
            _write_json(
                memory_path,
                {
                    "peak_allocated_bytes": peak_bytes,
                    "peak_allocated_gb": peak_gb,
                    "resident_mem_after_load_bytes": resident_bytes,
                    "resident_mem_after_load_gb": resident_gb,
                    "resident_reserved_after_load_bytes": resident_reserved_bytes,
                    "resident_reserved_after_load_gb": resident_reserved_gb,
                    "extra_peak_over_resident_bytes": extra_bytes,
                    "extra_peak_over_resident_gb": extra_gb,
                    "device": device,
                    "cuda_available": torch.cuda.is_available(),
                    "status": "ok",
                },
            )
        except Exception as exc:
            warnings.append(f"latency_memory_error: {exc}")
            latency_summary["status"] = "error"
            latency_summary["error"] = str(exc)
            _write_json(
                latency_path,
                {
                    "prefill_sec": None,
                    "decode_sec": None,
                    "tokens_per_sec": None,
                    "peak_allocated_bytes": None,
                    "resident_allocated_bytes": None,
                    "resident_reserved_bytes": None,
                    "extra_peak_over_resident_bytes": None,
                    "prompt_length": None,
                    "max_new_tokens": None,
                    "warmup_runs": warmup_runs,
                    "measured_runs": measured_runs,
                    "status": "error",
                    "error": str(exc),
                },
            )
            _write_json(
                memory_path,
                {
                    "peak_allocated_bytes": None,
                    "peak_allocated_gb": None,
                    "resident_mem_after_load_bytes": None,
                    "resident_mem_after_load_gb": None,
                    "resident_reserved_after_load_bytes": None,
                    "resident_reserved_after_load_gb": None,
                    "extra_peak_over_resident_bytes": None,
                    "extra_peak_over_resident_gb": None,
                    "device": device,
                    "cuda_available": torch.cuda.is_available(),
                    "status": "error",
                    "error": str(exc),
                },
            )
    else:
        latency_summary["skip_reason"] = "disabled_by_config"
        _write_json(
            latency_path,
            {
                "prefill_sec": None,
                "decode_sec": None,
                "tokens_per_sec": None,
                "peak_allocated_bytes": None,
                "resident_allocated_bytes": None,
                "resident_reserved_bytes": None,
                "extra_peak_over_resident_bytes": None,
                "prompt_length": None,
                "max_new_tokens": None,
                "warmup_runs": warmup_runs,
                "measured_runs": measured_runs,
                "status": "skipped",
                "skip_reason": "disabled_by_config",
            },
        )
        _write_json(
            memory_path,
            {
                "peak_allocated_bytes": None,
                "peak_allocated_gb": None,
                "resident_mem_after_load_bytes": None,
                "resident_mem_after_load_gb": None,
                "resident_reserved_after_load_bytes": None,
                "resident_reserved_after_load_gb": None,
                "extra_peak_over_resident_bytes": None,
                "extra_peak_over_resident_gb": None,
                "device": device,
                "cuda_available": torch.cuda.is_available(),
                "status": "skipped",
                "skip_reason": "disabled_by_config",
            },
        )

    if bench_dir:
        try:
            shutil.copy(latency_path, os.path.join(bench_dir, "latency.json"))
            shutil.copy(memory_path, os.path.join(bench_dir, "memory.json"))
        except Exception as exc:
            warnings.append(f"bench_copy_error: {exc}")

    lm_eval_summary = {
        "status": "skipped",
        "reason": "disabled_by_config",
        "requested": False,
        "results": {},
        "flat": {"lm_eval__status": "skipped", "lm_eval__tasks": ""},
    }
    if lm_eval_enabled:
        lm_eval_cfg = dict(config.get("lm_eval") or {})
        lm_eval_cfg["enabled"] = True
        if config.get("lm_eval_skip_reason"):
            lm_eval_cfg["skip_reason"] = config.get("lm_eval_skip_reason")
        lm_eval_summary = run_lm_eval_suite(
            model_name_or_path=config.get("lm_eval_model_name_or_path"),
            tokenizer_name_or_path=config.get("lm_eval_tokenizer_name_or_path"),
            out_dir=out_dir,
            config=lm_eval_cfg,
            device=device,
            dtype=config.get("lm_eval_dtype") or str(dtype).replace("torch.", ""),
            model_args_extra=config.get("lm_eval_model_args_extra"),
        )
        if lm_eval_summary.get("status") == "error":
            warnings.append(f"lm_eval_error: {lm_eval_summary.get('reason')}")
        if (
            lm_eval_summary.get("status") == "skipped"
            and lm_eval_summary.get("requested")
            and lm_eval_summary.get("fail_policy") != "skip"
        ):
            warnings.append(f"lm_eval_skipped: {lm_eval_summary.get('reason')}")

    summary = {
        "metric_plan": metric_plan,
        "tail_risk": tail_summary,
        "tail_risk_status": tail_summary.get("status", "ok" if tail_risk_enabled else "skipped"),
        "tail_risk_skip_reason": tail_summary.get("skip_reason"),
        "json_stress": json_summary,
        "json_stress_status": json_summary.get("status", "ok" if json_stress_enabled else "skipped"),
        "json_stress_skip_reason": json_summary.get("skip_reason"),
        "temperature_sweep": {
            "num_temps": len(temp_results.get("results", [])),
            "status": temp_results.get("status", "ok" if temperature_sweep_enabled else "skipped"),
        },
        "temperature_sweep_status": temp_results.get("status", "ok" if temperature_sweep_enabled else "skipped"),
        "temperature_sweep_skip_reason": temp_results.get("skip_reason"),
        "long_context": {
            "num_lengths": len(long_results.get("results", [])),
            "skipped": skip_long_context,
            "status": long_results.get("status", "skipped" if skip_long_context else "ok"),
        },
        "long_context_status": long_results.get("status", "skipped" if skip_long_context else "ok"),
        "long_context_skip_reason": long_results.get("skip_reason"),
        "perplexity": perplexity_summary,
        "perplexity_status": perplexity_summary.get("status"),
        "perplexity_skip_reason": perplexity_summary.get("skip_reason"),
        "mmlu": mmlu_summary,
        "mmlu_status": mmlu_summary.get("status"),
        "mmlu_skip_reason": mmlu_summary.get("skip_reason"),
        "zero_shot": zero_shot_summary,
        "zero_shot_status": zero_shot_summary.get("status"),
        "zero_shot_skip_reason": zero_shot_summary.get("skip_reason"),
        "size": size_summary,
        "size_status": size_summary.get("status"),
        "size_skip_reason": size_summary.get("skip_reason"),
        "latency": {
            "prefill_sec": latency_summary.get("prefill_sec"),
            "decode_sec": latency_summary.get("decode_sec"),
            "tokens_per_sec": latency_summary.get("tokens_per_sec"),
            "prompt_length": latency_summary.get("prompt_length"),
            "max_new_tokens": latency_summary.get("max_new_tokens"),
            "status": latency_summary.get("status"),
        },
        "latency_status": latency_summary.get("status"),
        "latency_skip_reason": latency_summary.get("skip_reason"),
        "memory": {
            "peak_allocated_bytes": latency_summary.get("peak_allocated_bytes"),
            "peak_memory_bytes": latency_summary.get("peak_allocated_bytes"),
            "resident_allocated_bytes": latency_summary.get("resident_allocated_bytes"),
            "resident_mem_after_load_bytes": latency_summary.get("resident_allocated_bytes"),
            "resident_reserved_bytes": latency_summary.get("resident_reserved_bytes"),
            "resident_reserved_after_load_bytes": latency_summary.get("resident_reserved_bytes"),
            "extra_peak_over_resident_bytes": latency_summary.get("extra_peak_over_resident_bytes"),
            "status": latency_summary.get("status"),
        },
        "memory_status": latency_summary.get("status"),
        "memory_skip_reason": latency_summary.get("skip_reason"),
        "lm_eval": lm_eval_summary,
        "lm_eval_status": lm_eval_summary.get("status"),
        "lm_eval_skip_reason": lm_eval_summary.get("reason") if lm_eval_summary.get("status") == "skipped" else None,
        "warnings": warnings,
    }

    _write_json(f"{out_dir}/eval_summary.json", summary)

    return summary


def run_evaluation_suite(
    model: torch.nn.Module,
    tokenizer,
    device: str,
    out_dir: str,
    eval_prompts: List[Dict[str, Any]],
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    latency_prompt: str,
    warmup_runs: int,
    measured_runs: int,
) -> Dict[str, Any]:
    config = {
        "eval_prompts": eval_prompts,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "latency_prompt": latency_prompt,
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "seed": seed,
    }
    return run_full_suite(
        model,
        tokenizer,
        out_dir=out_dir,
        config=config,
        device=device,
        dtype=torch.float32,
    )
