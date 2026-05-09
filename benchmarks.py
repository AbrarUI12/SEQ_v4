#!/usr/bin/env python3
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch

LOGGER = logging.getLogger(__name__)

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except Exception:
    load_dataset = None
    DATASETS_AVAILABLE = False


def get_directory_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for fname in files:
            fpath = os.path.join(root, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                total += os.path.getsize(fpath)
            except OSError:
                continue
    return total


def summarize_model_disk_footprint(model_dir: str) -> Dict[str, Any]:
    if not model_dir or not os.path.isdir(model_dir):
        return {
            "model_dir": model_dir,
            "exists": False,
            "total_bytes": None,
            "total_MB": None,
            "total_GB": None,
            "count_files": 0,
            "by_extension": {},
            "method": "missing_dir",
        }

    total_bytes = 0
    count_files = 0
    by_extension: Dict[str, Dict[str, Any]] = {}
    for root, _, files in os.walk(model_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            if not os.path.isfile(fpath):
                continue
            ext = os.path.splitext(fname)[1].lower() or ".no_ext"
            try:
                size = os.path.getsize(fpath)
            except OSError:
                size = 0
            total_bytes += size
            count_files += 1
            if ext not in by_extension:
                by_extension[ext] = {"bytes": 0, "count": 0}
            by_extension[ext]["bytes"] += size
            by_extension[ext]["count"] += 1

    total_mb = total_bytes / (1024 ** 2)
    total_gb = total_bytes / (1024 ** 3)
    return {
        "model_dir": model_dir,
        "exists": True,
        "total_bytes": int(total_bytes),
        "total_MB": float(total_mb),
        "total_GB": float(total_gb),
        "count_files": int(count_files),
        "by_extension": by_extension,
        "method": "actual_dir_size",
    }


def estimate_fp16_size(num_params: int, bytes_per_param: int = 2) -> Dict[str, Any]:
    total_bytes = int(num_params) * int(bytes_per_param)
    return {
        "method": "estimate_from_num_params",
        "num_params": int(num_params),
        "estimated_bytes": int(total_bytes),
        "estimated_MB": float(total_bytes / (1024 ** 2)),
        "estimated_GB": float(total_bytes / (1024 ** 3)),
    }


def count_model_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _load_text_dataset(dataset_name: str, split: str):
    if not DATASETS_AVAILABLE:
        raise RuntimeError("datasets is not available")
    name_lower = dataset_name.lower()
    if name_lower in {"wikitext2", "wikitext-2", "wikitext"}:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    else:
        ds = load_dataset(dataset_name, split=split)
    if "text" in ds.column_names:
        field = "text"
    else:
        field = ds.column_names[0]
    return ds, field


def _normalize_ppl_mode(mode: Optional[str]) -> str:
    mode_norm = str(mode or "proxy").strip().lower()
    if mode_norm not in {"proxy", "canonical"}:
        raise ValueError(f"Unsupported ppl_mode: {mode}")
    return mode_norm


def _tokenizer_name_or_path(tokenizer) -> str:
    name = getattr(tokenizer, "name_or_path", None)
    if isinstance(name, str) and name.strip():
        return name
    return tokenizer.__class__.__name__


def _build_ppl_result(
    *,
    mode: str,
    dataset_name: str,
    split: str,
    seq_len: int,
    max_examples: Optional[int],
    stride: Optional[int],
    tokenizer,
    full_corpus: bool,
    stream_mode: str,
    chunking: str,
    method: str,
    notes: str,
    add_special_tokens: bool,
    loss_impl: str,
    error: Optional[str] = None,
    avg_loss: Optional[float] = None,
    ppl: Optional[float] = None,
    num_supervised_tokens: int = 0,
    num_sequences: int = 0,
    num_examples: int = 0,
    num_dataset_rows: int = 0,
    sample_indices: Optional[List[int]] = None,
    tail_tokens_dropped: int = 0,
    num_corpus_tokens: int = 0,
) -> Dict[str, Any]:
    sample_indices = sample_indices or []
    return {
        "error": error,
        "mode": mode,
        "avg_loss": avg_loss,
        "ppl": ppl,
        "num_tokens": int(num_supervised_tokens),
        "num_supervised_tokens": int(num_supervised_tokens),
        "num_sequences": int(num_sequences),
        "num_full_chunks": int(num_sequences),
        "num_examples": int(num_examples),
        "num_dataset_rows": int(num_dataset_rows),
        "num_corpus_tokens": int(num_corpus_tokens),
        "seq_len": int(seq_len),
        "dataset_name": dataset_name,
        "split": split,
        "max_examples": max_examples,
        "stride": stride,
        "full_corpus": bool(full_corpus),
        "stream_mode": stream_mode,
        "chunking": chunking,
        "sample_indices": sample_indices,
        "method": method,
        "notes": notes,
        "tokenizer_name_or_path": _tokenizer_name_or_path(tokenizer),
        "add_special_tokens": bool(add_special_tokens),
        "loss_impl": loss_impl,
        "tail_dropped": bool(tail_tokens_dropped > 0),
        "tail_tokens_dropped": int(tail_tokens_dropped),
    }


def _log_ppl_result(payload: Dict[str, Any]) -> None:
    level = LOGGER.info if not payload.get("error") else LOGGER.warning
    level(
        "PPL mode=%s dataset=%s split=%s tokenizer=%s seq_len=%s stream=%s chunking=%s "
        "full_chunks=%s supervised_tokens=%s tail_dropped=%s tail_tokens_dropped=%s loss_impl=%s "
        "avg_loss=%s ppl=%s error=%s",
        payload.get("mode"),
        payload.get("dataset_name"),
        payload.get("split"),
        payload.get("tokenizer_name_or_path"),
        payload.get("seq_len"),
        payload.get("stream_mode"),
        payload.get("chunking"),
        payload.get("num_full_chunks"),
        payload.get("num_supervised_tokens"),
        payload.get("tail_dropped"),
        payload.get("tail_tokens_dropped"),
        payload.get("loss_impl"),
        payload.get("avg_loss"),
        payload.get("ppl"),
        payload.get("error"),
    )


def compute_proxy_ppl(
    model: torch.nn.Module,
    tokenizer,
    dataset_name: str = "wikitext2",
    split: str = "validation",
    seq_len: int = 256,
    max_examples: Optional[int] = None,
    stride: Optional[int] = None,
    device: str = "cuda",
    dtype: Optional[torch.dtype] = None,
    seed: int = 1234,
) -> Dict[str, Any]:
    """Fast proxy perplexity for development runs.

    This preserves the repo's original row-wise sampling behavior: select the
    first non-empty examples, tokenize each row separately, append EOS between
    rows, then score fixed-size windows. It is intentionally distinct from the
    canonical paper-style WikiText-2 evaluation.
    """
    del dtype
    mode = "proxy"
    method = "proxy_first_n_nonempty_rowwise"
    notes = "fast_proxy_rowwise_eos_joined"
    if not DATASETS_AVAILABLE:
        payload = _build_ppl_result(
            mode=mode,
            dataset_name=dataset_name,
            split=split,
            seq_len=seq_len,
            max_examples=max_examples,
            stride=stride,
            tokenizer=tokenizer,
            full_corpus=False,
            stream_mode="row_wise_eos_joined",
            chunking="non_overlapping" if stride in {None, seq_len} else "strided",
            method=method,
            notes=notes,
            add_special_tokens=False,
            loss_impl="hf_internal_shift",
            error="datasets_unavailable",
        )
        _log_ppl_result(payload)
        return payload

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        ds, field = _load_text_dataset(dataset_name, split)
    except Exception as exc:
        payload = _build_ppl_result(
            mode=mode,
            dataset_name=dataset_name,
            split=split,
            seq_len=seq_len,
            max_examples=max_examples,
            stride=stride,
            tokenizer=tokenizer,
            full_corpus=False,
            stream_mode="row_wise_eos_joined",
            chunking="non_overlapping" if stride in {None, seq_len} else "strided",
            method=method,
            notes=notes,
            add_special_tokens=False,
            loss_impl="hf_internal_shift",
            error=f"dataset_load_failed: {exc}",
        )
        _log_ppl_result(payload)
        return payload

    texts: List[str] = []
    sample_indices: List[int] = []
    for idx, row in enumerate(ds):
        text = row.get(field, "")
        if not isinstance(text, str) or not text.strip():
            continue
        texts.append(text)
        sample_indices.append(idx)
        if max_examples is not None and len(texts) >= max_examples:
            break

    eos_id = tokenizer.eos_token_id
    token_ids: List[int] = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if ids:
            token_ids.extend(ids)
            if eos_id is not None:
                token_ids.append(eos_id)

    if stride is None:
        stride = seq_len

    if len(token_ids) < seq_len:
        payload = _build_ppl_result(
            mode=mode,
            dataset_name=dataset_name,
            split=split,
            seq_len=seq_len,
            max_examples=max_examples,
            stride=stride,
            tokenizer=tokenizer,
            full_corpus=False,
            stream_mode="row_wise_eos_joined",
            chunking="non_overlapping" if stride == seq_len else "strided",
            method=method,
            notes=notes,
            add_special_tokens=False,
            loss_impl="hf_internal_shift",
            error="insufficient_tokens",
            num_examples=len(sample_indices),
            num_dataset_rows=len(ds),
            sample_indices=sample_indices,
            num_corpus_tokens=len(token_ids),
        )
        _log_ppl_result(payload)
        return payload

    total_nll = 0.0
    total_supervised_tokens = 0
    num_sequences = 0
    last_end = 0

    model.eval()
    with torch.no_grad():
        for start in range(0, len(token_ids) - seq_len + 1, stride):
            seq = token_ids[start : start + seq_len]
            input_ids = torch.tensor([seq], dtype=torch.long, device=device)
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            supervised_tokens = max(input_ids.shape[1] - 1, 0)
            total_nll += float(loss.item()) * supervised_tokens
            total_supervised_tokens += supervised_tokens
            num_sequences += 1
            last_end = start + seq_len

    tail_tokens_dropped = max(len(token_ids) - last_end, 0)
    avg_loss = total_nll / max(total_supervised_tokens, 1)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")

    payload = _build_ppl_result(
        mode=mode,
        dataset_name=dataset_name,
        split=split,
        seq_len=seq_len,
        max_examples=max_examples,
        stride=stride,
        tokenizer=tokenizer,
        full_corpus=False,
        stream_mode="row_wise_eos_joined",
        chunking="non_overlapping" if stride == seq_len else "strided",
        method=method,
        notes=notes,
        add_special_tokens=False,
        loss_impl="hf_internal_shift",
        avg_loss=float(avg_loss),
        ppl=float(ppl),
        num_supervised_tokens=total_supervised_tokens,
        num_sequences=num_sequences,
        num_examples=len(sample_indices),
        num_dataset_rows=len(ds),
        sample_indices=sample_indices,
        tail_tokens_dropped=tail_tokens_dropped,
        num_corpus_tokens=len(token_ids),
    )
    _log_ppl_result(payload)
    return payload


def compute_canonical_ppl(
    model: torch.nn.Module,
    tokenizer,
    dataset_name: str = "wikitext2",
    split: str = "test",
    seq_len: int = 2048,
    device: str = "cuda",
    dtype: Optional[torch.dtype] = None,
    seed: int = 1234,
    add_special_tokens: bool = False,
) -> Dict[str, Any]:
    """Canonical full-corpus perplexity for paper-style WikiText-2 reporting.

    Canonical mode follows the standard protocol used by many quantization
    papers: load the full eval split, join rows into one continuous text stream
    with ``"\\n\\n"``, tokenize once, score contiguous non-overlapping fixed
    windows, drop the final short tail, and compute perplexity from the exact
    number of supervised next-token targets.
    """
    del dtype
    mode = "canonical"
    method = "canonical_full_corpus_continuous_stream"
    notes = "paper_style_continuous_stream_non_overlapping_chunks"
    if not DATASETS_AVAILABLE:
        payload = _build_ppl_result(
            mode=mode,
            dataset_name=dataset_name,
            split=split,
            seq_len=seq_len,
            max_examples=None,
            stride=None,
            tokenizer=tokenizer,
            full_corpus=True,
            stream_mode="continuous_stream",
            chunking="non_overlapping",
            method=method,
            notes=notes,
            add_special_tokens=add_special_tokens,
            loss_impl="hf_internal_shift",
            error="datasets_unavailable",
        )
        _log_ppl_result(payload)
        return payload

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        ds, field = _load_text_dataset(dataset_name, split)
    except Exception as exc:
        payload = _build_ppl_result(
            mode=mode,
            dataset_name=dataset_name,
            split=split,
            seq_len=seq_len,
            max_examples=None,
            stride=None,
            tokenizer=tokenizer,
            full_corpus=True,
            stream_mode="continuous_stream",
            chunking="non_overlapping",
            method=method,
            notes=notes,
            add_special_tokens=add_special_tokens,
            loss_impl="hf_internal_shift",
            error=f"dataset_load_failed: {exc}",
        )
        _log_ppl_result(payload)
        return payload

    text_rows = ds[field]
    text = "\n\n".join((row if isinstance(row, str) else "") for row in text_rows)
    token_ids = tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"]
    full_chunks = len(token_ids) // seq_len
    tail_tokens_dropped = len(token_ids) - (full_chunks * seq_len)

    if full_chunks == 0:
        payload = _build_ppl_result(
            mode=mode,
            dataset_name=dataset_name,
            split=split,
            seq_len=seq_len,
            max_examples=None,
            stride=None,
            tokenizer=tokenizer,
            full_corpus=True,
            stream_mode="continuous_stream",
            chunking="non_overlapping",
            method=method,
            notes=notes,
            add_special_tokens=add_special_tokens,
            loss_impl="hf_internal_shift",
            error="insufficient_tokens",
            num_examples=len(text_rows),
            num_dataset_rows=len(text_rows),
            tail_tokens_dropped=tail_tokens_dropped,
            num_corpus_tokens=len(token_ids),
        )
        _log_ppl_result(payload)
        return payload

    usable_tokens = full_chunks * seq_len
    total_nll = 0.0
    total_supervised_tokens = 0

    model.eval()
    with torch.no_grad():
        for start in range(0, usable_tokens, seq_len):
            seq = token_ids[start : start + seq_len]
            input_ids = torch.tensor([seq], dtype=torch.long, device=device)
            outputs = model(input_ids=input_ids, labels=input_ids)
            supervised_tokens = max(input_ids.shape[1] - 1, 0)
            total_nll += float(outputs.loss.item()) * supervised_tokens
            total_supervised_tokens += supervised_tokens

    avg_loss = total_nll / max(total_supervised_tokens, 1)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    payload = _build_ppl_result(
        mode=mode,
        dataset_name=dataset_name,
        split=split,
        seq_len=seq_len,
        max_examples=None,
        stride=None,
        tokenizer=tokenizer,
        full_corpus=True,
        stream_mode="continuous_stream",
        chunking="non_overlapping",
        method=method,
        notes=notes,
        add_special_tokens=add_special_tokens,
        loss_impl="hf_internal_shift",
        avg_loss=float(avg_loss),
        ppl=float(ppl),
        num_supervised_tokens=total_supervised_tokens,
        num_sequences=full_chunks,
        num_examples=len(text_rows),
        num_dataset_rows=len(text_rows),
        tail_tokens_dropped=tail_tokens_dropped,
        num_corpus_tokens=len(token_ids),
    )
    _log_ppl_result(payload)
    return payload


def compute_ppl(
    model: torch.nn.Module,
    tokenizer,
    dataset_name: str = "wikitext2",
    split: str = "validation",
    seq_len: int = 256,
    max_examples: Optional[int] = None,
    stride: Optional[int] = None,
    device: str = "cuda",
    dtype: Optional[torch.dtype] = None,
    seed: int = 1234,
    mode: str = "proxy",
    full_corpus: Optional[bool] = None,
) -> Dict[str, Any]:
    """Dispatch between proxy and canonical perplexity modes.

    ``proxy`` keeps the repo's fast development behavior.
    ``canonical`` matches the standard full-corpus WikiText-2 paper protocol.
    """
    try:
        mode_norm = _normalize_ppl_mode(mode)
    except Exception as exc:
        payload = _build_ppl_result(
            mode=str(mode),
            dataset_name=dataset_name,
            split=split,
            seq_len=seq_len,
            max_examples=max_examples,
            stride=stride,
            tokenizer=tokenizer,
            full_corpus=bool(full_corpus),
            stream_mode="unknown",
            chunking="unknown",
            method="invalid_mode",
            notes="invalid_ppl_mode",
            add_special_tokens=False,
            loss_impl="hf_internal_shift",
            error=str(exc),
        )
        _log_ppl_result(payload)
        return payload

    if mode_norm == "canonical":
        canonical_split = split or "test"
        canonical_seq_len = int(seq_len or 2048)
        return compute_canonical_ppl(
            model,
            tokenizer,
            dataset_name=dataset_name,
            split=canonical_split,
            seq_len=canonical_seq_len,
            device=device,
            dtype=dtype,
            seed=seed,
            add_special_tokens=False,
        )

    return compute_proxy_ppl(
        model,
        tokenizer,
        dataset_name=dataset_name,
        split=split,
        seq_len=seq_len,
        max_examples=max_examples,
        stride=stride,
        device=device,
        dtype=dtype,
        seed=seed,
    )


def summarize_latency(latency_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not latency_info:
        return {
            "prefill_p50_ms": None,
            "prefill_p90_ms": None,
            "decode_p50_ms": None,
            "decode_p90_ms": None,
            "decode_tokens_per_sec_mean": None,
            "decode_tokens_per_sec_p50": None,
            "prompt_length": None,
            "max_new_tokens": None,
            "warmup_runs": None,
            "measured_runs": None,
        }

    prefill = latency_info.get("prefill_sec") or {}
    decode = latency_info.get("decode_sec") or {}
    tps = latency_info.get("tokens_per_sec") or {}

    return {
        "prefill_p50_ms": (prefill.get("p50") * 1000.0) if prefill.get("p50") is not None else None,
        "prefill_p90_ms": (prefill.get("p90") * 1000.0) if prefill.get("p90") is not None else None,
        "decode_p50_ms": (decode.get("p50") * 1000.0) if decode.get("p50") is not None else None,
        "decode_p90_ms": (decode.get("p90") * 1000.0) if decode.get("p90") is not None else None,
        "decode_tokens_per_sec_mean": tps.get("mean"),
        "decode_tokens_per_sec_p50": tps.get("p50"),
        "prompt_length": latency_info.get("prompt_length"),
        "max_new_tokens": latency_info.get("max_new_tokens"),
        "warmup_runs": latency_info.get("warmup_runs"),
        "measured_runs": latency_info.get("measured_runs"),
    }


def summarize_memory(memory_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not memory_info:
        return {
            "peak_allocated_bytes": None,
            "peak_allocated_gb": None,
            "resident_mem_after_load_bytes": None,
            "resident_mem_after_load_gb": None,
            "extra_peak_over_resident_bytes": None,
            "extra_peak_over_resident_gb": None,
            "device": None,
        }

    peak_bytes = memory_info.get("peak_allocated_bytes")
    if peak_bytes is None:
        peak_bytes = memory_info.get("peak_memory_bytes")
    resident_bytes = memory_info.get("resident_mem_after_load_bytes")
    if resident_bytes is None:
        resident_bytes = memory_info.get("resident_allocated_bytes")
    extra_bytes = memory_info.get("extra_peak_over_resident_bytes")
    if extra_bytes is None and peak_bytes is not None and resident_bytes is not None:
        extra_bytes = max(int(peak_bytes) - int(resident_bytes), 0)
    peak_gb = None
    resident_gb = None
    extra_gb = None
    if peak_bytes is not None:
        peak_gb = float(peak_bytes) / (1024 ** 3)
    if resident_bytes is not None:
        resident_gb = float(resident_bytes) / (1024 ** 3)
    if extra_bytes is not None:
        extra_gb = float(extra_bytes) / (1024 ** 3)

    return {
        "peak_allocated_bytes": peak_bytes,
        "peak_allocated_gb": peak_gb,
        "resident_mem_after_load_bytes": resident_bytes,
        "resident_mem_after_load_gb": resident_gb,
        "extra_peak_over_resident_bytes": extra_bytes,
        "extra_peak_over_resident_gb": extra_gb,
        "device": memory_info.get("device"),
    }


def build_bench_summary(
    size_info: Dict[str, Any],
    ppl_info: Dict[str, Any],
    latency_info: Optional[Dict[str, Any]],
    memory_info: Optional[Dict[str, Any]],
    effective_bits_per_param: Optional[float],
    notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    disk_bytes = size_info.get("total_bytes")
    disk_gb = size_info.get("total_GB")
    if disk_bytes is None:
        disk_bytes = size_info.get("estimated_bytes")
    if disk_gb is None:
        disk_gb = size_info.get("estimated_GB")

    return {
        "size": size_info,
        "ppl": ppl_info,
        "latency": summarize_latency(latency_info),
        "memory": summarize_memory(memory_info),
        "effective_bits_per_param": effective_bits_per_param,
        "disk_size_bytes": disk_bytes,
        "disk_size_GB": disk_gb,
        "notes": notes or [],
    }
