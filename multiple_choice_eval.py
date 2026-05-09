#!/usr/bin/env python3
import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

LOGGER = logging.getLogger(__name__)

try:
    from datasets import get_dataset_config_names, load_dataset

    DATASETS_AVAILABLE = True
except Exception:
    get_dataset_config_names = None
    load_dataset = None
    DATASETS_AVAILABLE = False


CHOICE_LABELS = ["A", "B", "C", "D", "E", "F"]


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _resolve_model_max_length(model: torch.nn.Module, tokenizer, fallback: int = 2048) -> int:
    config_max = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    tok_max = getattr(tokenizer, "model_max_length", None)
    values = []
    for val in (config_max, tok_max):
        try:
            val_i = int(val)
        except Exception:
            continue
        if val_i > 0 and val_i < 10_000_000:
            values.append(val_i)
    if not values:
        return fallback
    return max(32, min(values))


def _normalize_answer_label(answer: Any, num_choices: int) -> Optional[str]:
    if isinstance(answer, int):
        if 0 <= answer < num_choices and answer < len(CHOICE_LABELS):
            return CHOICE_LABELS[answer]
        return None
    if isinstance(answer, str):
        text = answer.strip().upper()
        if text in CHOICE_LABELS[:num_choices]:
            return text
        if text.isdigit():
            idx = int(text) - 1
            if 0 <= idx < num_choices and idx < len(CHOICE_LABELS):
                return CHOICE_LABELS[idx]
    return None


def _score_label_continuation(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    label: str,
    device: str,
    max_context_tokens: int,
) -> float:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(f" {label}", add_special_tokens=False)["input_ids"]
    if not target_ids:
        target_ids = tokenizer(label, add_special_tokens=False)["input_ids"]
    if not target_ids:
        return float("-inf")

    room = max(1, int(max_context_tokens) - len(target_ids))
    prompt_ids = prompt_ids[-room:]
    input_ids = torch.tensor([prompt_ids + target_ids], dtype=torch.long, device=device)
    labels = torch.full_like(input_ids, -100)
    labels[0, len(prompt_ids) :] = torch.tensor(target_ids, dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
    return float(-outputs.loss.item() * len(target_ids))


def _score_multiple_choice_example(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    choices: Sequence[str],
    answer_label: str,
    device: str,
    max_context_tokens: int,
) -> Dict[str, Any]:
    labels = CHOICE_LABELS[: len(choices)]
    scores = {}
    best_label = None
    best_score = None
    for label in labels:
        score = _score_label_continuation(model, tokenizer, prompt, label, device, max_context_tokens)
        scores[label] = score
        if best_score is None or score > best_score:
            best_score = score
            best_label = label
    return {
        "predicted_label": best_label,
        "correct_label": answer_label,
        "correct": best_label == answer_label,
        "scores": scores,
    }


def _build_mc_prompt(question: str, choices: Sequence[str]) -> str:
    lines = [f"Question: {question.strip()}"]
    for idx, choice in enumerate(choices):
        lines.append(f"{CHOICE_LABELS[idx]}. {str(choice).strip()}")
    lines.append("Answer:")
    return "\n".join(lines)


def _iter_first_n_examples(rows: Iterable[Dict[str, Any]], max_examples: Optional[int]) -> List[Dict[str, Any]]:
    selected = []
    for row in rows:
        selected.append(row)
        if max_examples is not None and len(selected) >= max_examples:
            break
    return selected


def _load_arc_examples(task_name: str, split: str, max_examples: Optional[int]) -> Tuple[str, List[Dict[str, Any]]]:
    dataset_name = "ai2_arc"
    config_name = "ARC-Easy" if task_name == "arc_easy" else "ARC-Challenge"
    ds = load_dataset(dataset_name, config_name, split=split)
    rows = []
    for row in ds:
        question = row.get("question")
        choices = row.get("choices") or {}
        texts = choices.get("text") if isinstance(choices, dict) else None
        labels = choices.get("label") if isinstance(choices, dict) else None
        answer_key = row.get("answerKey")
        if not question or not texts or not labels:
            continue
        ordered = sorted(zip(labels, texts), key=lambda item: str(item[0]))
        normalized_choices = [text for _, text in ordered][: len(CHOICE_LABELS)]
        normalized_labels = [str(label).upper() for label, _ in ordered]
        answer_label = str(answer_key).upper().strip() if answer_key is not None else None
        if answer_label and answer_label.isdigit():
            idx = int(answer_label) - 1
            answer_label = CHOICE_LABELS[idx] if 0 <= idx < len(normalized_choices) else None
        if answer_label not in normalized_labels:
            continue
        remapped_label = CHOICE_LABELS[normalized_labels.index(answer_label)]
        rows.append(
            {
                "question": question,
                "choices": normalized_choices,
                "answer_label": remapped_label,
            }
        )
        if max_examples is not None and len(rows) >= max_examples:
            break
    return f"{dataset_name}/{config_name}", rows


def _load_piqa_examples(split: str, max_examples: Optional[int]) -> Tuple[str, List[Dict[str, Any]]]:
    dataset_name = "piqa"
    ds = load_dataset(dataset_name, split=split)
    rows = []
    for row in ds:
        goal = row.get("goal")
        sol1 = row.get("sol1")
        sol2 = row.get("sol2")
        label = row.get("label")
        if not isinstance(goal, str) or not isinstance(sol1, str) or not isinstance(sol2, str):
            continue
        answer_label = _normalize_answer_label(label, 2)
        if answer_label is None:
            continue
        rows.append({"question": goal, "choices": [sol1, sol2], "answer_label": answer_label})
        if max_examples is not None and len(rows) >= max_examples:
            break
    return dataset_name, rows


def _resolve_mmlu_source(dataset_candidates: Sequence[str]) -> Tuple[Optional[str], Optional[List[str]], Optional[str]]:
    if not DATASETS_AVAILABLE:
        return None, None, "datasets_unavailable"
    for dataset_name in dataset_candidates:
        try:
            configs = list(get_dataset_config_names(dataset_name))
        except Exception as exc:
            LOGGER.warning("Unable to list MMLU configs for %s: %s", dataset_name, exc)
            continue
        usable = [
            cfg
            for cfg in configs
            if cfg and cfg.lower() not in {"all", "auxiliary_train"}
        ]
        if usable:
            return dataset_name, sorted(usable), None
    return None, None, f"mmlu_configs_unavailable:{','.join(dataset_candidates)}"


def _load_mmlu_rows(
    *,
    dataset_candidates: Sequence[str],
    split: str,
    max_subjects: Optional[int],
    max_examples_per_subject: Optional[int],
) -> Tuple[Optional[str], List[Tuple[str, Dict[str, Any]]], Optional[str]]:
    dataset_name, subjects, error = _resolve_mmlu_source(dataset_candidates)
    if error:
        return None, [], error
    selected_subjects = subjects[: max_subjects or len(subjects)]
    rows: List[Tuple[str, Dict[str, Any]]] = []
    for subject in selected_subjects:
        try:
            ds = load_dataset(dataset_name, subject, split=split)
        except Exception as exc:
            LOGGER.warning("Skipping MMLU subject %s from %s: %s", subject, dataset_name, exc)
            continue
        seen = 0
        for row in ds:
            question = row.get("question")
            choices = row.get("choices")
            answer = row.get("answer")
            if not isinstance(question, str) or not isinstance(choices, (list, tuple)):
                continue
            answer_label = _normalize_answer_label(answer, len(choices))
            if answer_label is None:
                continue
            rows.append(
                (
                    subject,
                    {
                        "question": question,
                        "choices": list(choices)[: len(CHOICE_LABELS)],
                        "answer_label": answer_label,
                    },
                )
            )
            seen += 1
            if max_examples_per_subject is not None and seen >= max_examples_per_subject:
                break
    if not rows:
        return dataset_name, [], "mmlu_no_examples_loaded"
    return dataset_name, rows, None


def run_mmlu_eval(
    model: torch.nn.Module,
    tokenizer,
    out_path: str,
    *,
    device: str,
    split: str = "test",
    max_subjects: Optional[int] = 8,
    max_examples_per_subject: Optional[int] = 4,
    dataset_candidates: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "task": "mmlu",
        "accuracy": None,
        "num_examples": 0,
        "num_subjects": 0,
        "dataset_name": None,
        "split": split,
        "error": None,
        "subjects": [],
    }
    if not DATASETS_AVAILABLE:
        payload["error"] = "datasets_unavailable"
        _write_json(out_path, payload)
        return payload

    dataset_candidates = list(dataset_candidates or ["cais/mmlu", "hendrycks_test"])
    dataset_name, rows, error = _load_mmlu_rows(
        dataset_candidates=dataset_candidates,
        split=split,
        max_subjects=max_subjects,
        max_examples_per_subject=max_examples_per_subject,
    )
    payload["dataset_name"] = dataset_name
    if error:
        payload["error"] = error
        _write_json(out_path, payload)
        return payload

    max_context_tokens = _resolve_model_max_length(model, tokenizer)
    subject_stats: Dict[str, Dict[str, Any]] = {}
    results = []
    correct = 0
    for subject, row in rows:
        prompt = _build_mc_prompt(row["question"], row["choices"])
        scored = _score_multiple_choice_example(
            model,
            tokenizer,
            prompt=prompt,
            choices=row["choices"],
            answer_label=row["answer_label"],
            device=device,
            max_context_tokens=max_context_tokens,
        )
        scored["subject"] = subject
        scored["question"] = row["question"]
        scored["choices"] = row["choices"]
        results.append(scored)
        stats = subject_stats.setdefault(subject, {"correct": 0, "num_examples": 0})
        stats["num_examples"] += 1
        if scored["correct"]:
            correct += 1
            stats["correct"] += 1

    payload["accuracy"] = correct / max(len(results), 1)
    payload["num_examples"] = len(results)
    payload["num_subjects"] = len(subject_stats)
    payload["subjects"] = [
        {
            "subject": subject,
            "accuracy": stats["correct"] / max(stats["num_examples"], 1),
            "num_examples": stats["num_examples"],
        }
        for subject, stats in sorted(subject_stats.items())
    ]
    payload["results"] = results
    _write_json(out_path, payload)
    return payload


def _load_zero_shot_task_rows(task_name: str, split: str, max_examples: Optional[int]) -> Tuple[str, List[Dict[str, Any]]]:
    task_name = str(task_name).strip().lower()
    if task_name in {"arc_easy", "arc_challenge"}:
        return _load_arc_examples(task_name, split=split, max_examples=max_examples)
    if task_name == "piqa":
        return _load_piqa_examples(split=split, max_examples=max_examples)
    raise ValueError(f"Unsupported zero-shot task: {task_name}")


def run_zero_shot_suite(
    model: torch.nn.Module,
    tokenizer,
    out_path: str,
    *,
    device: str,
    tasks: Sequence[Dict[str, Any]],
    default_max_examples: Optional[int] = 64,
    default_split: str = "validation",
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "tasks": [],
        "mean_accuracy": None,
        "micro_accuracy": None,
        "num_tasks": 0,
        "num_examples": 0,
        "error": None,
    }
    if not DATASETS_AVAILABLE:
        payload["error"] = "datasets_unavailable"
        _write_json(out_path, payload)
        return payload

    max_context_tokens = _resolve_model_max_length(model, tokenizer)
    total_correct = 0
    total_examples = 0
    task_accuracies = []

    for task_cfg in tasks:
        task_name = str(task_cfg.get("name", "")).strip().lower()
        if not task_name:
            continue
        split = str(task_cfg.get("split", default_split))
        max_examples = task_cfg.get("max_examples", default_max_examples)
        try:
            max_examples = None if max_examples is None else int(max_examples)
        except Exception:
            max_examples = default_max_examples
        task_summary: Dict[str, Any] = {
            "task": task_name,
            "split": split,
            "max_examples": max_examples,
            "accuracy": None,
            "num_examples": 0,
            "dataset_name": None,
            "error": None,
            "results": [],
        }
        try:
            dataset_name, rows = _load_zero_shot_task_rows(task_name, split=split, max_examples=max_examples)
        except Exception as exc:
            task_summary["error"] = str(exc)
            payload["tasks"].append(task_summary)
            continue

        task_summary["dataset_name"] = dataset_name
        correct = 0
        for row in rows:
            prompt = _build_mc_prompt(row["question"], row["choices"])
            scored = _score_multiple_choice_example(
                model,
                tokenizer,
                prompt=prompt,
                choices=row["choices"],
                answer_label=row["answer_label"],
                device=device,
                max_context_tokens=max_context_tokens,
            )
            scored["question"] = row["question"]
            scored["choices"] = row["choices"]
            task_summary["results"].append(scored)
            if scored["correct"]:
                correct += 1
        task_summary["num_examples"] = len(task_summary["results"])
        if task_summary["num_examples"] > 0:
            task_summary["accuracy"] = correct / task_summary["num_examples"]
            task_accuracies.append(task_summary["accuracy"])
            total_correct += correct
            total_examples += task_summary["num_examples"]
        else:
            task_summary["error"] = task_summary["error"] or "zero_shot_no_examples_loaded"
        payload["tasks"].append(task_summary)

    payload["num_tasks"] = len(payload["tasks"])
    payload["num_examples"] = total_examples
    if task_accuracies:
        payload["mean_accuracy"] = sum(task_accuracies) / len(task_accuracies)
    if total_examples > 0:
        payload["micro_accuracy"] = total_correct / total_examples
    if not payload["tasks"]:
        payload["error"] = "zero_shot_no_tasks_configured"
    _write_json(out_path, payload)
    return payload
