"""PPL-only compatibility entrypoint for LLMC.

LLMC eagerly imports its optional lmms-eval VQA integration even when the run
only requests perplexity.  Some supported Python versions cannot install the
old lmms-eval dependency pins.  Supply inert VQA symbols so the unused module
can import, then execute LLMC's real entrypoint unchanged.
"""

from __future__ import annotations

import runpy
import sys
from types import ModuleType


def _unused(*_args, **_kwargs):
    raise RuntimeError("lmms-eval is unavailable in the PPL-only LLMC launcher")


def _empty_datetime(*_args, **_kwargs) -> str:
    # eval_vqa evaluates this optional helper in a function default at import.
    return ""


class _UnusedLmmsBase:
    """Base class sufficient for eager-imported, unused VQA model wrappers."""

    pass


def _install_unused_lmms_eval_stubs() -> None:
    parent = ModuleType("lmms_eval")
    parent.__path__ = []  # type: ignore[attr-defined]
    modules: dict[str, dict[str, object]] = {
        "lmms_eval.evaluator": {"evaluate": _unused},
        "lmms_eval.evaluator_utils": {"run_task_tests": _unused},
        "lmms_eval.loggers": {"__path__": []},
        "lmms_eval.loggers.evaluation_tracker": {"EvaluationTracker": type("EvaluationTracker", (), {})},
        "lmms_eval.tasks": {
            "TaskManager": type("TaskManager", (), {}),
            "get_task_dict": _unused,
        },
        "lmms_eval.utils": {
            "get_datetime_str": _empty_datetime,
            "make_table": _unused,
            "simple_parse_args_string": _unused,
        },
        "lmms_eval.api": {"__path__": []},
        "lmms_eval.api.model": {"lmms": _UnusedLmmsBase},
        "lmms_eval.api.instance": {"Instance": type("Instance", (), {})},
        "lmms_eval.models": {"__path__": []},
        "lmms_eval.models.llava": {"Llava": type("Llava", (), {})},
        "lmms_eval.models.internvl2": {"InternVL2": type("InternVL2", (), {})},
        "lmms_eval.models.llava_onevision": {"Llava_OneVision": type("Llava_OneVision", (), {})},
        "lmms_eval.models.llava_hf": {"LlavaHf": type("LlavaHf", (), {})},
        "lmms_eval.models.qwen2_vl": {"Qwen2_VL": type("Qwen2_VL", (), {})},
        "lmms_eval.models.qwen2_5_vl": {"Qwen2_5_VL": type("Qwen2_5_VL", (), {})},
        "lmms_eval.models.video_llava": {"VideoLLaVA": type("VideoLLaVA", (), {})},
    }
    sys.modules["lmms_eval"] = parent
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        sys.modules[name] = module


if __name__ == "__main__":
    _install_unused_lmms_eval_stubs()
    runpy.run_module("llmc.__main__", run_name="__main__")
