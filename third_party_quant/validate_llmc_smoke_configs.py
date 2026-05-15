from __future__ import annotations

from pathlib import Path, PurePosixPath
import json
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = SCRIPT_DIR / "llmc_smoke_configs"
ALLOWED_SAVE_ROOT = PurePosixPath("/mnt/e/SEQ_Clean/results/llmc_smoke")
CONFIG_NAMES = (
    "gptq_opt125m_smoke.yml",
    "smoothquant_opt125m_smoke.yml",
    "awq_opt125m_smoke.yml",
    "rtn_opt125m_smoke.yml",
    "llm_int8_opt125m_smoke.yml",
)


def config_contains_pileval(config: object) -> bool:
    return "pileval" in json.dumps(config, sort_keys=True).lower()


def parse_scalar(value: str) -> object:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
        return value[1:-1]
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_simple_yaml(config_path: Path) -> dict[str, object]:
    root: dict[str, object] = {}
    stack: list[tuple[int, object]] = [(-1, root)]

    with config_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            indent = len(line) - len(line.lstrip(" "))
            while len(stack) > 1 and indent <= stack[-1][0]:
                stack.pop()

            container = stack[-1][1]
            if stripped.startswith("- "):
                if not isinstance(container, list):
                    raise ValueError(f"Unexpected list item in {config_path.as_posix()}")
                container.append(parse_scalar(stripped[2:].strip()))
                continue

            key, sep, value = stripped.partition(":")
            if not sep:
                raise ValueError(f"Invalid line in {config_path.as_posix()}: {stripped}")
            key = key.strip()
            value = value.strip()

            if not isinstance(container, dict):
                raise ValueError(f"Unexpected mapping in {config_path.as_posix()}: {stripped}")

            if not value:
                next_container: object = [] if key == "eval_pos" else {}
                container[key] = next_container
                stack.append((indent, next_container))
            else:
                container[key] = parse_scalar(value)

    return root


def validate_config(config_path: Path) -> tuple[bool, list[str]]:
    errors: list[str] = []

    config = load_simple_yaml(config_path)

    calib = config.get("calib", {})
    eval_cfg = config.get("eval", {})
    save_cfg = config.get("save", {})

    if calib.get("name") != "wikitext2":
        errors.append("calib.name must be wikitext2")
    if calib.get("n_samples", 0) > 4:
        errors.append("calib.n_samples must be <= 4")
    if calib.get("seq_len", 0) > 128:
        errors.append("calib.seq_len must be <= 128")
    if eval_cfg.get("seq_len", 0) > 128:
        errors.append("eval.seq_len must be <= 128")
    if eval_cfg.get("inference_per_block") is not True:
        errors.append("eval.inference_per_block must be true")
    if save_cfg.get("save_vllm") is not False:
        errors.append("save.save_vllm must be false")
    if save_cfg.get("save_fake") is not False:
        errors.append("save.save_fake must be false")
    if save_cfg.get("save_trans") is not False:
        errors.append("save.save_trans must be false")
    if config.get("model", {}).get("path") != "facebook/opt-125m":
        errors.append("model.path must be facebook/opt-125m for current OPT smoke configs")
    if config.get("model", {}).get("tokenizer_mode") != "slow":
        errors.append("model.tokenizer_mode must be slow")

    save_path_raw = save_cfg.get("save_path")
    if not isinstance(save_path_raw, str):
        errors.append("save.save_path must be a string")
    else:
        save_path = PurePosixPath(save_path_raw)
        try:
            save_path.relative_to(ALLOWED_SAVE_ROOT)
        except ValueError:
            errors.append(
                f"save.save_path must be under {ALLOWED_SAVE_ROOT.as_posix()}/"
            )

    if config_contains_pileval(config):
        errors.append("config must not reference pileval anywhere")

    return not errors, errors


def main() -> int:
    overall_ok = True

    for config_name in CONFIG_NAMES:
        config_path = CONFIG_DIR / config_name
        if not config_path.is_file():
            overall_ok = False
            print(f"FAIL {config_name}: file not found at {config_path.as_posix()}")
            continue

        ok, errors = validate_config(config_path)
        if ok:
            print(f"PASS {config_name}")
        else:
            overall_ok = False
            print(f"FAIL {config_name}")
            for error in errors:
                print(f"  - {error}")

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
