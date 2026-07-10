#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    import run_compare_matrix as runner
    from seq_core.precision_policy import (
        compute_percentile_ranks,
        is_attention_output_projection,
        is_attention_projection,
        is_gate_down_projection,
        is_lm_head,
        tier_rank,
        upgrade_tier,
    )

    methods = ["seq", "seq_v0", "seq_v1", "seq_v2", "seq_v3", "seq_v4", "seq_v5"]
    expected = {
        "seq": "SEQ-v0",
        "seq_v0": "SEQ-v0",
        "seq_v1": "SEQ-v1",
        "seq_v2": "SEQ-v2",
        "seq_v3": "SEQ-v3",
        "seq_v4": "SEQ-v4",
        "seq_v5": "SEQ-v5",
    }

    config_path = ROOT / "experiments_seq_variants.yaml"
    assert config_path.exists(), f"missing {config_path}"
    with config_path.open("r") as f:
        base_config = yaml.safe_load(f) or {}

    validated = runner._validate_methods(methods)
    assert validated == methods, validated
    assert len(validated) == len(set(validated)), "duplicate output method names"

    for method in methods:
        assert runner.is_seq_method(method), f"not recognized as SEQ method: {method}"
        path = runner.seq_variant_config_path(method, base_config, ROOT)
        assert path.exists(), f"missing variant config for {method}: {path}"
        with path.open("r") as f:
            parsed = yaml.safe_load(f) or {}
        assert isinstance(parsed, dict), f"variant config is not a mapping: {path}"
        assert parsed.get("variant_name") == expected[method], (method, parsed.get("variant_name"))
        loaded = runner.load_seq_variant_config(method, base_config, ROOT)
        assert loaded["variant_name"] == expected[method], loaded

    for idx in range(6):
        folder = ROOT / "seq_variants" / f"SEQ-v{idx}"
        assert (folder / "README.md").exists(), folder
        assert (folder / "variant_config.yaml").exists(), folder

    assert tier_rank("int4") < tier_rank("int8") < tier_rank("fp16")
    assert upgrade_tier("int4", "int8") == "int8"
    assert upgrade_tier("fp16", "int8") == "fp16"
    assert is_attention_output_projection("model.layers.0.self_attn.o_proj")
    assert is_attention_projection("model.layers.0.self_attn.q_proj")
    assert is_gate_down_projection("model.layers.0.mlp.down_proj")
    assert is_lm_head("lm_head")
    assert compute_percentile_ranks({"b": 2.0, "a": 1.0}) == {"a": 0.0, "b": 1.0}

    print("SEQ variant smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
