from pathlib import Path

import yaml

from scripts.run_llmc_w4_baselines import parse_ppl, render_config


def _native_config(method: str) -> dict:
    if method == "awq":
        return {
            "base": {"seed": 0},
            "model": {},
            "calib": {"name": "pileval", "download": False, "path": "old", "n_samples": 128, "bs": -1, "seq_len": 512, "preproc": "pileval_awq", "seed": 0},
            "eval": {"path": "old"},
            "quant": {"method": "Awq", "weight": {"bit": 4, "group_size": 128}},
            "save": {},
        }
    return {
        "base": {"seed": 0},
        "model": {},
        "calib": {"name": "wikitext2", "download": False, "path": "old", "n_samples": 128, "bs": 1, "seq_len": 2048, "preproc": "wikitext2_gptq", "seed": 0},
        "eval": {"path": "old"},
        "quant": {"method": "GPTQ", "weight": {"bit": 4, "group_size": 128}},
        "save": {},
    }


def test_render_config_preserves_native_calibration(tmp_path: Path) -> None:
    repo = tmp_path / "llmc"
    for method, relative in {
        "awq": "configs/quantization/methods/Awq/awq_w_only.yml",
        "gptq": "configs/quantization/methods/GPTQ/gptq_w_only.yml",
    }.items():
        source = repo / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(yaml.safe_dump(_native_config(method)), encoding="utf-8")
        rendered = render_config(
            method=method,
            model="meta-llama/Llama-3.2-1B",
            model_type="Llama",
            llmc_repo=repo,
            out_path=tmp_path / method / "config.yml",
            eval_seq_len=2048,
        )
        native = _native_config(method)["calib"]
        for key in ("name", "n_samples", "bs", "seq_len", "preproc"):
            assert rendered["calib"][key] == native[key]
        assert rendered["calib"]["download"] is True
        assert "path" not in rendered["calib"]
        assert rendered["eval"]["eval_pos"] == ["fake_quant"]


def test_parse_ppl_uses_last_result(tmp_path: Path) -> None:
    log = tmp_path / "llmc.log"
    log.write_text(
        "EVAL: ppl on wikitext2 is 9.75\nnoise\nEVAL: ppl on wikitext2 is 10.125\n",
        encoding="utf-8",
    )
    assert parse_ppl(log) == ("wikitext2", 10.125)
