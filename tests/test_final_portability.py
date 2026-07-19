from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_gitattributes_has_no_bom_and_shell_files_are_lf_only():
    attributes = (ROOT / ".gitattributes").read_bytes()
    assert not attributes.startswith(b"\xef\xbb\xbf")
    assert b"*.sh text eol=lf" in attributes
    for path in (ROOT / "scripts").glob("*.sh"):
        data = path.read_bytes()
        assert b"\r\n" not in data, path
        assert data.startswith(b"#!/usr/bin/env bash\n"), path


def test_active_runtime_code_has_no_machine_specific_mnt_d_defaults():
    forbidden = (b"/mnt/d/Abrar", b"/mnt/d/LightCompress")
    for directory in (ROOT / "seq_core", ROOT / "analysis", ROOT / "scripts"):
        for pattern in ("*.py", "*.sh"):
            for path in directory.glob(pattern):
                data = path.read_bytes()
                for value in forbidden:
                    assert value not in data, f"{value!r} remains in {path}"


def test_existing_pipeline_remains_the_orchestrator():
    text = (ROOT / "scripts" / "run_final_seq_pipeline.sh").read_text(encoding="utf-8")
    assert "run_phase()" in text
    assert "phase_gate" in text
    assert "phase_full_matrix" in text
    assert "--require-sweep-points" in text
