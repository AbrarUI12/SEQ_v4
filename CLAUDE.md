# SEQ — project brief for agents (read `docs/AGENT_ONBOARDING.md` first)

**What this is:** a research project on per-channel mixed-precision **weight
quantization** for LLMs. Original goal (compress to 5–7 effective bits ≤ FP16 PPL)
has been **falsified by our own data**; the project is now a rigorous **audit /
findings paper**. Current verdict: **audit** (not a SOTA method).

**Branch:** `claude/seq-compression-quantization-yktihv` — commit/push here only.

**Two environments (know which you are):**
- **Code/analysis agent (no GPU, no torch)** — you can read code, run the pure-stdlib
  tests, `py_compile`, analyze committed results JSON, write/fix code, and
  regenerate `docs/COMPARISON.md` (CPU-only) *after* materializing LFS (below). You
  **cannot** run `seq_core.channel_sweep`, LightCompress, or anything torch/GPU.
- **GPU runner (WSL, RTX 4090, `.venv-seq` + LightCompress)** — runs the pipeline
  `scripts/run_final_seq_pipeline.sh`. This is where sweeps/baselines are produced.

**CRITICAL — Git LFS gotcha:** `.gitattributes` LFS-tracks `runs/final/**/*.json`,
so in a fresh clone the result JSONs are **pointer stubs**, not data. Run
`git lfs install && git lfs pull` after cloning, or the analysis tools see no data.
(A pending cleanup migrates these small JSONs out of LFS — see the onboarding doc.)
The branch history was **force-rewritten** once; if your tree diverges, sync with
`git fetch && git reset --hard origin/<branch> && git lfs pull`.

**Current state (one line):** framing = audit; findings F1–F5 established; the
published `COMPARISON.md` is baseline-only due to the LFS bug and must be
regenerated (top TODO, W1). Full detail + numbers: `docs/PROJECT_STATUS_AND_ROADMAP.md`.
Paper draft: `docs/FINDINGS_PAPER.md`.

**Conventions:** don't open PRs unless asked; keep large model artifacts out of git
(LFS is for `*.safetensors/*.bin/*.pt/*.pth` only); follow your own session's commit
trailers; never put a raw model identifier into commits/PRs/code.
