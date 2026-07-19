# Agent onboarding / handoff — start here

You are a new agent picking up the **SEQ** project from git. Read this top to
bottom once; it takes ~10 minutes and prevents every trap the last agents hit.

---

## 1. First 5 minutes — sync the repo correctly

```bash
git clone <this-repo-url> SEQ-clean-v4 && cd SEQ-clean-v4      # or: cd existing clone
git lfs install                                               # REQUIRED (see §5 trap A)
git checkout claude/seq-compression-quantization-yktihv
git fetch origin && git reset --hard origin/claude/seq-compression-quantization-yktihv
git lfs pull                                                  # materialize result JSONs
```
Verify LFS actually pulled (a real file, not a 129-byte pointer):
```bash
head -c 40 runs/final/sweeps/hqq/Llama-3.2-1B/residual_max/seed-1234/channel_pareto.json
# GOOD: '{"model": ...'    BAD: 'version https://git-lfs.github.com/...'  -> rerun git lfs pull
```

## 2. Which environment are you? (this determines what you can do)

| capability | **Code/analysis agent** (no GPU, no torch) | **GPU runner** (WSL RTX 4090) |
|---|---|---|
| read code, docs, committed results | yes | yes |
| `python -m compileall`, stdlib tests | yes | yes |
| regenerate `docs/COMPARISON.md` (CPU) | yes *after `git lfs pull`* | yes |
| write/fix code, docs, scripts | yes | yes |
| run `seq_core.channel_sweep` / sweeps | no (no torch/GPU) | yes |
| run LightCompress (GPTQ/AWQ/RTN) | no | yes |
| downstream lm-eval | no | yes |

If `python -c "import torch"` fails, you are the **code agent** — do W1 + code work
below and hand GPU tasks to the runner. Don't try to run sweeps.

## 3. Setup per role

**Code agent:** nothing to install. Smoke-test (torch-free):
```bash
python -m compileall -q seq_core analysis scripts
python tests/test_channel_utils.py && python tests/test_greedy_select.py && python tests/test_comparison.py
```

**GPU runner (WSL):** you need `.venv-seq` (torch + transformers + hqq + lm-eval)
and a pinned LightCompress. If `.venv-seq` already exists, run:
```bash
export HF_TOKEN=...      # gated Llama-3.2 access; never commit/log it
bash scripts/bootstrap_final_environment.sh --llmc-repo /mnt/e/LightCompress
```
NOTE: `bootstrap_final_environment.sh` **assumes `.venv-seq` already exists** (it adds
hqq+pytest and sets up the LightCompress venv). On a brand-new machine, first create
`.venv-seq` with the SEQ deps (torch, transformers, hqq==0.2.8.post1, datasets,
lm-eval, safetensors, matplotlib), then run bootstrap. LightCompress must live at a
**whitespace-free** path (its tooling breaks on spaces); pinned commit is in the
script. Version pins that make LLMC work are in `configs/llmc_constraints.txt`.

## 4. Where everything is

- **Start-here state + all real numbers + roadmap:** `docs/PROJECT_STATUS_AND_ROADMAP.md`
- **Paper draft (v0):** `docs/FINDINGS_PAPER.md`
- **Gate verdict + selector PPLs:** `runs/final/reports/GATE_SUMMARY.md`
- **Comparison table (currently baseline-only — regenerate, W1):** `docs/COMPARISON.md`
- **Matrix spec (the experiment grid):** `configs/final_comparison_matrix.json`
- **Pipeline orchestrator:** `scripts/run_final_seq_pipeline.sh` (phases: preflight →
  llmc baselines → gptq validate → gate → gate_summary → full_matrix → validate →
  comparison → plots → publish; `--resume`, `--publish`).
- **Core code:** `seq_core/channel_sweep.py` (sweep driver), `channel_protect.py`
  (column-split FP16 protection), `greedy_select.py` (OMP + greedy_indep ablation,
  `exact_k` for matched-bit budgets), `gptq_llmc_base.py` (load LLMC fake-quant base),
  `storage_accounting.py` (weight-only bits axis), `channel_utils.py` (pure helpers).
- **Analysis:** `analysis/build_comparison.py` (matched-bit Pareto; loud on LFS
  pointers), `final_matrix.py` (gate/validate), `plot_final_results.py`.
- **Earlier module-level audit:** `docs/FINDINGS_run{1..6}.md`.

## 5. Traps that already cost us days — do not rediscover

- **A. Result JSONs are in Git LFS.** `.gitattributes` LFS-tracks
  `runs/final/**/*.json`. Fresh clones get pointer stubs; analysis tools then see no
  data and can silently produce a **baseline-only comparison**. Always `git lfs pull`.
  Pending cleanup: migrate these small JSONs out of LFS **on a tree that holds the
  real files** — `git lfs untrack "runs/final/**/*.json" && git add --renormalize
  runs/final && git commit`. Never renormalize in a pointer-only clone (commits stubs).
- **B. History was force-rewritten** (isolated results clone). If your branch shows
  "unrelated histories", `git reset --hard origin/<branch>` (your gitignored `runs/`
  and LFS files survive).
- **C. Accounting axis = weight-only bits/param** (embeddings/lm_head excluded, = GPTQ-4
  at 4.0). Never compare on the full-model average (it charges a 4-bit base ~7 bits).
- **D. `greedy`/`greedy_indep` on a GPTQ base are catastrophic** (PPL 40–107) — this
  is a *finding*, not a bug (protection breaks GPTQ's error compensation). Keep those
  rows; don't "fix" them.
- **E. `exact_k=True`** is required for matched-bit comparisons so selectors spend the
  full budget (already wired in the pipeline).
- **F. CRLF / spaced paths** (Windows↔WSL): shell files are LF-only (`.gitattributes`);
  keep LightCompress off spaced paths.
- **G. Drive WSL from a real WSL terminal**, not PowerShell-wrapping bash (nested-quote
  and CRLF corruption bit us repeatedly). Use `nohup ... &` for long runs.

## 6. Current state (2026-07, one paragraph)
The original goal (compress to 5–7 bits ≤ FP16 PPL) is **falsified**. Framing =
**audit**. Findings (Llama-3.2-1B/3B, matched weight bits): F1 activation-outlier
protection helps a data-free RTN/HQQ base (all selectors ≫ random); F2 interaction-
aware greedy does not beat simple selection (≤0.03 PPL); **F3 (headline) protection
is antagonistic to error-compensated (GPTQ) bases** — residual-driven selection is
catastrophic; F4 the base quantizer is the Pareto ceiling (1B: no SEQ frontier point;
3B: one point, `residual_max`-on-GPTQ 4.82b/8.099 beats GPTQ-4 8.304). Verdict and
numbers: `docs/PROJECT_STATUS_AND_ROADMAP.md`.

## 7. What to do next (priority order)
1. **W1 (code agent CAN do this):** after `git lfs pull`, regenerate the comparison
   table so SEQ rows appear (the published one is baseline-only). Command in
   `PROJECT_STATUS_AND_ROADMAP.md §3 W1`; verify `grep -c 'SEQ:' docs/COMPARISON.md`
   > 0 and the 3B `residual_max(gptq_llmc)` point is marked with a star. Commit + push.
2. **W2 (GPU):** build `scripts/run_downstream_eval.sh` (+ pipeline phase) and run
   lm-eval (hellaswag/arc/piqa/winogrande) at the operating points. Biggest gap.
3. **W3:** consolidate `FINDINGS_run{1..6}` + related-work table into the paper.
4. **W4 (optional GPU):** 8B and/or Qwen2.5-3B for the two headline findings.
5. **Track B (optional, higher-risk):** prototype **protect-then-recompensate** (pick
   FP16 columns first, then GPTQ-compensate the rest) — the one lead F3 motivates.

## 8. Conventions
- Commit/push only to `claude/seq-compression-quantization-yktihv`. No PRs unless asked.
- Follow **your own session's** commit trailers (don't copy another session's URL).
- Never put a raw model identifier in commits/PRs/code/artifacts.
- LFS is for model binaries only (`*.safetensors/*.bin/*.pt/*.pth`) — not for JSONs.
- Two agents may work concurrently: `git fetch` before you start, prefer editing
  distinct files, and keep the history linear.
