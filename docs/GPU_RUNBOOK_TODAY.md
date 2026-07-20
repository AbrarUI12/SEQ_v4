# GPU runbook — finish the two GPU-heavy paper tasks in one session

**Goal for today's GPU window (RTX 5090):** produce the last two pieces of evidence
the Findings paper is missing —

- **W1** — regenerate `docs/COMPARISON.md` so the SEQ rows (incl. the 3B
  `residual_max`-on-GPTQ Pareto point) appear instead of a baseline-only table
  (CPU, minutes — the blocker was the LFS pointer bug, not compute).
- **W2** — **downstream lm-eval (finding F5)** at the paper operating points
  (GPU, the biggest missing piece).

Run these **on the box that has the real result JSONs and the GPU** (your WSL /
data tree with `.venv-seq` + LightCompress). This session's code agent staged the
scripts but cannot run torch/GPU or materialize LFS.

> Time budget: do **3B first** — it carries the Pareto frontier point and the
> headline contrasts. If the window is tight, a 3B-only run still completes F5.

---

## 0. Sync the scripts onto the GPU box + materialize data

The new scripts live on branch **`claude/agent-onboarding-docs-voe832`**. The real
`runs/final/**` JSONs are LFS objects (fresh clones get 129-byte pointer stubs).

```bash
cd "/mnt/e/seq v4/SEQ-clean-v4"          # your data+GPU tree
git fetch origin claude/agent-onboarding-docs-voe832
git checkout claude/agent-onboarding-docs-voe832   # same commit as the SEQ branch today
git lfs install && git lfs pull          # <-- turns the result-JSON stubs into real data
# sanity: this must print JSON, NOT 'version https://git-lfs...'
head -c 120 runs/final/reports/gate_summary.json; echo
```

If `git lfs pull` cannot fetch the objects but this tree is the one that
*generated* the runs, the real JSONs are already on disk — just make sure the 4
staged files are present (`configs/downstream_operating_points.json`,
`scripts/run_downstream_eval.sh`, `analysis/build_downstream_table.py`, this file).

---

## 1. W1 — fix LFS tracking + regenerate the comparison table (CPU, minutes)

Small result JSONs should never have been in LFS. Migrate them out **on this data
tree only** (never in a pointer-only clone), then regenerate:

```bash
source .venv-seq/bin/activate
git lfs untrack "runs/final/**/*.json"
git add --renormalize runs/final
git add .gitattributes
git commit -m "Stop LFS-tracking result JSONs; keep LFS for model binaries only"

# regenerate the table + plots from the validated sweeps
python analysis/build_comparison.py --sweeps runs/final/sweeps \
  --baselines runs/final/staging/results/baselines.json \
  --signals act_max,act_scale,residual_rms,residual_max,greedy,greedy_indep,random,tier_alloc \
  --require-sweep-points \
  --out docs/COMPARISON.md --csv results/final_comparison.csv --json results/final_comparison.json
python analysis/plot_final_results.py --input results/final_comparison.csv --output-dir figures/final
```

**Verify W1:**
```bash
grep -c 'SEQ:' docs/COMPARISON.md            # must be large, not 0
grep -n 'residual_max' docs/COMPARISON.md    # 3B row should be marked a frontier point (★)
```
(Equivalent alternative if you prefer the staged pipeline path:
`bash scripts/run_final_seq_pipeline.sh --llmc-repo /mnt/e/LightCompress --llmc-venv /mnt/e/LightCompress/.venv-llmc --output-root runs/final --phase comparison --resume` then `--phase plots`.)

---

## 2. W2 — downstream lm-eval at the operating points (GPU)

Prerequisite: the GPTQ-4 baselines must already exist at
`runs/final/llmc/<Model>/gptq/artifacts/fake_quant_model` (they do, from the final
run). Operating points, checkpoints, and expected PPLs are declared in
`configs/downstream_operating_points.json`.

**2a. Dry-run — confirm every command/path resolves (no GPU):**
```bash
bash scripts/run_downstream_eval.sh --dry-run
```

**2b. Smoke — cheap end-to-end sanity on 3B (a few min on a 5090):**
```bash
bash scripts/run_downstream_eval.sh --models meta-llama/Llama-3.2-3B --limit 200 --resume
```
This exports the 4 SEQ checkpoints once (hqq4, resmax_gptq, greedy_gptq, best_hqq),
then lm-evals FP16 / GPTQ-4 / the four SEQ points on 200 examples/task. Expect the
`greedy_gptq` point to score far below the others (that IS finding F3 downstream).

**2c. Full run — remove `--limit`; 3B then 1B (`--resume` skips finished points):**
```bash
bash scripts/run_downstream_eval.sh --resume        # config order = 3B first, then 1B
```
Per-point outputs land in `runs/final/downstream/<Model>/<point>/` (lm-eval results
+ `--log_samples` logs + a `seq_meta.json` sidecar).

---

## 3. Build the F5 table (CPU — can run while GPU continues)

```bash
python analysis/build_downstream_table.py \
  --root runs/final/downstream --config configs/downstream_operating_points.json \
  --out docs/DOWNSTREAM.md --csv results/downstream.csv --json results/downstream.json
```
Produces `docs/DOWNSTREAM.md` (per-task accuracy + macro-avg per model) and
**paired-bootstrap 95% CIs** for the three contrasts (resmax_gptq vs gptq4,
greedy_gptq vs gptq4, best_hqq vs hqq4). These drop straight into the paper's §7 (F5).

---

## 4. (Optional) reload-validation on an exported checkpoint

Confirms an exported SEQ checkpoint reproduces its measured PPL (expected values
are in the config):
```bash
python scripts/validate_saved_seq_reload.py \
  runs/final/downstream/checkpoints/Llama-3.2-3B/resmax_gptq --expected 8.099 --tolerance 0.5
```

---

## 5. Done-for-today checklist

- [ ] `docs/COMPARISON.md` has many `SEQ:` rows and the 3B `residual_max(gptq…)` ★ point.
- [ ] `runs/final/downstream/<Model>/<point>/lm_eval/` populated for both models.
- [ ] `docs/DOWNSTREAM.md` renders; greedy_gptq−gptq4 CI excludes 0 (F3), resmax_gptq−gptq4 ≈ 0 or positive (F4).
- [ ] Commit + push to `claude/agent-onboarding-docs-voe832`. Note `runs/` and
      `results/` are gitignored (only specific files are force-tracked), so **new**
      outputs there need `-f`; `docs/` and `figures/final/` commit normally:
      ```bash
      git add docs/COMPARISON.md docs/DOWNSTREAM.md figures/final results/final_comparison.csv results/final_comparison.json
      git add -f results/downstream.json results/downstream.csv          # new files under ignored results/
      git commit -m "W1: regenerate comparison table; W2: downstream lm-eval (F5)"
      git push -u origin claude/agent-onboarding-docs-voe832
      ```
      The aggregated `docs/DOWNSTREAM.md` + `results/downstream.json` already carry
      the F5 numbers and paired CIs, so the raw `runs/final/downstream/` tree
      (per-point results + `--log_samples` JSONL) can stay **local scratch** — do
      not commit it, and never commit `runs/final/downstream/checkpoints/` (multi-GB
      fake-quant model dirs; `*.safetensors` is ignored anyway).

Once these land, the paper's F5 section and the Pareto table are complete — only
the CPU writing tasks (W3 consolidation, related work, stats appendix) remain, and
those don't need the GPU.
