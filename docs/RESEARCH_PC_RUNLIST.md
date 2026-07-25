# Research-PC run list (weekly GPU window)

Copy-paste commands for the research PC (WSL, RTX 5090 + 4090, `.venv-seq`, LightCompress
+ regenerated GPTQ `fake_quant_model` bases). This file is tracked, so it travels with
`git pull origin main` — it is the source of truth for GPU-side runs, replacing ad-hoc
copy-paste blocks.

Before anything: `git pull origin main` and `source .venv-seq/bin/activate`.

> [!WARNING]
> **Never point `--out_dir` at `/tmp` on WSL.** WSL's `/tmp` is tmpfs and is wiped when
> the WSL instance idles or is shut down (`wsl --shutdown`), so results written there can
> vanish before you commit them. Always write under the repo (an NTFS mount) as shown
> below, and `tee` the console log for provenance. `results/` is `.gitignore`d, so the
> small text artifacts must be **force-added** (`git add -f`); never add model weights or
> any `fake_quant_model/` directory.

After each item: commit result JSON/logs (never weights), `git push origin main`. The
local box then pulls and rebuilds `COMPARISON.md`/`DOWNSTREAM.md`/figures and the paper.

---

## ⭐ NEXT SESSION (2026-07-26) — STEP 0: DIAGNOSE lm_eval (blocks ALL downstream)

Both lm-eval steps last time died the instant `lm_eval` was invoked (no output, no results),
so there are **zero** downstream numbers. Nothing else downstream can proceed until this is
fixed. This diagnosis is independent of the pending local code fixes — run it whenever.

```bash
cd /mnt/d/Abrar/SEQ/seq_v4 && git pull origin main
PY="$PWD/.venv-seq/bin/python"

# 1) Is lm_eval importable, and what version?
"$PY" -c "import lm_eval,sys; print('lm_eval', lm_eval.__version__, '| py', sys.version.split()[0])"; echo "import_exit=$?"

# 2) Minimal DIRECT run (1 task, 4 examples) with full stderr to console — surfaces the real
#    error the pipeline's tee swallowed. Uses the FP16 base (no custom checkpoint) to isolate
#    lm_eval itself from any export issue.
mkdir -p results/lm_eval_diag_out
"$PY" -m lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-3.2-3B,dtype=float16 \
  --tasks lambada_openai --limit 4 --batch_size 1 --device cuda:0 \
  --output_path results/lm_eval_diag_out 2>&1 | tee results/lm_eval_diag.log; echo "eval_exit=${PIPESTATUS[0]}"
```

**Report back:** the `lm_eval <version>` line, `import_exit`/`eval_exit`, and the last ~30
lines of `results/lm_eval_diag.log`. Likely culprits: an lm_eval API change (task names,
`--output_path`, `--log_samples`), a dep broken/renamed after a venv rebuild, or OOM. I fix
from the actual error, then hand back a corrected downstream run list.

---

## (SUPERSEDED 2026-07-26) v1.0 sprint — Steps 2/3 done; Steps 1/4 blocked by broken lm_eval

_Kept for reference only. Steps 2 (Qwen sweeps) and 3 (Qwen verify_materialized) completed.
Steps 1 and 4 failed at the `lm_eval` call. **Do NOT run the block below as-is** — the
corrected reruns (with `--skip_lm_head` + fixed group size) come after the local code fixes,
and downstream needs the STEP 0 lm_eval fix first._

```
cd /mnt/d/Abrar/SEQ/seq_v4 && git pull origin main && source .venv-seq/bin/activate

# STEP 1 (HEADLINE, ~30 min) — greedy_gptq downstream on the catastrophic checkpoint.
# Checkpoint is already exported (catastrophic 51.68/63.82); --resume skips re-export and
# runs lm-eval fresh (stale eval dirs were deleted). Do NOT delete the checkpoint.
bash scripts/run_downstream_eval.sh --points greedy_gptq --resume 2>&1 | tee results/greedy_gptq_reeval2.log
python analysis/build_downstream_table.py --root runs/final/downstream \
  --config configs/downstream_operating_points.json --out docs/DOWNSTREAM.md \
  --csv results/downstream.csv --json results/downstream.json
git add -f runs/final/downstream/Llama-3.2-3B/greedy_gptq/lm_eval/**/*.json \
           runs/final/downstream/Llama-3.2-1B/greedy_gptq/lm_eval/**/*.json \
           runs/final/downstream/*/greedy_gptq/seq_meta.json \
           results/downstream.csv results/downstream.json docs/DOWNSTREAM.md results/greedy_gptq_reeval2.log
git commit -m "greedy_gptq downstream re-eval on catastrophic checkpoint" && git push origin main
# If lm-eval ERRORS (it didn't finish last time), paste the last ~30 lines of the log.

# STEP 2 (Qwen sweeps, ~2-4 h) — HQQ+GPTQ axes. --models runs ONLY Qwen (Llama untouched);
# --llmc-repo/--llmc-venv are required args but unused by these phases.
bash scripts/run_final_seq_pipeline.sh --models Qwen/Qwen2.5-3B --phase full_matrix \
  --llmc-repo "$PWD" --llmc-venv "$PWD/.venv-seq" 2>&1 | tee results/qwen_full_matrix.log
bash scripts/run_final_seq_pipeline.sh --models Qwen/Qwen2.5-3B --phase gate \
  --llmc-repo "$PWD" --llmc-venv "$PWD/.venv-seq" 2>&1 | tee results/qwen_gate.log
git add -f runs/final/sweeps/**/Qwen2.5-3B/**/channel_pareto.json results/qwen_*.log
git commit -m "Qwen2.5-3B sweeps (HQQ+GPTQ axes)" && git push origin main

# STEP 3 (Qwen F3 check, ~10 min) — greedy@GPTQ verify_materialized for the model-dependence table.
mkdir -p results/f3check_Qwen2.5-3B
python -m seq_core.channel_sweep --model Qwen/Qwen2.5-3B --backend hqq --base_quantizer gptq_llmc \
  --gptq_model_path "$PWD/runs/final/llmc/Qwen2.5-3B/gptq/artifacts/fake_quant_model" \
  --select greedy --protect_fracs 0.02 --base_bits 4 --seed 1234 --ppl_mode canonical \
  --calibration_prompts calibration_prompts.json \
  --out_dir results/f3check_Qwen2.5-3B --verify_materialized 2>&1 | tee results/f3check_Qwen2.5-3B/run.log
git add -f results/f3check_Qwen2.5-3B/channel_pareto.json results/f3check_Qwen2.5-3B/channel_pareto.md \
           results/f3check_Qwen2.5-3B/run.log
git commit -m "Qwen2.5-3B greedy@GPTQ verify_materialized (F3 cross-family)" && git push origin main

# STEP 4 (Qwen downstream, ~1-2 h) — all 7 operating points.
bash scripts/run_downstream_eval.sh --models Qwen/Qwen2.5-3B 2>&1 | tee results/qwen_downstream.log
python analysis/build_downstream_table.py --root runs/final/downstream \
  --config configs/downstream_operating_points.json --out docs/DOWNSTREAM.md \
  --csv results/downstream.csv --json results/downstream.json
git add -f runs/final/downstream/Qwen2.5-3B/**/lm_eval/**/*.json \
           runs/final/downstream/Qwen2.5-3B/**/seq_meta.json \
           results/downstream.csv results/downstream.json docs/DOWNSTREAM.md results/qwen_downstream.log
git commit -m "Qwen2.5-3B downstream (cross-family breadth)" && git push origin main

# REPORT BACK:
#  STEP 1: greedy_gptq macro-avg accuracy + lambada acc, 1B and 3B (did accuracy COLLAPSE?).
#  STEP 3: Qwen greedy@GPTQ FP16 / runtime_ppl / materialized_ppl (catastrophic or healthy?).
#  STEP 4: Qwen macro-avg accuracy for each of the 7 points.
#  Also paste Qwen FP16/gptq4/hqq4/resmax/greedy/best_hqq PPLs from the sweeps (to fill expected_ppl).
```

---

## 1. F3 A/B diagnostic — greedy@GPTQ `--verify_materialized` (blocking, ~minutes each)

Settles whether greedy-on-GPTQ's catastrophe is real (A) or an export bug (B). The
answer is read from two log lines per model:
- `greedy … eff=… ppl=…` — runtime PPL (does the catastrophe reproduce?)
- `verify_materialized … runtime_ppl=… materialized_ppl=… Δ=…` — Δ≈0 means the export
  round-trips faithfully (⇒ not an export bug).

```bash
mkdir -p results/f3check_1b results/f3check_3b

python -m seq_core.channel_sweep --model meta-llama/Llama-3.2-1B --backend hqq \
  --base_quantizer gptq_llmc \
  --gptq_model_path "$PWD/runs/final/llmc/Llama-3.2-1B/gptq/artifacts/fake_quant_model" \
  --select greedy --protect_fracs 0.02 --base_bits 4 --seed 1234 \
  --ppl_mode canonical --calibration_prompts calibration_prompts.json \
  --out_dir results/f3check_1b --verify_materialized 2>&1 | tee results/f3check_1b/run.log

python -m seq_core.channel_sweep --model meta-llama/Llama-3.2-3B --backend hqq \
  --base_quantizer gptq_llmc \
  --gptq_model_path "$PWD/runs/final/llmc/Llama-3.2-3B/gptq/artifacts/fake_quant_model" \
  --select greedy --protect_fracs 0.02 --base_bits 4 --seed 1234 \
  --ppl_mode canonical --calibration_prompts calibration_prompts.json \
  --out_dir results/f3check_3b --verify_materialized 2>&1 | tee results/f3check_3b/run.log

git add -f results/f3check_1b/channel_pareto.json results/f3check_1b/channel_pareto.md \
           results/f3check_1b/run.log \
           results/f3check_3b/channel_pareto.json results/f3check_3b/channel_pareto.md \
           results/f3check_3b/run.log
git commit -m "F3 A/B diagnostic: greedy@GPTQ verify_materialized (1B+3B) results"
git push origin main
```

**Result — DONE (`0de596e`), regenerated base, verify_materialized Δ≈0 everywhere:**

| model | FP16 | greedy@GPTQ runtime | materialized | Δ export | verdict |
|---|---|---|---|---|---|
| Llama-3.2-1B | 9.76 | 63.95 | 63.82 | −0.12 | catastrophe reproduces |
| Llama-3.2-3B | 7.82 | 51.68 | 51.76 | +0.08 | catastrophe reproduces |
| Llama-2-7B | 5.47 | 5.57 | 5.57 | ~0 | **healthy** (Δ+0.10) |

⇒ **(B) export bug falsified** (materialize is faithful); **(A) F3 is real**, survives base
regeneration; and **F3 is model-family-dependent** (hits Llama-3.2, not Llama-2-7B). No
re-run needed. Failures were non-blocking (3.1-8B / Qwen2.5-1.5B missing GPTQ artifact;
Qwen3-4B OOM in Hessian collection).

---

# DAY 1 — sprint jobs (priority order)

## 1b. CLOSE the downstream contradiction (blocking, ~15 min) — do this first

3B greedy@GPTQ is catastrophic (51.68) AND exports faithfully, yet the §7 **downstream**
checkpoint scored healthy (lambada 4.17). `verify_materialized` only tests *in-memory*
materialize, not save→disk→reload — so reload the **on-disk** downstream checkpoint that
lm-eval actually scored and measure its WikiText-2 PPL. `--expected` is set to the sweep
runtime; read `reload_ppl` (PASS ⇒ checkpoint is catastrophic ⇒ the healthy downstream was
a stale/wrong checkpoint; big gap to ~8 ⇒ the save→reload path drops protection).

```bash
python scripts/validate_saved_seq_reload.py \
  runs/final/downstream/checkpoints/Llama-3.2-3B/greedy_gptq \
  --expected 51.68 --tolerance 0.5 2>&1 | tee results/f3check_3b/reload_existing.log

python scripts/validate_saved_seq_reload.py \
  runs/final/downstream/checkpoints/Llama-3.2-1B/greedy_gptq \
  --expected 63.95 --tolerance 0.5 2>&1 | tee results/f3check_1b/reload_existing.log
```

If the checkpoint dir is missing (cleaned up), regenerate it fresh (NO `--resume`) with the
exact downstream export command, then reload-validate `results/f3check_3b/_freshexport`:

```bash
python -m seq_core.channel_sweep --model meta-llama/Llama-3.2-3B --backend hqq \
  --base_bits 4 --protect_fracs 0.02 --seed 1234 --ppl_mode canonical \
  --calibration_prompts calibration_prompts.json --base_quantizer gptq_llmc \
  --gptq_model_path "$PWD/runs/final/llmc/Llama-3.2-3B/gptq/artifacts/fake_quant_model" \
  --out_dir results/f3check_3b/_freshexport/_sweep \
  --save_model_path results/f3check_3b/_freshexport --save_signal greedy --save_k_frac 0.02 \
  --select greedy 2>&1 | tee results/f3check_3b/freshexport.log
python scripts/validate_saved_seq_reload.py results/f3check_3b/_freshexport \
  --expected 51.68 --tolerance 0.5 2>&1 | tee -a results/f3check_3b/freshexport.log
```

Commit: `git add -f results/f3check_*/reload_existing.log results/f3check_*/freshexport.log \
runs/final/downstream/checkpoints/*/greedy_gptq/reload_validation.json` (if written), then
commit + push. **Report the `reload_ppl` for 1B and 3B.**

## 2. Unify GPTQ base provenance (paper §12.2, ~2–4h)

The sweep phases read the GPTQ base from `runs/final/llmc/<model>/gptq/artifacts/fake_quant_model`,
which is now the **regenerated** base — so re-running them (without `--resume`, so they
overwrite) makes §5/§6/§7/COMPARISON all cite one base. Covers `residual_max, residual_rms,
act_max, act_scale, random×3seeds` (full_matrix) and `greedy, greedy_indep, random`
(gate), fractions {0.02,0.05,0.1,0.2}, for 1B+3B, both bases:

```bash
bash scripts/run_final_seq_pipeline.sh --phase full_matrix 2>&1 | tee results/resweep_full_matrix.log
bash scripts/run_final_seq_pipeline.sh --phase gate        2>&1 | tee results/resweep_gate.log

git add -f runs/final/sweeps/**/channel_pareto.json results/resweep_*.log
git commit -m "Unify GPTQ base provenance: re-sweep GPTQ+HQQ axes on regenerated base (1B+3B)"
git push origin main
```

(Re-running the HQQ cells too is intentional — it removes all provenance ambiguity for a
submission. If GPU time is tight, `--phase gate` alone refreshes the F3/greedy numbers.)

## 3. random@HQQ matched-bit downstream point (paper §12.3, ~20 min)

Point already added to `configs/downstream_operating_points.json` (pull first). Exports a
random-channel control at the same 7.70 bits as `best_hqq`, then lm-evals it:

```bash
bash scripts/run_downstream_eval.sh --points random_hqq --resume 2>&1 | tee results/random_hqq_downstream.log
# then rebuild the table (CPU, can be done locally):
python analysis/build_downstream_table.py --root runs/final/downstream \
  --config configs/downstream_operating_points.json --out docs/DOWNSTREAM.md \
  --csv results/downstream.csv --json results/downstream.json
git add -f runs/final/downstream/**/*.json results/downstream.* docs/DOWNSTREAM.md
git commit -m "Add random@HQQ matched-bit downstream control (F1 sharpening)"
git push origin main
```

## 4. (If window remains) cross-family — Qwen2.5-3B

Sweep + downstream for Qwen2.5-3B (GPTQ baseline already in
`runs/final/llmc/Qwen2.5-3B`). Out of scope for v1.0 but strengthens F1/F3/F4 generality.
