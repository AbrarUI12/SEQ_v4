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

## ⭐ CORRECTED GPU WORKLIST (2026-07-26) — code fixes L1-L5 are LANDED

`lm_eval` is fine (v0.4.12, verified); the earlier "failures" were long runs cut off when the
session ended. Local integrity fixes are pushed: `--skip-lm-head`, g128 storage, whole
`--greedy_early_stop`, `scripts/run_protect_then_gptq.py`, per-point seeds. **`git pull` then
run the items below**, in priority order. Models: L3=meta-llama/Llama-3.2-3B (headline),
L1=meta-llama/Llama-3.2-1B, Q3=Qwen/Qwen2.5-3B, L27=meta-llama/Llama-2-7b-hf (`<name>` = the
basename). Rules: never `/tmp`; `tee` logs; `git add -f`; never commit weights/`checkpoints/**`.
Leave downstream (G4) sessions **uninterrupted** — each model is 6 full tasks (~1-3 h).

```bash
cd /mnt/d/Abrar/SEQ/seq_v4 && git pull origin main && source .venv-seq/bin/activate
M=meta-llama/Llama-3.2-3B; N=Llama-3.2-3B      # repeat the block per model (L3 first, then L1, Q3, L27)

# G1 — corrected sweeps (skip_lm_head + g128 storage):
bash scripts/run_final_seq_pipeline.sh --models "$M" --phase full_matrix --skip-lm-head \
  --llmc-repo "$PWD" --llmc-venv "$PWD/.venv-seq" 2>&1 | tee results/sweep_${N}_fm.log
bash scripts/run_final_seq_pipeline.sh --models "$M" --phase gate --skip-lm-head \
  --llmc-repo "$PWD" --llmc-venv "$PWD/.venv-seq" 2>&1 | tee results/sweep_${N}_gate.log

# G2 — F3 verify_materialized (exact-k) + early-stop causal control:
python -m seq_core.channel_sweep --model "$M" --backend hqq --base_quantizer gptq_llmc \
  --gptq_model_path "$PWD/runs/final/llmc/$N/gptq/artifacts/fake_quant_model" \
  --select greedy --protect_fracs 0.02 --base_bits 4 --seed 1234 --skip_lm_head \
  --ppl_mode canonical --calibration_prompts calibration_prompts.json \
  --out_dir results/f3_$N --verify_materialized 2>&1 | tee results/f3_$N.log
python -m seq_core.channel_sweep --model "$M" --backend hqq --base_quantizer gptq_llmc \
  --gptq_model_path "$PWD/runs/final/llmc/$N/gptq/artifacts/fake_quant_model" \
  --select greedy --protect_fracs 0.02 --base_bits 4 --seed 1234 --skip_lm_head --greedy_early_stop \
  --ppl_mode canonical --calibration_prompts calibration_prompts.json \
  --out_dir results/f3_earlystop_$N --verify_materialized 2>&1 | tee results/f3_es_$N.log

# G3' — DECISIVE causal experiment, REBUILT on sequential GPTQ (supersedes the earlier G3;
# that one used the one-shot path and produced a degenerate base >3000 PPL — discard those
# results/owq_*.json). Runs 2 sequential GPTQ passes + 4 PPL evals (~1-1.5 h on 3B).
# Read `verdict`: A_collapses_B_healthy => post-hoc restoration causally breaks compensation.
python scripts/run_protect_then_gptq.py --model "$M" --base_bits 4 --group_size 128 \
  --protect_frac 0.02 --seed 1234 --n_calib 128 --out results/owq_seq_$N.json 2>&1 | tee results/owq_seq_$N.log

git add -f runs/final/sweeps/**/channel_pareto.json results/sweep_*.log \
           results/f3_*/channel_pareto.json results/f3_*.log results/owq_seq_*.json results/owq_seq_*.log
git commit -m "corrected sweeps + F3 causal ($N)" && git push origin main

# G4 — full-scale downstream (uniform, NO --limit), per model. Long; keep uninterrupted:
bash scripts/run_downstream_eval.sh --models "$M" 2>&1 | tee results/downstream_$N.log
# (optional) matched-bit random control over 3 seeds (adds ~2x lm-eval; do on L3/L1 if time):
bash scripts/run_downstream_eval.sh --models "$M" --points random_hqq_s2,random_hqq_s3 2>&1 | tee -a results/downstream_$N.log
python analysis/build_downstream_table.py --root runs/final/downstream \
  --config configs/downstream_operating_points.json --out docs/DOWNSTREAM.md \
  --csv results/downstream.csv --json results/downstream.json
git add -f runs/final/downstream/$N/**/lm_eval/**/*.json runs/final/downstream/$N/**/seq_meta.json \
           results/downstream.* docs/DOWNSTREAM.md results/downstream_$N.log
git commit -m "downstream $N (full scale, skip_lm_head)" && git push origin main
```

**Report back per model:** G2 FP16/runtime/materialized PPL (+ early-stop PPL); G3′
`ppl_fp16/ppl_base_gptq/ppl_A_posthoc_restore/ppl_B_protect_before_gptq` + `verdict`; G4 the
per-point macro accuracies.

**Current priority (paper v1.0 is written; these two fill its only gaps):**
1. **G4 downstream for L3 + L1** — fills §8, the last empty section. A `set -e` bug that
   aborted this silently is fixed (commit 66d794c), so pull first. Keep the session running.
2. **G3′ for L3** — fills §9; converts the mechanism from hypothesis to a causal claim.
Everything else (G1 sweeps, G2) is already done and in the paper.

---

### (archived) STEP 0 lm_eval diagnosis — RESOLVED
lm_eval works (v0.4.12, exit 0). The earlier no-output "failures" were interrupted long runs,
not a broken lm_eval. Kept only as a record.

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
