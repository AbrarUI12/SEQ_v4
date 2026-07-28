# Research-PC run list (weekly GPU window)

Copy-paste commands for the research PC (WSL, RTX 5090 + 4090, `.venv-seq`, LightCompress
+ regenerated GPTQ `fake_quant_model` bases). This file is tracked, so it travels with
`git pull origin main` — it is the source of truth for GPU-side runs, replacing ad-hoc
copy-paste blocks.

Before anything, set `SEQ_REPO` to wherever this checkout lives on the box and export it — the
repo has moved between machines (`/mnt/d/Abrar/SEQ/seq_v4` → `/mnt/e/seq v4/SEQ-clean-v4`) and the
current path **contains a space**, so it must stay double-quoted everywhere:

```bash
export SEQ_REPO="/mnt/e/seq v4/SEQ-clean-v4"     # adjust per machine
cd "$SEQ_REPO" && git pull origin main && source .venv-seq/bin/activate
```

Do not hardcode an absolute repo path into a command you paste — that is exactly what broke the
first Job 3 run.

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

## ✅ ALL PLANNED EXPERIMENTS ARE DONE — remaining work only

E1–E5 have returned and the paper (`docs/FINDINGS_PAPER.md`, v3.0) is written on them. Headline:
a reload-verified 6.3× WikiText-2 perplexity collapse (8.172 → 51.76 on Llama-3.2-3B) with **no**
downstream cost (macro 67.61% vs 67.10% for its base; +0.51 pts [+0.13, +0.86]).

**Three items remain, in priority order.** None blocks the current draft; each strengthens it.

1. **Repair the Llama-3.2-1B export (highest value).** Its `greedy_gptq` checkpoint reloads at
   **203.72** against an expected 63.95 (`FAIL`), so the 1B downstream row is excluded from the
   paper and the decoupling rests on one model. Re-export fresh and reload-validate; if it passes,
   run downstream for that point to replicate the headline on a second model.
   ```bash
   rm -rf runs/final/downstream/checkpoints/Llama-3.2-1B/greedy_gptq \
          runs/final/downstream/Llama-3.2-1B/greedy_gptq
   bash scripts/run_downstream_eval.sh --models meta-llama/Llama-3.2-1B --points greedy_gptq 2>&1 \
     | tee results/reexport_1B_greedy.log
   python scripts/validate_saved_seq_reload.py \
     runs/final/downstream/checkpoints/Llama-3.2-1B/greedy_gptq \
     --expected 63.95 --tolerance 0.5 2>&1 | tee results/reload_1B_greedy_retry.log
   ```
   **Report `reload_ppl`.** If it lands near 63.95 the row can be included; if it disagrees again,
   that itself is a finding about export determinism and should be reported.

2. **Per-token loss decomposition** (explains *how* the decoupling is possible) — **script ready**:
   `scripts/per_token_loss_decomposition.py`. It scores the base and the damaged checkpoint on the
   *same* token stream (verified to reproduce the paper's exact 141 windows / 288,627 supervised
   targets) and reports what share of the perplexity increase the worst-k% of tokens carry, plus
   the perplexity with those tokens excluded.
   ```bash
   python scripts/per_token_loss_decomposition.py \
     --base  runs/final/llmc/Llama-3.2-3B/gptq/artifacts/fake_quant_model \
     --other runs/final/downstream/checkpoints/Llama-3.2-3B/greedy_gptq \
     --tokenizer meta-llama/Llama-3.2-3B \
     --out results/pertoken_Llama-3.2-3B.json 2>&1 | tee results/pertoken_Llama-3.2-3B.log
   ```
   **Report** `frac_tokens_worse`, `median_delta_nll`, the `concentration` block and the
   `trimmed_perplexity` block. If excluding the worst ~1% of tokens collapses the 8.17→51.76 gap,
   the damage is concentrated in a tail and the decoupling is explained.

3. **Ordering intervention (E2) — never produced a result.** Its log predates the instrumentation
   fix and stalled at `block 1/28`. The sequential pass is now instrumented and in-pass selection
   runs in float32, so **size it before committing hours**: run the 2-block diagnostic, multiply
   the per-block time by 28, and only launch the full job if the estimate is acceptable.
   ```bash
   # (a) size it first — minutes, prints which sub-step dominates
   python scripts/measure_objective_alignment.py --model meta-llama/Llama-3.2-3B \
     --base_bits 4 --group_size 128 --seed 1234 --n_calib 128 --max_blocks 2 \
     --out results/align_diag_3B.json 2>&1 | tee results/align_diag_3B.log
   # (b) only if (a) projects an acceptable runtime:
   python scripts/run_protect_then_gptq.py --model meta-llama/Llama-3.2-3B --base_bits 4 \
     --group_size 128 --protect_frac 0.02 --seed 1234 --n_calib 128 \
     --out results/owq_seq_Llama-3.2-3B.json 2>&1 | tee results/owq_seq_Llama-3.2-3B.log
   ```
   **Sanity-check `ppl_base_gptq` first** — it must be near FP16 (~8 on 3B). If it is in the
   hundreds or thousands the internal base is degenerate and all three arms are uninformative
   (this is exactly how the first attempt failed). It stays future work in the paper unless it
   returns a clean result.

---

## 🛑 REFERENCE — stalled jobs, and the rules that prevent it

**One GPU job at a time.** E1 was holding 23.7 GiB while E4 ran, so lm_eval was starved onto CPU
(that is the "only 620 MiB" observation) — correct results, but glacial. Before launching
anything: `nvidia-smi` must show an idle GPU.

**E1 and E2 both stalled at `seq-gptq: block 1/28`. That was our bug, not a model property.**
Our internal sequential GPTQ had never been run above tiny test models — all real GPTQ bases came
from LightCompress. Two confirmed cost bugs, both now fixed:
- E1 shipped every layer's full ΔW `[out,in]` fp32 to CPU: **12.7 GB** of pointless traffic on 3B.
  It now keeps only `[in]`-sized column summaries.
- E2's in-pass greedy ran `A@Hf` in **float64** (412 GFLOP per `down_proj`; consumer GPUs run fp64
  at ~1/64 rate). In-pass selection is now float32 (k is small, so the drift protection float64
  was added for is unnecessary).

**Everything is now instrumented.** Each run logs phase banners plus per-block, per-layer and
per-sample progress with elapsed time, GPU/CPU memory and a running ETA. Judge health from the
log, not from GPU%:
```
[t+ 3m12s] START block 4: hessians over 128 samples | gpu 5.31/7.02GB alloc, 8.00GB reserved | rss 12.4GB
    hessian samples: 64/128 (50.0%)  41.2s elapsed, 1.6/s, eta  40.1s | ...
    layer mlp.down_proj  (3072, 8192) done in 6.4s | ...
seq-gptq: block 4/28 COMPLETE in 96.2s | ...
```
If the newest line is minutes old *and* names a step that should take seconds, it is stalled. If
progress lines keep arriving, it is healthy even at low GPU utilization.

### Bounded diagnostic — run this before any full E1/E2 job
`--max_blocks 2` processes only the first two decoder blocks (minutes, not hours) and prints
exactly which sub-step dominates:
```bash
cd "$SEQ_REPO" && git pull origin main && source .venv-seq/bin/activate
nvidia-smi   # confirm idle first
python scripts/measure_objective_alignment.py --model meta-llama/Llama-3.2-3B \
  --base_bits 4 --group_size 128 --seed 1234 --n_calib 128 --max_blocks 2 \
  --out results/align_diag_3B.json 2>&1 | tee results/align_diag_3B.log
```
**Report:** the per-block `COMPLETE in Xs` line and the slowest `layer …` line. Multiply the block
time by 28 for the full-run estimate. Only launch the full runs if that estimate is acceptable.
For the real E1 measurement a bounded subset is scientifically fine (alignment is a per-layer
statistic; a median over ~20 layers is stable) — use `--max_blocks 8` and it is recorded in the
output JSON as `max_blocks`, so the subset is disclosed rather than hidden.

### Triage for the currently-running jobs
- **E1 (stalled)** — kill it; the fix is landed. Preserve and push the log first.
- **E2 (stalled)** — same; re-run after pulling.
- **E4 (slow, 620 MiB)** — its numbers are valid, only slow. With E1 killed it should speed up.
  Verify progress by log growth, not GPU%:
  `stat -c %s results/downstream_Llama-3.2-3B.log` twice, 60 s apart.

---

## 🚨 E5 — RUN THIS FIRST: is the catastrophe a selector-Hessian artifact?

A reviewer found a logical hole we could not answer, and chasing it uncovered a **confound that
may generate the entire headline result**. The greedy selector's score is the *exact* decrease in
`tr(ΔW H ΔWᵀ)` from restoring a column, and every pick had **positive** gain — so protection
provably *lowers* the layer's reconstruction error, yet perplexity explodes 8.17 → 51.68.

The likely reason: the two Hessians are estimated from completely different data.

| | calibration for H | tokens |
|---|---|---|
| GPTQ base (LightCompress) | WikiText-2, 128 × 2048 | **~262,000** |
| greedy selector (ours) | 51 short prompts in `calibration_prompts.json` | **~500** |

For layers with 3072–8192 input channels, H from ~500 out-of-distribution tokens is
**rank-deficient by ~10×**. The selector may simply be ranking noise. New flag
`--selector_calib_samples N` estimates the selector's H from WikiText-2 at GPTQ's scale instead;
the run records `selector_calib_source` and `selector_hessian_tokens` in its JSON.

```bash
cd "$SEQ_REPO" && git pull origin main && source .venv-seq/bin/activate
for M in meta-llama/Llama-3.2-3B meta-llama/Llama-3.2-1B; do
  N="${M##*/}"
  python -m seq_core.channel_sweep --model "$M" --backend hqq --base_quantizer gptq_llmc \
    --gptq_model_path "$PWD/runs/final/llmc/$N/gptq/artifacts/fake_quant_model" \
    --select greedy --protect_fracs 0.02,0.05,0.1,0.2 --base_bits 4 --seed 1234 --skip_lm_head \
    --ppl_mode canonical --calibration_prompts calibration_prompts.json \
    --selector_calib_samples 128 \
    --out_dir "results/e5_matchedcalib_$N" 2>&1 | tee "results/e5_${N}.log"
done
git add -f results/e5_matchedcalib_*/channel_pareto.json results/e5_*.log
git commit -m "E5: greedy@GPTQ with selector Hessian matched to GPTQ calibration" && git push origin main
```

**Report:** the `greedy` perplexity at each budget for both models, plus the logged
`selector Hessian: wikitext2:128x2048, N rows` line. Compare against the current numbers
(3B 51.68 / 1B 63.95, both measured with the ~500-token Hessian).

**Either outcome is a good result, so run it before anything else:**
- **Catastrophe shrinks/vanishes** → the finding becomes *"Hessian-based protection on a
  compensated base is only as good as the selector's Hessian; an under-estimated one is
  catastrophic precisely because the compensated residual is large, while Hessian-free selectors
  are robust."* Actionable and fully explained.
- **Catastrophe persists** → the obvious confound is excluded and the compensation-structure
  account becomes far stronger.

---

## ⭐⭐ HANDOFF EXPERIMENTS E1–E4 (2026-07-26) — for paper v2.0 "Objective Collision"

The paper's headline is now the **base-conditioned inversion**: the same Hessian-weighted
selector is the *best* signal on data-free HQQ (3B 8.100 vs random 8.319) and the *worst* on
compensated GPTQ (8.172 → 51.68). The explanation is **objective collision** — the selector
maximizes reduction of ‖ΔW X‖², exactly what GPTQ minimizes, so on a compensated base it ranks
compensation-bearing columns instead of salient ones. These four experiments close the paper's
open items. **Hand them out one at a time**; each is self-contained.

Common preamble (all experiments):
```bash
cd "$SEQ_REPO" && git pull origin main && source .venv-seq/bin/activate
```
Rules: never `/tmp` for outputs; `tee` logs under `results/`; `git add -f` (results/ and runs/
are gitignored); **never** commit weights or anything under `runs/final/downstream/checkpoints/`;
a job is finished when the shell prompt returns — do not poll PIDs or "monitor" processes.

### E4 — downstream evaluation (highest value; fills paper §9)
Long (~1–3 h per model) and **must run uninterrupted**; earlier attempts were cut off mid-run.
A `set -euo pipefail` abort on a missing `lm_eval/` dir was fixed, so pull first.
```bash
for M in meta-llama/Llama-3.2-3B meta-llama/Llama-3.2-1B; do
  N="${M##*/}"
  bash scripts/run_downstream_eval.sh --models "$M" 2>&1 | tee "results/downstream_${N}.log"
done
python analysis/build_downstream_table.py --root runs/final/downstream \
  --config configs/downstream_operating_points.json --out docs/DOWNSTREAM.md \
  --csv results/downstream.csv --json results/downstream.json
git add -f runs/final/downstream/*/*/lm_eval/**/*.json runs/final/downstream/*/*/seq_meta.json \
           results/downstream.csv results/downstream.json docs/DOWNSTREAM.md results/downstream_*.log
git commit -m "E4: full-scale downstream (corrected scope)" && git push origin main
```
**Report:** per-point macro accuracy for each model, especially `greedy_gptq` (does accuracy
collapse, matching its perplexity of ~52/64?).

### E2 — ordering intervention (causal; fills §6.4b)
Same protected set S, only *when* it is made exact differs. ~1–1.5 h per model.
```bash
for M in meta-llama/Llama-3.2-3B meta-llama/Llama-3.2-1B; do
  N="${M##*/}"
  python scripts/run_protect_then_gptq.py --model "$M" --base_bits 4 --group_size 128 \
    --protect_frac 0.02 --seed 1234 --n_calib 128 \
    --out "results/owq_seq_${N}.json" 2>&1 | tee "results/owq_seq_${N}.log"
done
git add -f results/owq_seq_*.json results/owq_seq_*.log
git commit -m "E2: ordering intervention (post-hoc vs select-before-GPTQ)" && git push origin main
```
**Report:** `ppl_fp16`, `ppl_base_gptq`, `ppl_A_posthoc_restore`, `ppl_B_protect_before_gptq`,
`verdict`. **Sanity-check `ppl_base_gptq` first** — it must be close to the FP16 value (~8 on 3B).
If it is in the hundreds or thousands the base is degenerate and the arms are uninformative.

### E1 — objective alignment (the mechanism measurement; fills §6.4a and explains §7)
Cheap (~20–40 min per model, no perplexity evaluations). Runs on all four models.
```bash
for M in meta-llama/Llama-3.2-3B meta-llama/Llama-3.2-1B Qwen/Qwen2.5-3B meta-llama/Llama-2-7b-hf; do
  N="${M##*/}"
  python scripts/measure_objective_alignment.py --model "$M" --base_bits 4 --group_size 128 \
    --seed 1234 --n_calib 128 --out "results/align_${N}.json" 2>&1 | tee "results/align_${N}.log"
done
git add -f results/align_*.json results/align_*.log
git commit -m "E1: selector-vs-compensation rank alignment" && git push origin main
```
**Report:** the `summary` block per model (median ρ per selector) and `ranking_by_abs_median_rho`.
**Pre-registered prediction:** `greedy_gain` > `hessian_diag` > `residual_rms`/`residual_max` in
|ρ|, and alignment should be *higher on the susceptible models* (Llama-3.2-1B/3B) than on the
safe ones (Qwen2.5-3B, Llama-2-7B) — that would turn §7's model dependence into a diagnostic.

### E3 — intermediate-coupling selector (fills the alignment axis; §6.4c)
Adds `hessian_diag` (Hessian diagonal, no cross terms = "half" the objective) to the GPTQ panel.
Cheap; prediction is harm strictly between `residual_max` (safe) and `greedy_indep` (harmful).
```bash
for M in meta-llama/Llama-3.2-3B meta-llama/Llama-3.2-1B; do
  N="${M##*/}"
  python -m seq_core.channel_sweep --model "$M" --backend hqq --base_quantizer gptq_llmc \
    --gptq_model_path "$PWD/runs/final/llmc/$N/gptq/artifacts/fake_quant_model" \
    --signals hessian_diag --protect_fracs 0.02,0.05,0.1,0.2 --base_bits 4 --seed 1234 \
    --skip_lm_head --ppl_mode canonical --calibration_prompts calibration_prompts.json \
    --out_dir "runs/final/sweeps/gptq_llmc/$N/hessian_diag/seed-1234" 2>&1 | tee "results/hdiag_${N}.log"
done
git add -f runs/final/sweeps/gptq_llmc/*/hessian_diag/seed-1234/channel_pareto.json results/hdiag_*.log
git commit -m "E3: hessian_diag intermediate-coupling selector on GPTQ" && git push origin main
```
**Report:** the `hessian_diag` perplexity at each budget for both models.

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
cd "$SEQ_REPO" && git pull origin main && source .venv-seq/bin/activate
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
cd "$SEQ_REPO" && git pull origin main && source .venv-seq/bin/activate

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
