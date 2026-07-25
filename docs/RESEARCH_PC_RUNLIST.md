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
