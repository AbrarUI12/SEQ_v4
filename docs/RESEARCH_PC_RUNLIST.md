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

**Result so far (1B, regenerated base):** greedy@GPTQ frac=0.02 → runtime_ppl=63.9462,
materialized_ppl=63.8248, Δ=−0.1214; FP16 baseline 9.7572 (Δ +54.19). ⇒ catastrophe
**reproduces** and the export **round-trips faithfully** ⇒ **(A) F3 is real, not an
export bug.** 3B still to run. (Original base reported 104.16 for 1B; regenerated ≈64 —
same phenomenon, magnitude differs with base provenance, cf. item 2.)

## 2. Unify GPTQ base provenance (paper §12.2)

Re-run the GPTQ-axis sweeps (`greedy`, `greedy_indep`, `residual_max`, `random` × fracs)
on the regenerated base via the pipeline sweep phase, so every GPTQ number shares one
base. Commit updated `runs/final/sweeps/gptq_llmc/**` JSONs.

```bash
bash scripts/run_final_seq_pipeline.sh   # sweep phase; see script flags
```

## 3. random@HQQ matched-bit downstream point (paper §12.3)

The point is prepped locally in `configs/downstream_operating_points.json` beforehand.

```bash
bash scripts/run_downstream_eval.sh --points random_hqq --resume
```

## 4. (If window remains) cross-family — Qwen2.5-3B

Sweep + downstream for Qwen2.5-3B (GPTQ baseline already in
`runs/final/llmc/Qwen2.5-3B`).
