# When Outlier Protection Backfires

Code and results for a controlled audit of **post-hoc mixed-precision channel protection** in
post-training LLM weight quantization.

**Headline finding.** On an error-compensated (GPTQ) base, selecting input channels by a
residual-driven set objective and restoring them to FP16 *after* quantization can destroy the
model — WikiText-2 perplexity 8.17 → 38–55 on Llama-3.2-3B and 10.56 → 64–78 on Llama-3.2-1B —
while activation-magnitude and random selection of the same number of channels at identical
storage are harmless. The failure survives base regeneration and export/reload, and is
model-dependent (absent on Llama-2-7B and Qwen2.5-3B). On a data-free (HQQ) base, by contrast,
protection gives a small but consistent gain over a matched-bit random control.

Paper draft: [`docs/FINDINGS_PAPER.md`](docs/FINDINGS_PAPER.md).

## Install

```bash
python -m venv .venv-seq && source .venv-seq/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Gated models (Llama) require `huggingface-cli login`. A CUDA GPU is needed for sweeps and
evaluations; rebuilding tables/figures from the committed JSON is CPU-only.

## Reproduce

**Rebuild every table and figure from committed results (CPU, seconds):**

```bash
python analysis/plot_final_results.py --input results/final_comparison.csv --output-dir figures/final
python analysis/build_downstream_table.py --root runs/final/downstream \
  --config configs/downstream_operating_points.json --out docs/DOWNSTREAM.md \
  --csv results/downstream.csv --json results/downstream.json
```

**One protection sweep** (selector × budget on one base, writes `channel_pareto.json`):

```bash
python -m seq_core.channel_sweep --model meta-llama/Llama-3.2-3B --backend hqq \
  --base_quantizer gptq_llmc --gptq_model_path <path-to-gptq-fake-quant-dir> \
  --select greedy --protect_fracs 0.02 --base_bits 4 --seed 1234 --skip_lm_head \
  --ppl_mode canonical --calibration_prompts calibration_prompts.json \
  --out_dir runs/example --verify_materialized
```

`--skip_lm_head` is required for a valid comparison: it keeps `lm_head` in FP16 so the
quantized scope matches the GPTQ baseline. `--verify_materialized` re-measures perplexity after
materializing the protection to dense weights; the two must agree.

**The full matrix, the causal experiment, and downstream evaluation** are driven by
`scripts/run_final_seq_pipeline.sh`, `scripts/run_protect_then_gptq.py` and
`scripts/run_downstream_eval.sh`. Exact invocations used to produce the committed results are
in [`docs/RESEARCH_PC_RUNLIST.md`](docs/RESEARCH_PC_RUNLIST.md).

**Tests / smoke gate:**

```bash
bash scripts/smoke_local.sh    # static checks + unit tests (+ GPU smokes if torch is present)
```

## Layout

| path | contents |
|---|---|
| `seq_core/` | protection, selectors (`greedy_select.py`), GPTQ (`gptq.py`), storage accounting, sweep driver |
| `scripts/` | pipeline runners, causal experiment, downstream evaluation, validation utilities |
| `analysis/` | table/figure builders (pure stdlib + matplotlib) |
| `runs/final/sweeps/` | committed sweep results (`channel_pareto.json` per selector/seed) |
| `results/`, `figures/`, `docs/` | derived tables, figures, paper and run documentation |

## Conventions for results

Every perplexity in the paper comes from a committed `channel_pareto.json` carrying its own
provenance (`skip_lm_head`, `base_group_size`, `seed`, `greedy_early_stop`). Storage is
**theoretical weight-only** bits per quantized-linear parameter: low-bit values, group
scales/zeros at the base's true group size, FP16 correction columns and channel indices;
embeddings, `lm_head` and norms are FP16 in every method and excluded. Model weights and
fake-quantized checkpoints are deliberately **not** committed.

## License

MIT — see [`LICENSE`](LICENSE).
