# Codex prompt — Job 3 (audit follow-ups: 1B no-pad, calibration sensitivity, log restoration)

Paste everything between the rulers into Codex on the research PC.

---

You are working in the SEQ research repo on a WSL box with an RTX 5090/4090.

**Setup (do this first, exactly):**

```bash
cd /mnt/d/Abrar/SEQ/seq_v4
git pull origin main
source .venv-seq/bin/activate
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
```

Three jobs. A and B are **read-only with respect to published results** — they write only into
`results/`, and they must not touch `runs/final/sweeps/**`, `runs/final/downstream/**`, or any
`fake_quant_model/` directory. `results/` is `.gitignore`d, so result JSONs must be
**force-added** (`git add -f`). **Never `git add` model weights.**

Never point `--out_dir` at `/tmp` — WSL's `/tmp` is tmpfs and is wiped on idle/shutdown.

Job C re-runs three downstream evaluations in place; it must reproduce their published
accuracies exactly, and the gate below checks that.

Run the jobs **sequentially** (each wants the whole GPU). Total ≈ 4–4.5 h. If the window is short,
**Job C is the highest priority** — it closes a reproducibility gap on the paper's headline
interval — followed by A, then B.

---

## Preflight (2 min, do not skip)

```bash
# 1. The new flags must exist. If any line prints nothing, STOP — you are on a stale commit.
python -m seq_core.channel_sweep --help | grep -E "no_pad_calibration|selector_calib_seed|selector_calib_tokens|selector_calib_samples"

# 2. The 1B GPTQ base must be present (weights are not in git).
ls -d runs/final/llmc/Llama-3.2-1B/gptq/artifacts/fake_quant_model
```

If the 1B `fake_quant_model` is **missing**, regenerate it before Job A with the recorded
LightCompress command (its config is tracked):

```bash
/mnt/d/LightCompress/.venv-llmc/bin/torchrun --standalone --nproc_per_node=1 \
  --module scripts.llmc_ppl_entrypoint \
  --config /mnt/d/Abrar/SEQ/seq_v4/runs/final/llmc/Llama-3.2-1B/gptq/config.yml \
  --task_id gptq_w4_g128_llama32_1b
```

Then confirm the base reproduces its recorded perplexity: `runs/final/llmc/Llama-3.2-1B/gptq/summary.json`
records `ppl = 10.4306`. If the regenerated base is more than ~0.05 away from that, **stop and
report it** — do not run Job A on a base that does not match the published one.

---

## Job A — no-pad calibration robustness on Llama-3.2-1B (≈45 min)

**Why.** The audit showed the scalar selectors' statistics were collected on prompts padded to
2048, so they were pad-dominated. We already re-measured on real tokens for Llama-3.2-3B
(commit `9f6cc8a`) and the ordering survived: three of four scalars were unmoved and only
`act_scale` shifted (8.586 → 8.142). This repeats that on 1B so §5.1 is not a single-model claim.

These are the exact 3B commands with the model swapped. Do not change any other flag.

**A1 — GPTQ-4 base, four scalars, four budgets:**

```bash
mkdir -p results
python -m seq_core.channel_sweep \
  --model meta-llama/Llama-3.2-1B --backend hqq --base_bits 4 \
  --base_quantizer gptq_llmc \
  --gptq_model_path "$PWD/runs/final/llmc/Llama-3.2-1B/gptq/artifacts/fake_quant_model" \
  --signals act_max,act_scale,residual_max,residual_rms \
  --protect_fracs 0.02,0.05,0.1,0.2 \
  --seed 1234 --ppl_mode canonical --skip_lm_head \
  --calibration_prompts calibration_prompts.json \
  --no_pad_calibration \
  --out_dir results/nopad_gptq_Llama-3.2-1B \
  2>&1 | tee results/nopad_gptq_Llama-3.2-1B.log
```

**A2 — HQQ-4 base, same scalars plus the random control, two budgets:**

```bash
python -m seq_core.channel_sweep \
  --model meta-llama/Llama-3.2-1B --backend hqq --base_bits 4 \
  --base_quantizer hqq \
  --signals act_max,act_scale,residual_max,residual_rms,random \
  --protect_fracs 0.02,0.2 \
  --seed 1234 --ppl_mode canonical --skip_lm_head \
  --calibration_prompts calibration_prompts.json \
  --no_pad_calibration \
  --out_dir results/nopad_hqq_Llama-3.2-1B \
  2>&1 | tee results/nopad_hqq_Llama-3.2-1B.log
```

**Gate.** Both output JSONs must record `"pad_calibration": false`. Check and report:

```bash
python - <<'PY'
import json
for f in ["results/nopad_gptq_Llama-3.2-1B/channel_pareto.json",
          "results/nopad_hqq_Llama-3.2-1B/channel_pareto.json"]:
    d = json.load(open(f))
    assert d["pad_calibration"] is False, f"{f}: pad_calibration is not false"
    print(f"\n== {f}  fp16={d['baseline_fp16_ppl']:.4f}  pad_calibration={d['pad_calibration']}")
    for r in sorted(d["results"], key=lambda r: (r["signal"], r["k_frac"])):
        print(f"   {r['signal']:>14s}  k={r['k_frac']:<5} ppl={r['ppl']:.4f}")
PY
```

**Report** the `k_frac=0.02` perplexity for each signal on each base. The padded reference values
for 1B are already in the repo at `runs/final/sweeps/gptq_llmc/Llama-3.2-1B/scalars/seed-1234/`
and `.../residual_max/seed-1234/` (and the `hqq/` equivalents) — do not re-run those.

---

## Job B — selector-calibration sample sensitivity on Llama-3.2-3B (≈2.5 h)

**Why.** §5.3 compares the selector's own 128×2048 Hessian against the base's, and the audit
correctly showed the comparison is **size-matched, not sample-matched**: the LightCompress base
used its own preprocessing (`wikitext2_gptq`, seed 0) and its calibration tokens were never saved,
so token identity is not recoverable for that base. What *is* recoverable is a **sensitivity
bound**: hold the token budget fixed at 128×2048 and vary only the selector's calibration draw.
If greedy@GPTQ is stable across independent draws, sample choice cannot explain the §5.3 effect.

`--selector_calib_seed` (new) decouples the calibration draw from `--seed`, which also drives
channel selection — so `--seed` stays at 1234 in all three arms and only the draw changes.

**B0 — regression arm (must reproduce the published number exactly).** Published §5.3 value:
`results/e5_matchedcalib_Llama-3.2-3B/channel_pareto.json`, greedy @ 0.02 = **58.2386**.

```bash
python -m seq_core.channel_sweep \
  --model meta-llama/Llama-3.2-3B --backend hqq --base_bits 4 \
  --base_quantizer gptq_llmc \
  --gptq_model_path "$PWD/runs/final/llmc/Llama-3.2-3B/gptq/artifacts/fake_quant_model" \
  --select greedy --protect_fracs 0.02,0.05,0.1,0.2 \
  --selector_calib_samples 128 --selector_calib_seed 1234 \
  --seed 1234 --ppl_mode canonical --skip_lm_head \
  --calibration_prompts calibration_prompts.json \
  --out_dir results/calibsens_3B_seed1234 \
  2>&1 | tee results/calibsens_3B_seed1234.log
```

If B0's greedy@0.02 is not 58.2386 ± 0.01, **stop and report** — the new flag has changed the
default path, which is a bug, and B1/B2 would be uninterpretable.

**B1 and B2 — independent draws at the same budget.** Identical except the two marked values:

```bash
for S in 2345 3456; do
  python -m seq_core.channel_sweep \
    --model meta-llama/Llama-3.2-3B --backend hqq --base_bits 4 \
    --base_quantizer gptq_llmc \
    --gptq_model_path "$PWD/runs/final/llmc/Llama-3.2-3B/gptq/artifacts/fake_quant_model" \
    --select greedy --protect_fracs 0.02,0.05,0.1,0.2 \
    --selector_calib_samples 128 --selector_calib_seed "$S" \
    --seed 1234 --ppl_mode canonical --skip_lm_head \
    --calibration_prompts calibration_prompts.json \
    --out_dir "results/calibsens_3B_seed${S}" \
    2>&1 | tee "results/calibsens_3B_seed${S}.log"
done
```

**Gate.** Each JSON must record its own `selector_calib_seed` and
`selector_calib_source = "wikitext2:128x2048"`:

```bash
python - <<'PY'
import json, statistics
rows = []
for s in (1234, 2345, 3456):
    f = f"results/calibsens_3B_seed{s}/channel_pareto.json"
    d = json.load(open(f))
    assert d["selector_calib_seed"] == s, f"{f}: selector_calib_seed={d['selector_calib_seed']}"
    assert d["selector_calib_samples"] == 128
    print(f"== seed {s}  source={d['selector_calib_source']}  seed(sel)={d['seed']}")
    for r in sorted(d["results"], key=lambda r: r["k_frac"]):
        print(f"     k={r['k_frac']:<5} ppl={r['ppl']:.4f}")
        if r["k_frac"] == 0.02:
            rows.append(r["ppl"])
print(f"\ngreedy@0.02 across draws: {[round(x,4) for x in rows]}")
print(f"  mean={statistics.mean(rows):.4f}  sd={statistics.pstdev(rows):.4f}  "
      f"range={max(rows)-min(rows):.4f}")
print(f"  published size-matched value 58.2386; full-calibration value 51.68")
PY
```

**Report** the three greedy@0.02 perplexities, their mean/sd/range, and whether the spread is
small relative to the 58.24 vs 51.68 gap that §5.3 attributes to estimator quality.

---

## Job C — restore the per-example logs behind three published CIs (≈1 h)

**Why.** `analysis/build_downstream_table.py` computes the paired bootstrap from lm-eval's
`samples_<task>_*.jsonl` files. Three points have their `results_*.json` but **no** sample logs, so
the repo cannot today reproduce the intervals it publishes:

| point | CI it backs |
|---|---|
| `Llama-3.2-3B/greedy_gptq` | **+0.51 [+0.13, +0.86]** — the headline |
| `Llama-3.2-1B/greedy_gptq` | +0.32 [−0.07, +0.73] — the replication |
| `Llama-3.2-3B/random_hqq`  | +0.92 [+0.50, +1.34] — §7 |

The published numbers are sound (each bootstrap's central estimate matches the two arms' current
accuracies to 1e-9, and every `per_task.n` is a full set size) — the logs simply were never
committed after the re-runs. See `docs/PAPER_GAPS.md` Issue 4.

**The accuracies must not move.** These are the same checkpoints; a re-run only adds the sample
logs. Expected macro accuracy: 3B `greedy_gptq` **67.61**, 1B `greedy_gptq` **58.67**,
3B `random_hqq` **67.21**.

```bash
bash scripts/run_downstream_eval.sh --models meta-llama/Llama-3.2-3B --points greedy_gptq \
  2>&1 | tee results/relog_3B_greedy.log
bash scripts/run_downstream_eval.sh --models meta-llama/Llama-3.2-1B --points greedy_gptq \
  2>&1 | tee results/relog_1B_greedy.log
bash scripts/run_downstream_eval.sh --models meta-llama/Llama-3.2-3B --points random_hqq \
  2>&1 | tee results/relog_3B_random_hqq.log
```

Do **not** pass `--resume`: a stale checkpoint reused under `--resume` is the exact failure that
produced a wrong F3 row once already. If the script re-exports the checkpoints, reload-validate
before trusting anything downstream:

```bash
python scripts/validate_saved_seq_reload.py runs/final/downstream/checkpoints/Llama-3.2-3B/greedy_gptq --expected 51.68 --tolerance 0.5
python scripts/validate_saved_seq_reload.py runs/final/downstream/checkpoints/Llama-3.2-1B/greedy_gptq --expected 63.95 --tolerance 0.5
```

**Gate — the recomputed CIs must reproduce the published ones:**

```bash
python analysis/build_downstream_table.py --root runs/final/downstream \
  --config configs/downstream_operating_points.json --out docs/DOWNSTREAM.md \
  --csv results/downstream.csv --json results/downstream.json

python - <<'PY'
import json
d = json.load(open("results/downstream.json"))
expect = {("meta-llama/Llama-3.2-3B","greedy_gptq_vs_gptq4"):      (+0.51, +0.13, +0.86),
          ("meta-llama/Llama-3.2-1B","greedy_gptq_vs_gptq4"):      (+0.32, -0.07, +0.73),
          ("meta-llama/Llama-3.2-3B","best_hqq_vs_random_hqq"):    (+0.92, +0.50, +1.34)}
for (model, name), (diff, lo, hi) in expect.items():
    c = d["contrasts"][model][name]
    m = c.get("macro_avg") or {}
    got = tuple(round(100*m[k], 2) for k in ("diff","lo","hi")) if m.get("diff") is not None else None
    print(f"{model:26s} {name:26s} paired={c.get('paired')} published={(diff,lo,hi)} recomputed={got}")
    assert c.get("paired") is True, "still unpaired -- sample logs did not land"
PY
grep -c "UNPAIRED" docs/DOWNSTREAM.md    # expect 0
```

If any recomputed interval differs from the published one by more than ~0.02 pts, **stop and
report both**; do not commit the regenerated table. Bootstrap resampling is seeded, so small
differences are still worth reporting rather than absorbing.

`docs/DOWNSTREAM.md` will also gain an `n/task` column: the three 1B points `fp16`, `hqq4` and
`best_hqq` are expected to show `⚠ 200` (a known, already-documented reduced scope that the paper
does not cite). That is correct output, not a failure.

---

## Commit

```bash
git add -f results/nopad_gptq_Llama-3.2-1B/channel_pareto.json results/nopad_gptq_Llama-3.2-1B.log \
           results/nopad_hqq_Llama-3.2-1B/channel_pareto.json  results/nopad_hqq_Llama-3.2-1B.log \
           results/calibsens_3B_seed*/channel_pareto.json      results/calibsens_3B_seed*.log
git status --short          # verify NOTHING under runs/final/ or any *.safetensors is staged
git commit -m "Audit follow-ups: 1B no-pad calibration robustness; 3B selector-calibration sample sensitivity"
git push origin main
```

## What to report back

1. Preflight: did all four flags appear; was the 1B base present or regenerated (and its PPL).
2. **Job A**: the `k_frac=0.02` perplexity per signal for both bases, plus each run's `fp16`
   baseline, and confirmation that `pad_calibration` is `false` in both JSONs.
3. **Job B**: whether B0 reproduced 58.2386; the three greedy@0.02 values; mean/sd/range.
4. **Job C**: the three recomputed CIs next to the published ones; whether `paired` is now `True`
   for all three; the three macro accuracies (expect 67.61 / 58.67 / 67.21); whether any
   checkpoint had to be re-exported and its reload-validation value; and the `grep -c "UNPAIRED"`
   count (expect 0).
5. Anything that failed a gate, verbatim, with the surrounding log lines.

Do **not** edit `docs/FINDINGS_PAPER.md`, `paper/main.tex`, or `docs/AUDIT_RESPONSE.md` — the
local box folds these numbers into the paper.

---

## Notes for the write-up (local box, not for Codex)

Job B bounds **sensitivity to the calibration sample**. It does **not** establish token identity
with the LightCompress base, and §5.3 must continue to say the comparison is size-matched rather
than sample-matched. `scripts/dump_calibration_tokens.py` + `--selector_calib_tokens` make a
genuinely token-identical comparison possible for any base we generate ourselves in future; they
cannot retroactively recover the external base's tokens.
