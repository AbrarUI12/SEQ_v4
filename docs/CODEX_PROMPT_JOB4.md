# Codex prompt — Job 4 (diagnose the 1B reload failure; run the two independent jobs)

Paste everything between the rulers into Codex on the research PC.

## What happened

Job 3 banked the 3B sample logs (commit `9e749a7`) — the headline evidence is safe. It then rebuilt
the missing 1B GPTQ base (PPL **10.4047** against a recorded **10.4306**, which passed my ±0.05
gate) and exported 1B `greedy@GPTQ`, which **reloaded at 104.15 against an expected 63.95**.

That is the third distinct value this one export has produced: **203.72** (original, disclosed as a
failure in Appendix A), **63.83** (the re-export the paper currently reports), and now **104.15**.
Two hypotheses, and one cheap run separates them:

- **H1 — the base changed the answer.** The rebuilt base is not bit-identical to the one the paper's
  63.95 came from. Because `greedy` ranks channels by a Hessian-weighted residual objective computed
  *against the base*, a small change in the base can select a different channel set entirely. This
  is the paper's own thesis about fragility, so it would be a result, not a bug.
- **H2 — the export/reload path is broken on 1B.** The 203.72 incident was never root-caused, only
  papered over by re-exporting. If runtime and reload disagree, that is a real defect and it would
  also put the 3B chain in question.

**Also, correcting my own instruction from Job 3:** I told you to stop everything on any failed
gate. That was wrong and it cost you three hours. Jobs A and B have **no dependency** on Job C —
A is scalar selectors on 1B, B is greedy on 3B against its own intact base. Run them regardless of
what the diagnostic says. Only stop the work that actually depends on a failed gate.

---

You are working in the SEQ research repo on a WSL box with an RTX 5090/4090.

```bash
cd "/mnt/e/seq v4/SEQ-clean-v4"
git pull origin main
source .venv-seq/bin/activate
REPO="$(git rev-parse --show-toplevel)"; export REPO
echo "REPO=[$REPO]"
```

The repo path contains a space — every `$REPO` below is double-quoted; keep it that way.
`results/` is `.gitignore`d, so force-add result files. Never `git add` weights, `checkpoints/`,
or `fake_quant_model/`. Never write `--out_dir` to `/tmp`.

Run in this order: **D** (20 min, decisive), then **A** (45 min), then **B** (2.5 h).

---

## Job D — why did the 1B export reload at 104? (≈20 min, do first)

### D1. Which LightCompress built the new base? (30 seconds)

The recorded base was built with LightCompress at commit
`86f564ddb1d6548b228c67a10509a4ed7264345c` from `/mnt/d/LightCompress`. You rebuilt from
`/mnt/e/LightCompress`. **If that is a different commit, it is the most likely cause of everything
below** — a different GPTQ implementation produces a different compensated base, hence a different
residual, hence a different greedy channel set.

```bash
cd "$REPO"
python -c "
import json;d=json.load(open('runs/final/llmc/Llama-3.2-1B/gptq/summary.json'))
print('rebuilt commit :', d.get('llmc_commit'))
print('rebuilt ppl    :', d.get('ppl'))
print('recorded commit: 86f564ddb1d6548b228c67a10509a4ed7264345c')
print('recorded ppl   : 10.430636405944824')
"
git -C /mnt/e/LightCompress log --oneline -1
git -C /mnt/e/LightCompress status --short | head -5
```

**Report both commits.** If they differ, also report whether `/mnt/e/LightCompress` can be checked
out at `86f564d` — do **not** check it out yet, just say whether the commit exists there.

### D2. Runtime vs materialized vs reload (≈15 min)

This is the decisive measurement. `--verify_materialized` computes perplexity twice inside one
process — once through the protection forward pass, once after collapsing it to dense weights — so
it isolates the export from the selection.

```bash
cd "$REPO"
python -m seq_core.channel_sweep \
  --model meta-llama/Llama-3.2-1B --backend hqq --base_bits 4 \
  --base_quantizer gptq_llmc \
  --gptq_model_path "$REPO/runs/final/llmc/Llama-3.2-1B/gptq/artifacts/fake_quant_model" \
  --select greedy --protect_fracs 0.02 \
  --seed 1234 --ppl_mode canonical --skip_lm_head \
  --calibration_prompts calibration_prompts.json \
  --verify_materialized \
  --out_dir results/d2_verify_1B_rebuilt \
  2>&1 | tee results/d2_verify_1B_rebuilt.log

grep -E "verify_materialized|baseline_base_ppl|FP16 baseline" results/d2_verify_1B_rebuilt.log | tail -5
python -c "
import json;d=json.load(open('results/d2_verify_1B_rebuilt/channel_pareto.json'))
print('base ppl :', d.get('baseline_base_ppl'))
print('fp16 ppl :', d.get('baseline_fp16_ppl'))
for r in d['results']: print(' greedy k=%s runtime_ppl=%.4f' % (r['k_frac'], r['ppl']))
"
```

**How to read it — report which case you see, do not try to fix it:**

| runtime | materialized | vs reload 104.15 | Meaning |
|---|---|---|---|
| ≈104 | ≈104 | agrees | **H1.** Export is faithful; the rebuilt base genuinely selects a different, worse channel set. A result, not a bug. |
| ≈63.9 | ≈63.9 | disagrees | **H2.** Selection is fine, the save→disk→reload path is corrupting the checkpoint. Serious. |
| ≈63.9 | ≈104 | agrees | **H2 variant.** The in-memory materialization is what breaks. Serious. |
| anything else | | | Report the three numbers and stop. |

Whatever the case, **report `base ppl`, `fp16 ppl`, `runtime_ppl`, `materialized_ppl`, and the
reload 104.15 together.** Do not re-export or re-run to "get a better number" — the disagreement
*is* the measurement.

### D3. Is the selection itself deterministic? (≈3 min, only if D2 showed H1)

Same command, same seed, different output directory. If two identical runs on the same base give
different perplexities, the selector is non-deterministic and that is its own finding.

```bash
cd "$REPO"
python -m seq_core.channel_sweep \
  --model meta-llama/Llama-3.2-1B --backend hqq --base_bits 4 \
  --base_quantizer gptq_llmc \
  --gptq_model_path "$REPO/runs/final/llmc/Llama-3.2-1B/gptq/artifacts/fake_quant_model" \
  --select greedy --protect_fracs 0.02 \
  --seed 1234 --ppl_mode canonical --skip_lm_head \
  --calibration_prompts calibration_prompts.json \
  --out_dir results/d3_repeat_1B_rebuilt \
  2>&1 | tee results/d3_repeat_1B_rebuilt.log
```

Report whether D3's greedy@0.02 equals D2's to 4 decimal places.

**Then continue to A and B regardless of what D showed.**

---

## Job A — no-pad calibration robustness on Llama-3.2-1B (≈45 min)

The 1B base now exists, so this runs. These are scalar activation selectors — they never touch the
Hessian or the residual, so they are unaffected by the greedy instability above.

```bash
cd "$REPO"; mkdir -p results
python -m seq_core.channel_sweep \
  --model meta-llama/Llama-3.2-1B --backend hqq --base_bits 4 \
  --base_quantizer gptq_llmc \
  --gptq_model_path "$REPO/runs/final/llmc/Llama-3.2-1B/gptq/artifacts/fake_quant_model" \
  --signals act_max,act_scale,residual_max,residual_rms \
  --protect_fracs 0.02,0.05,0.1,0.2 \
  --seed 1234 --ppl_mode canonical --skip_lm_head \
  --calibration_prompts calibration_prompts.json \
  --no_pad_calibration \
  --out_dir results/nopad_gptq_Llama-3.2-1B \
  2>&1 | tee results/nopad_gptq_Llama-3.2-1B.log

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

**Gate:**

```bash
python - <<'PY'
import json
for f in ["results/nopad_gptq_Llama-3.2-1B/channel_pareto.json",
          "results/nopad_hqq_Llama-3.2-1B/channel_pareto.json"]:
    d = json.load(open(f))
    assert d["pad_calibration"] is False, f"{f}: pad_calibration is not false"
    print(f"\n== {f}  fp16={d['baseline_fp16_ppl']:.4f}  base={d.get('baseline_base_ppl')}")
    for r in sorted(d["results"], key=lambda r: (r["signal"], r["k_frac"])):
        print(f"   {r['signal']:>14s}  k={r['k_frac']:<5} ppl={r['ppl']:.4f}")
PY
```

Note the GPTQ arm now sits on the **rebuilt** base, so its numbers are not directly comparable with
the padded 1B references in `runs/final/sweeps/`. Report the base perplexity alongside, and I will
handle the comparison in the write-up. The HQQ arm is unaffected — HQQ is data-free and rebuilt
identically every time.

---

## Job B — selector-calibration sample sensitivity on Llama-3.2-3B (≈2.5 h)

Uses only the 3B base, which is intact and untouched by any of the above.

**B0 — regression arm. Published value: greedy @ 0.02 = 58.2386.**

```bash
cd "$REPO"
python -m seq_core.channel_sweep \
  --model meta-llama/Llama-3.2-3B --backend hqq --base_bits 4 \
  --base_quantizer gptq_llmc \
  --gptq_model_path "$REPO/runs/final/llmc/Llama-3.2-3B/gptq/artifacts/fake_quant_model" \
  --select greedy --protect_fracs 0.02,0.05,0.1,0.2 \
  --selector_calib_samples 128 --selector_calib_seed 1234 \
  --seed 1234 --ppl_mode canonical --skip_lm_head \
  --calibration_prompts calibration_prompts.json \
  --out_dir results/calibsens_3B_seed1234 \
  2>&1 | tee results/calibsens_3B_seed1234.log
```

If B0's greedy@0.02 is not 58.2386 ± 0.01, **report it and still run B1/B2** — given what Job D is
investigating, a 3B drift would itself be important information, and the three-seed spread is
interpretable as long as all three arms use the same base.

**B1, B2 — independent draws at the same budget:**

```bash
cd "$REPO"
for S in 2345 3456; do
  python -m seq_core.channel_sweep \
    --model meta-llama/Llama-3.2-3B --backend hqq --base_bits 4 \
    --base_quantizer gptq_llmc \
    --gptq_model_path "$REPO/runs/final/llmc/Llama-3.2-3B/gptq/artifacts/fake_quant_model" \
    --select greedy --protect_fracs 0.02,0.05,0.1,0.2 \
    --selector_calib_samples 128 --selector_calib_seed "$S" \
    --seed 1234 --ppl_mode canonical --skip_lm_head \
    --calibration_prompts calibration_prompts.json \
    --out_dir "results/calibsens_3B_seed${S}" \
    2>&1 | tee "results/calibsens_3B_seed${S}.log"
done

python - <<'PY'
import json, statistics
rows = []
for s in (1234, 2345, 3456):
    d = json.load(open(f"results/calibsens_3B_seed{s}/channel_pareto.json"))
    assert d["selector_calib_seed"] == s
    print(f"== seed {s}  source={d['selector_calib_source']}  base={d.get('baseline_base_ppl')}")
    for r in sorted(d["results"], key=lambda r: r["k_frac"]):
        print(f"     k={r['k_frac']:<5} ppl={r['ppl']:.4f}")
        if r["k_frac"] == 0.02: rows.append(r["ppl"])
print(f"\ngreedy@0.02 across draws: {[round(x,4) for x in rows]}")
print(f"  mean={statistics.mean(rows):.4f}  sd={statistics.pstdev(rows):.4f}  range={max(rows)-min(rows):.4f}")
print("  published size-matched 58.2386; full-calibration 51.68")
PY
```

---

## Commit

```bash
cd "$REPO"
git add -f results/d2_verify_1B_rebuilt/channel_pareto.json results/d2_verify_1B_rebuilt.log \
           results/d3_repeat_1B_rebuilt/channel_pareto.json results/d3_repeat_1B_rebuilt.log \
           results/nopad_*_Llama-3.2-1B/channel_pareto.json results/nopad_*_Llama-3.2-1B.log \
           results/calibsens_3B_seed*/channel_pareto.json  results/calibsens_3B_seed*.log \
           results/rebuild_1B_gptq_base.log 2>/dev/null

# Also commit the rebuilt base's summary/config so its provenance is on the record.
git add -f runs/final/llmc/Llama-3.2-1B/gptq/summary.json runs/final/llmc/Llama-3.2-1B/gptq/config.yml

if git status --short | grep -qE "\.safetensors|\.bin$|/checkpoints/|fake_quant_model"; then
  echo "STOP: weights staged"; else echo "clean"; fi

git commit -m "Job 4: 1B export diagnostic on the rebuilt base; 1B no-pad; 3B calibration sensitivity"
git push origin main
```

## What to report back

1. **D1** — the rebuilt `llmc_commit` vs the recorded `86f564d...`, the rebuilt ppl, and whether
   `86f564d` exists in `/mnt/e/LightCompress`.
2. **D2** — `base ppl`, `fp16 ppl`, `runtime_ppl`, `materialized_ppl`, and which row of the table
   you matched. This is the most important thing in the whole run.
3. **D3** — whether the repeat reproduced D2 to 4 dp.
4. **A** — `k_frac=0.02` perplexity per signal for both bases, plus each run's base and fp16.
5. **B** — B0 vs 58.2386; the three greedy@0.02 values; mean/sd/range.
6. Anything else that failed, verbatim — but **only stop the work that depends on it**.

Do not edit `docs/FINDINGS_PAPER.md`, `paper/main.tex` or `docs/AUDIT_RESPONSE.md`.
