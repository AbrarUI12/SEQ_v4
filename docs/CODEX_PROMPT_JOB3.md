# Codex prompt — Job 3 (v2, after the 1B base failure)

Paste everything between the rulers into Codex on the research PC.

**What changed from v1.** v1 hardcoded the repo at `/mnt/d/Abrar/SEQ/seq_v4`; the box is actually at
`/mnt/e/seq v4/SEQ-clean-v4` (note the **space**). v1 also checked for the 1B GPTQ base only as a
Job A precondition, while Job C's 1B arm needs it too — so the run got 40 minutes in before failing.
v2 derives every path, gates all jobs up front, and regenerates the base through the renderer rather
than the stale committed `config.yml` (whose `save_path` is an absolute path to the old box).

---

You are working in the SEQ research repo on a WSL box with an RTX 5090/4090.

## Setup

```bash
cd "/mnt/e/seq v4/SEQ-clean-v4"     # or wherever the repo is; everything below derives from it
git pull origin main
source .venv-seq/bin/activate

REPO="$(git rev-parse --show-toplevel)"; export REPO
echo "REPO=[$REPO]"
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
```

**The repo path contains a space.** Every expansion of `$REPO` below is double-quoted. Keep it that
way if you adapt anything — an unquoted `$REPO` will split into two arguments and fail confusingly.

`results/` is `.gitignore`d, so result JSONs must be **force-added** (`git add -f`). **Never
`git add` model weights, `checkpoints/`, or any `fake_quant_model/` directory.** Never point
`--out_dir` at `/tmp` — WSL's `/tmp` is tmpfs and is wiped on idle/shutdown.

---

## Step 0 — bank the 3B work that already succeeded (do this first)

The previous run completed 3B `greedy_gptq` and restored its per-example sample logs, but nothing
was committed. Those logs are the artifact that closes the paper's headline confidence interval, and
a `wsl --shutdown` would lose them. Save them before doing anything else.

```bash
cd "$REPO"
git add -f "runs/final/downstream/Llama-3.2-3B/greedy_gptq/lm_eval" || true
git add -f results/relog_3B_greedy.log 2>/dev/null || true

# MUST print 'clean'. If it prints STOP, unstage the offending path and re-check.
if git status --short | grep -qE "\.safetensors|\.bin$|/checkpoints/|fake_quant_model"; then
  echo "STOP: weights staged"; else echo "clean"; fi

git status --short
git commit -m "Restore per-example sample logs for 3B greedy_gptq (PAPER_GAPS Issue 4, partial)"
git push origin main
```

Do **not** regenerate `docs/DOWNSTREAM.md` yet — with 1B still missing its logs, regenerating would
downgrade that row to `UNPAIRED approx`. The table gets rebuilt once both models have their logs.

Report how many `samples_*.jsonl` files this committed (expect 6).

---

## Preflight — gates every job below (2 min, do not skip)

```bash
cd "$REPO"

# 1. The flags added after the audit must exist. Any missing line = stale commit, STOP.
python -m seq_core.channel_sweep --help | grep -E "no_pad_calibration|selector_calib_seed|selector_calib_tokens|selector_calib_samples"

# 2. Which GPTQ bases exist? (weights are not in git, so these are per-machine)
for M in Llama-3.2-1B Llama-3.2-3B; do
  D="$REPO/runs/final/llmc/$M/gptq/artifacts/fake_quant_model"
  if [ -d "$D" ]; then echo "OK   $M"; else echo "MISSING $M"; fi
done

# 3. Is LightCompress available? Needed only to rebuild a missing base.
LLMC_REPO="${LLMC_REPO:-/mnt/d/LightCompress}"
if [ -d "$LLMC_REPO" ]; then echo "OK   LLMC at $LLMC_REPO"; else echo "MISSING LightCompress"; fi
export LLMC_REPO
```

**Read the result before proceeding:**

| 1B base | 3B base | LightCompress | What to run |
|---|---|---|---|
| OK | OK | — | Everything: Job C, then A, then B |
| MISSING | OK | present | Rebuild 1B (below), then Job C, A, B |
| MISSING | OK | **absent** | **Job B only.** Report that C's 1B arm and Job A are blocked |
| any MISSING | MISSING | present | Rebuild whichever is missing, then proceed |

Do not improvise around a missing base. Report and run what is runnable.

### Rebuilding a missing GPTQ base (≈6 min for 1B)

Use the renderer, **not** `runs/final/llmc/Llama-3.2-1B/gptq/config.yml` — that committed config has
an absolute `save_path` pointing at the old machine, so re-running it writes the artifacts somewhere
that does not exist here. `scripts/run_llmc_w4_baselines.py` re-renders `save_path` from the current
repo.

```bash
cd "$REPO"
python scripts/run_llmc_w4_baselines.py \
  --model meta-llama/Llama-3.2-1B --model-type Llama --methods gptq \
  --llmc-repo "$LLMC_REPO" --llmc-venv "$LLMC_REPO/.venv-llmc" \
  --out-dir runs/final/llmc/Llama-3.2-1B --force \
  2>&1 | tee results/rebuild_1B_gptq_base.log
```

**Gate — the rebuilt base must match the published one:**

```bash
python - <<'PY'
import json, os
p = os.path.join(os.environ["REPO"], "runs/final/llmc/Llama-3.2-1B/gptq/summary.json")
d = json.load(open(p))
print("status =", d.get("status"), " ppl =", d.get("ppl"))
assert abs(d["ppl"] - 10.4306) < 0.05, f"base moved: {d['ppl']} vs recorded 10.4306"
print("OK: 1B GPTQ base reproduces its recorded perplexity")
PY
ls -d "$REPO/runs/final/llmc/Llama-3.2-1B/gptq/artifacts/fake_quant_model"
```

If the perplexity moved by more than ~0.05, **stop and report it**. Every 1B number in the paper sits
downstream of this base, so a base that does not reproduce invalidates the comparison rather than
merely being inconvenient.

---

## Job C — restore the per-example logs behind the published CIs (≈1 h, highest priority)

**Why.** `analysis/build_downstream_table.py` computes the paired bootstrap from lm-eval's
`samples_<task>_*.jsonl`. Three points have their `results_*.json` but no sample logs, so the repo
cannot currently reproduce the intervals it publishes: 3B `greedy_gptq` (**+0.51 [+0.13, +0.86]**,
the headline), 1B `greedy_gptq` (+0.32), 3B `random_hqq` (+0.92). The numbers are verified sound —
each bootstrap's central estimate matches its arms' current accuracies to 1e-9 and every
`per_task.n` is a full set size — the logs were simply never committed. See `docs/PAPER_GAPS.md`
Issue 4.

**These are re-runs of existing checkpoints, so the accuracies must not move.** Expected macro:
3B `greedy_gptq` **67.61**, 1B `greedy_gptq` **58.67**, 3B `random_hqq` **67.21**.

Step 0 already did 3B `greedy_gptq`. Run the remaining two:

```bash
cd "$REPO"
bash scripts/run_downstream_eval.sh --models meta-llama/Llama-3.2-1B --points greedy_gptq \
  2>&1 | tee results/relog_1B_greedy.log
bash scripts/run_downstream_eval.sh --models meta-llama/Llama-3.2-3B --points random_hqq \
  2>&1 | tee results/relog_3B_random_hqq.log
```

**Do not pass `--resume`.** `run_downstream_eval.sh` skips the export only under `--resume`, which is
exactly why the base is required — and a stale `--resume` checkpoint already produced a wrong F3 row
once in this project. Rebuilding the base is the correct fix; reaching for `--resume` is not.

Because the export runs fresh, reload-validate before trusting anything downstream:

```bash
python scripts/validate_saved_seq_reload.py \
  "$REPO/runs/final/downstream/checkpoints/Llama-3.2-1B/greedy_gptq" --expected 63.95 --tolerance 0.5
```

**Gate — recomputed CIs must reproduce the published ones:**

```bash
cd "$REPO"
python analysis/build_downstream_table.py --root runs/final/downstream \
  --config configs/downstream_operating_points.json --out docs/DOWNSTREAM.md \
  --csv results/downstream.csv --json results/downstream.json

python - <<'PY'
import json
d = json.load(open("results/downstream.json"))
expect = {("meta-llama/Llama-3.2-3B","greedy_gptq_vs_gptq4"):   (+0.51, +0.13, +0.86),
          ("meta-llama/Llama-3.2-1B","greedy_gptq_vs_gptq4"):   (+0.32, -0.07, +0.73),
          ("meta-llama/Llama-3.2-3B","best_hqq_vs_random_hqq"): (+0.92, +0.50, +1.34)}
bad = []
for (model, name), pub in expect.items():
    c = d["contrasts"][model][name]; m = c.get("macro_avg") or {}
    got = tuple(round(100*m[k], 2) for k in ("diff","lo","hi")) if m.get("diff") is not None else None
    print(f"{model:26s} {name:26s} paired={c.get('paired')} published={pub} recomputed={got}")
    if c.get("paired") is not True or got is None: bad.append(name)
    elif max(abs(a-b) for a, b in zip(got, pub)) > 0.02: bad.append(name)
print("MISMATCHES:", bad or "none")
PY
grep -c "UNPAIRED" docs/DOWNSTREAM.md    # expect 0
```

If any interval differs by more than ~0.02 pts, or `paired` is not `True`, **stop and report both
the published and recomputed values**. Do not commit the regenerated table in that case.

`docs/DOWNSTREAM.md` will also gain an `n/task` column, and the three 1B points `fp16`, `hqq4` and
`best_hqq` are expected to show `⚠ 200`. That is a known, already-documented reduced scope that the
paper does not cite — correct output, not a failure.

---

## Job A — no-pad calibration robustness on Llama-3.2-1B (≈45 min)

**Requires the 1B GPTQ base.** Skip and report if it could not be rebuilt.

**Why.** The audit showed the scalar selectors' statistics were collected on prompts padded to 2048,
so they were pad-dominated. We re-measured on real tokens for 3B (commit `9f6cc8a`) and the ordering
survived: three of four scalars unmoved, only `act_scale` shifted (8.586 → 8.142). This repeats it on
1B so §5.1 is not a single-model claim. These are the 3B commands with the model swapped — change
nothing else.

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

**Gate.** Both JSONs must record `"pad_calibration": false`:

```bash
python - <<'PY'
import json
for f in ["results/nopad_gptq_Llama-3.2-1B/channel_pareto.json",
          "results/nopad_hqq_Llama-3.2-1B/channel_pareto.json"]:
    d = json.load(open(f))
    assert d["pad_calibration"] is False, f"{f}: pad_calibration is not false"
    print(f"\n== {f}  fp16={d['baseline_fp16_ppl']:.4f}")
    for r in sorted(d["results"], key=lambda r: (r["signal"], r["k_frac"])):
        print(f"   {r['signal']:>14s}  k={r['k_frac']:<5} ppl={r['ppl']:.4f}")
PY
```

**Report** the `k_frac=0.02` perplexity per signal per base. Padded references for 1B are already in
the repo under `runs/final/sweeps/gptq_llmc/Llama-3.2-1B/` — do not re-run those.

---

## Job B — selector-calibration sample sensitivity on Llama-3.2-3B (≈2.5 h)

**Requires only the 3B base**, so this runs even if the 1B base could not be rebuilt.

**Why.** §5.3 compares the selector's own 128×2048 Hessian against the base's, and the audit showed
that comparison is **size-matched, not sample-matched**: the LightCompress base used its own
preprocessing (`wikitext2_gptq`, seed 0) and its calibration tokens were never saved, so token
identity is unrecoverable. What *is* recoverable is a sensitivity bound — hold the budget at
128×2048 and vary only the draw. `--selector_calib_seed` decouples that draw from `--seed`, which
also drives channel selection, so `--seed` stays 1234 in all three arms.

**B0 — regression arm. Published §5.3 value: greedy @ 0.02 = 58.2386.**

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

If B0's greedy@0.02 is not 58.2386 ± 0.01, **stop and report** — the new flag has changed the default
path, which is a bug, and B1/B2 would be uninterpretable.

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
```

**Gate:**

```bash
python - <<'PY'
import json, statistics
rows = []
for s in (1234, 2345, 3456):
    f = f"results/calibsens_3B_seed{s}/channel_pareto.json"
    d = json.load(open(f))
    assert d["selector_calib_seed"] == s, f"{f}: selector_calib_seed={d['selector_calib_seed']}"
    assert d["selector_calib_samples"] == 128
    print(f"== seed {s}  source={d['selector_calib_source']}")
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
git add -f results/nopad_*_Llama-3.2-1B/channel_pareto.json results/nopad_*_Llama-3.2-1B.log \
           results/calibsens_3B_seed*/channel_pareto.json   results/calibsens_3B_seed*.log \
           results/relog_*.log results/rebuild_1B_gptq_base.log 2>/dev/null

git add -f "runs/final/downstream/Llama-3.2-1B/greedy_gptq/lm_eval" \
           "runs/final/downstream/Llama-3.2-3B/random_hqq/lm_eval"
git add docs/DOWNSTREAM.md results/downstream.csv results/downstream.json

if git status --short | grep -qE "\.safetensors|\.bin$|/checkpoints/|fake_quant_model"; then
  echo "STOP: weights staged"; else echo "clean"; fi

git commit -m "Job 3: restore per-example logs; 1B no-pad robustness; 3B calibration sample sensitivity"
git push origin main
```

## What to report back

1. **Step 0**: how many `samples_*.jsonl` were committed for 3B `greedy_gptq`, and the push result.
2. **Preflight**: which bases were OK/MISSING, whether LightCompress was present, which row of the
   decision table applied, and — if the 1B base was rebuilt — its `summary.json` ppl.
3. **Job C**: the three recomputed CIs beside the published ones; whether `paired` is `True` for all
   three; the macro accuracies (expect 67.61 / 58.67 / 67.21); the 1B reload-validation value; and
   the `grep -c "UNPAIRED"` count (expect 0).
4. **Job A**: `k_frac=0.02` perplexity per signal for both bases, each run's fp16 baseline, and that
   `pad_calibration` is `false` in both JSONs.
5. **Job B**: whether B0 reproduced 58.2386; the three greedy@0.02 values; mean/sd/range.
6. Anything that failed a gate, **verbatim**, with the surrounding log lines — and stop there rather
   than working around it.

Do **not** edit `docs/FINDINGS_PAPER.md`, `paper/main.tex` or `docs/AUDIT_RESPONSE.md` — the local
box folds these numbers into the paper.

---

## Note for the write-up (local box, not for Codex)

Job B bounds sensitivity to the calibration **sample**. It does not establish token identity with the
LightCompress base, so §5.3 must continue to say the comparison is size-matched rather than
sample-matched. `scripts/dump_calibration_tokens.py` + `--selector_calib_tokens` make a genuinely
token-identical comparison possible for any base we generate ourselves in future; they cannot
retroactively recover the external base's tokens.
