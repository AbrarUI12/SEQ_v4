# Verification gaps — evidence and required paper changes

Four issues found when checking the draft against the result artifacts (Issue 4 added 2026-07-28). All three are confirmed;
one is worse than first reported, and it exposes a metadata bug. **The headline result is
unaffected** — that is established first, because it determines how much of the paper is at risk.

---

## Headline is safe

The §4 decoupling rests on `greedy_gptq` vs `gptq4`, on both models. Both points, on both models,
were evaluated on the **full test sets**:

| model | point | n/task | source |
|---|---|---|---|
| 3B | `gptq4`, `greedy_gptq` | full (1172/1267/1838/2376/5153/10042) | `runs/final/downstream/*/lm_eval/*/results_*.json` |
| 1B | `gptq4`, `greedy_gptq` | full (same) | ” |

So `+0.51 [+0.13, +0.86]` (3B) and `+0.32 [−0.07, +0.73]` (1B) are full-set vs full-set. No change
required to §4.

---

## Issue 1 — three 1B points were evaluated at n=200/task, and the metadata says otherwise

**Confirmed and broader than reported.** On Llama-3.2-1B, `fp16`, `hqq4` and `best_hqq` were
evaluated at **200 examples per task**; every other point on 1B, and every point on 3B, used the
full sets. Full audit in `results/lambada_and_eval_scope.csv` (column `n_samples_per_task`).

**The sidecar metadata is wrong.** `runs/final/downstream/Llama-3.2-1B/{fp16,hqq4,best_hqq}/seq_meta.json`
all record `"limit": null`, i.e. full-set. The true count is only recoverable from `n-samples` in
the raw lm-eval JSON. The sidecar is written from the *current* run configuration rather than from
the run that produced the results, so it silently misreports reused/stale evaluations. This is the
same class of failure as the earlier stale-checkpoint incident and should be treated as a bug in
`scripts/run_downstream_eval.sh`, not merely a documentation slip.

**Consequences for the paper.**

- **§7 / F1 on 1B is affected.** `best_hqq − hqq4 = +1.67 [−0.08, +3.33]` compares n=200 against
  n=200. It is internally matched, so not invalid, but it is low-powered — which is precisely why
  that interval is an order of magnitude wider than every other contrast. State the sample size
  and stop presenting it alongside full-set contrasts without qualification.
- **Any 1B comparison against `fp16` is not like-for-like** (n=200 vs full) and should not be made.
- **3B is unaffected**; the `best_hqq − hqq4 = +1.28 [+0.84, +1.68]` and
  `best_hqq − random_hqq = +0.92 [+0.50, +1.34]` contrasts used for the F1 claim are full-set.

**Required action:** either re-run the three 1B points at full scale, or annotate every affected
number with `n=200` and restrict the F1 claim to 3B. Do not average n=200 and full-set rows into
one table without marking them.

**Status 2026-07-28 — annotated, and not paper-blocking.** Checked against the draft: the paper
cites **no** 1B `fp16`, `hqq4` or `best_hqq` number. Its only 1B downstream row is
`greedy_gptq` vs `gptq4` (+0.32 [−0.07, +0.73]), which Issue 1's own audit confirms is full-set
vs full-set, and the F1 contrasts in §7 (+1.28, +0.92) are 3B. The exposure was to
`docs/DOWNSTREAM.md`, an artifact shipped with the paper, which did not mark the reduced scope.
`analysis/build_downstream_table.py` now reads the true count from each run's **own lm-eval
output** (never from `seq_meta.json`, whose `limit` field is written from the *current* config and
is what misreported these runs), prints an `n/task` column, marks any non-full point ⚠, and
appends a warning to any contrast whose two arms differ in scope or are both reduced. The
regeneration is held until Issue 4's logs are restored.

---

## Issue 2 — LAMBADA perplexity was untraceable from the bundle (now fixed)

The paper's 4.17 / 4.28 are **correct**, but the bundle excluded the raw lm-eval JSONs, so they
could not be checked. They live at
`runs/final/downstream/<model>/<point>/lm_eval/*/results_*.json` under
`results.lambada_openai["perplexity,none"]`.

Extracted for every point into **`results/lambada_and_eval_scope.csv`**, which now carries both the
LAMBADA perplexity and the evaluated sample count with a source path per row:

| model | fp16 | gptq4 | greedy_gptq | resmax | hqq4 | best_hqq | random_hqq |
|---|---|---|---|---|---|---|---|
| 3B | 3.874 | **4.282** | **4.173** | 4.226 | 4.373 | 4.030 | 4.177 |
| 1B | 5.443ᵃ | 7.113 | 6.759 | 6.271 | 7.522ᵃ | 6.006ᵃ | — |

ᵃ n=200/task (Issue 1).

**Required action:** cite the CSV in the provenance appendix; the paper's §4.1 numbers stand.


---

## Issue 3 — `docs/COMPARISON.md` predates the corrections

Last regenerated in `069c021` (2026-07-20), **before** the `--skip_lm_head` scope fix and the g128
storage fix (both landed in `0831be2`). It is therefore built from runs where `lm_head` was
quantized and group metadata was charged at g64.

| quantity | COMPARISON.md | corrected sweep |
|---|---|---|
| SEQ k=0.02 weight bits | 4.82 | **4.572** |
| 1B GPTQ-4 base PPL | 10.405 | **10.557** (reloaded, one evaluator) |

**Consequence:** §7's "the base dominates the frontier" cites this table, including the
"4.82 bits / +0.5-bit premium" framing. Those figures are from the superseded axis.

**RESOLVED 2026-07-28.** Regenerated from the corrected sweeps
(`analysis/build_comparison.py` over `runs/final/sweeps`, 73/69/67 points for 1B/3B/Qwen2.5-3B).
The table now agrees with §5.1 and §7 line for line: base `residual_max` k=0 at **4.25** bits /
**8.171**, `residual_max` k=0.02 at **4.57** bits / **8.0998**, `greedy` k=0.02 at **51.6831**, and
on HQQ at 2% `greedy` **8.1005** / `act_max` **8.1083** / `residual_rms` **8.1412** vs `random`
**8.3181**. One paper digit was corrected to match (§7 random control 8.319 → **8.318**); every
other cited value was already right. `results/final_comparison.{csv,json}` regenerated with it.

---

## Issue 4 — the per-example logs behind three paired CIs are not in the repo

**Confirmed 2026-07-28. Reproducibility gap, not a numerical error.**

`analysis/build_downstream_table.py` computes the paired bootstrap from lm-eval's
`samples_<task>_*.jsonl` files. Three points have their `results_*.json` but **no** sample logs:

| point | results_*.json | samples_*.jsonl | CI it backs |
|---|---|---|---|
| `Llama-3.2-3B/greedy_gptq` | 1 | **0** | **+0.51 [+0.13, +0.86]** — the headline |
| `Llama-3.2-1B/greedy_gptq` | 1 | **0** | +0.32 [−0.07, +0.73] — the replication |
| `Llama-3.2-3B/random_hqq`  | 2 | **0** | +0.92 [+0.50, +1.34] — §7 F1 sharpening |

Cause: commit `c79d2fb` correctly deleted the *stale* greedy_gptq lm-eval outputs (the stale-checkpoint
incident); the re-runs that replaced them (`f0a1544`, `7f3a5a7`) pushed only `results_*.json`. The
logs existed on the eval box when the CIs were computed and were never committed.

**The published numbers are sound.** Two independent checks:

1. In the committed `results/downstream.json`, each bootstrap's central estimate equals the macro
   difference of the two arms' *current* accuracies to 1e-9 (3B `+0.5074` vs `+0.5074`; 1B
   `+0.3202` vs `+0.3202`). A bootstrap run against the deleted stale samples could not agree with
   the fresh accuracies.
2. Every `per_task.n` is the full set size (1172/2376/10042/5153/1838/1267), so the CIs are
   full-set, matching Issue 1's audit.

**Consequence.** Regenerating the tables today downgrades these three lines to
`UNPAIRED approx (no sample logs)`. `docs/DOWNSTREAM.md` and `results/downstream.{json,csv}` are
therefore **deliberately left at their committed versions** until the logs are restored — do not
overwrite them with a regeneration first.

**Required action.** Re-run those three evaluations with `--log_samples` (GPU, ~1 h), confirm each
recomputed CI reproduces the published one, then regenerate. Scheduled as Job C in
`docs/CODEX_PROMPT_JOB3.md`. Until then the artifact set cannot independently reproduce the
headline interval, and an artifact reviewer would find that.

---

## Summary

| issue | severity | headline at risk? | fix |
|---|---|---|---|
| 1B n=200 on three points + wrong `limit` metadata | **high** (audit rows + a real bug) | no | re-run at full scale, or annotate and restrict F1 to 3B |
| LAMBADA untraceable | low | no | fixed: `results/lambada_and_eval_scope.csv` |
| COMPARISON.md stale | medium (§7 only) | no | **resolved** — regenerated 2026-07-28; §7 random control corrected 8.319 → 8.318 |
| per-example logs missing for 3 points | **high** (artifact reproducibility) | no — CIs verified sound | re-run those evals with `--log_samples` (Job C) |
