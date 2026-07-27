# Verification gaps — evidence and required paper changes

Three issues found when checking the draft against the result artifacts. All three are confirmed;
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

**Required action:** regenerate COMPARISON.md from the corrected sweeps before the claim is
restated, or restrict §7 to numbers taken directly from the corrected `channel_pareto.json` files
(the storage-axis figures already quoted in §5.1 — 4.25 bits base, 4.57 bits at 2% — are correct).
Until regenerated, treat COMPARISON.md as **superseded** and do not cite it.

---

## Summary

| issue | severity | headline at risk? | fix |
|---|---|---|---|
| 1B n=200 on three points + wrong `limit` metadata | **high** (audit rows + a real bug) | no | re-run at full scale, or annotate and restrict F1 to 3B |
| LAMBADA untraceable | low | no | fixed: `results/lambada_and_eval_scope.csv` |
| COMPARISON.md stale | medium (§7 only) | no | regenerate, or cite corrected sweeps directly |
