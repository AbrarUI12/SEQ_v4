# Response to the external code audit (2026-07-28)

An external reviewer audited the code behind *When Perplexity Lies*. This document records, for
each finding: whether we could verify it, the evidence, which paper claim it touches, and what we
changed. **All three findings that concern this repository are correct**, and the critical one is
our own bug.

**Scope.** The audit also inspected a curated repository at `D:/0 Repositories/perplexity-paradox`
(findings 5–9: downstream aggregator lineage, manifest revisions, hard-coded validation, command
scope). That repository is not ours — our curated release is `protection-paradox` — so those
findings are **not assessed here**. Several of them describe good practice we should adopt
regardless, and are listed at the end as accepted-in-principle.

---

## Finding 1 — `greedy_gain` did not measure the greedy objective

**Status: CONFIRMED, FIXED, AND RE-RUN. Critical. Our error.**

`scripts/measure_objective_alignment.py` scored the greedy selector as
`residual_energy × hessian_diag` = `‖ΔW_j‖²·H_jj`, and its docstring asserted this was "exactly
what greedy/greedy_indep rank by". The selector's actual first-step gain
(`seq_core/greedy_select.py`) is

```
G_j = 2⟨ΔW_j, (ΔW H)_j⟩ − ‖ΔW_j‖² H_jj
```

Our expression is only the **subtracted** term; the off-diagonal coupling term is absent.

**Evidence.** On correlated (realistic) Hessians, out=64, in=256, 20 trials: the top-ranked
channel differs in **16/20** trials, median Spearman(true, proxy) = **+0.41**. With a *diagonal*
Hessian the two coincide exactly — which isolates the cross-term as the difference. Both facts are
now pinned by `tests/test_first_step_gains.py`.

**Affected claim.** §6's conclusion that the compensation account is falsified. The reported
ordering (`residual_rms` +0.99 > `greedy_gain` +0.39 > `hessian_diag` −0.13) is explained by the
proxy literally being the product of the other two, not by anything about greedy.

**Change.** `first_step_gains()` added to `greedy_select.py` as the single definition, shared with
the selector via `_marginal_gains()`. It is computed **in-pass** in `gptq.py` while the (identically
damped) Hessian is resident, so no full-Hessian storage is needed. The probe now reports the true
`greedy_gain` and retains the old expression as `diag_proxy` so the published number stays
auditable. §6 has been re-run and restored on the corrected quantity (below).

**Not affected.** The selector itself is correctly implemented (the auditor agrees), so every
perplexity, the collapse, the downstream results and the per-token decomposition stand.

**Corrected result (56 layers/model).** Scored by the true objective, the harmful selector is
**anti**-correlated with compensation magnitude: ρ = **−0.230** (3B) / **−0.142** (1B), positive in
only **29%** of layers, against `residual_rms` at **+0.988 / +0.981** in **100%** of layers. The
superseded proxy reproduces at +0.388 / +0.384, confirming the diagnosis that the published column
was the proxy. **The refutation therefore stands and is stronger on the right quantity** — the
harmful selector systematically *avoids* the columns the compensation account requires it to
target. §6 has been restored on this basis; nothing is withdrawn.

---

## Finding 2 — "the same 128×2048 tokens as the base" is false

**Status: CONFIRMED. High.**

**Evidence.** Base: `runs/final/llmc/Llama-3.2-3B/gptq/summary.json` records
`preproc=wikitext2_gptq, seed=0`. Selector: `channel_sweep.py` calls
`build_gptq_calibration(..., seed=args.seed)` with seed 1234 and a different sampler. Same corpus
and token budget; different windows and preprocessing.

**Affected claim.** §5.3's wording and its causal reading. The measurements (3B 58.24, 1B 11.39)
are unaffected.

**Change.** §5.3 now says the comparison is **size-matched, not sample-matched**, and states
explicitly that it bounds estimator *quality* and does not eliminate calibration-sample mismatch
as a confound. A genuine sample-matched test requires persisting the base's calibration token ids;
that is listed as outstanding work.

---

## Finding 3/4 — scalar-signal calibration is padding-dominated

**Status: CONFIRMED, DISCLOSED, AND ROBUSTNESS-TESTED. High.**

**Evidence.** `seq_core/signals.py` tokenized with `padding="max_length", max_length=seq_len`, and
the hooks accumulate every position. With ~51 short prompts padded to 2048, the activation
statistics behind `act_max`, `act_scale`, `residual_max`, `residual_rms` are dominated by pad/EOS
states. `seq_core/gptq.py` explicitly avoids padding for the Hessian path for exactly this reason
("would make H ~97% pad-token statistics"), so the two calibration paths were inconsistent.

**Affected claim.** §5.1's description of the scalar selectors as "informed", and the HQQ result
in §7 where `act_max` beats random by ~0.2 PPL despite pad-dominated statistics.

**Change.** Disclosed in §3 (Setup) and Limitations. A `--no_pad_calibration` flag now threads
through `signals.py` → `channel_sweep.py`, and the run records `pad_calibration` in its output so
every artifact is self-describing. The published default (padding on) is preserved so existing
numbers remain reproducible. The robustness re-measurement is reported below.

**Robustness result (Llama-3.2-3B).** Re-measured on real tokens only: `act_max` 8.181→8.178,
`residual_max` 8.100→8.100, `residual_rms` 8.146→8.146 are unmoved; only `act_scale` shifts
(8.586→8.142), so its apparent weakness was a padding artifact. Under honest calibration all four
scalars lie within 8.10–8.15 against a base of 8.172, and on HQQ the informed selectors
(8.100–8.141) still beat random (8.319). **The §5.1 conclusion and the §7 HQQ claim both survive**,
and §5.1 is cleaner without the `act_scale` outlier. Not yet repeated on 1B.

---

## Findings 5–9 — not assessed (different repository)

These concern a curated repo we did not produce. We accept the following in principle and will
apply them to `protection-paradox`:

- Downstream outputs should be atomic run bundles (checkpoint digest, run id, model revision, task
  scope, results and per-example samples together), rejecting mixed timestamps and partial reports.
- Sample alignment should key on `doc_id`, not `doc_hash` (LAMBADA has 5,153 rows but 5,151 unique
  hashes, so hashing silently drops two examples).
- Model revisions should be enforced at GPTQ preparation and at evaluation, not merely declared.
- Expected base and protected perplexities should be **mandatory gates** before a downstream run,
  so a wrong base cannot complete a sweep unnoticed.
- Validation should recompute from raw outputs rather than compare hard-coded summaries.

The auditor's observation that passing tests can give false assurance is well taken: our smoke
pilots exercise entry points on a tiny model and deliberately do **not** validate science. That is
stated in the curated repo's README and `KNOWN_DEVIATIONS.md`, and this response adds golden tests
that would have caught Finding 1.

---

## Status summary

| # | Finding | Verified | Severity | Headline at risk | Fixed |
|---|---|---|---|---|---|
| 1 | `greedy_gain` is a diagonal proxy | yes | critical | no | **resolved** — fixed, re-run, refutation stronger (ρ = −0.23/−0.14) |
| 2 | "same tokens" is false | yes | high | no | wording corrected; `--selector_calib_tokens` added for future token-identity; sample-sensitivity test scheduled |
| 3/4 | padded scalar calibration | yes | high | no | **resolved on 3B** — ordering unchanged; only `act_scale` was an artifact |
| 5–9 | curated-repo lineage/manifest | not assessed | — | no | accepted in principle |

**The headline is unaffected.** Sweep 51.68 → reload 51.76 → macro 67.61% vs 67.10%, +0.51
[+0.13, +0.86]; the per-token decomposition; the identity chain; the 1B replication. The errors
were in an analysis probe and in descriptions of calibration, not in the selector, the sweeps or
the evaluation.
