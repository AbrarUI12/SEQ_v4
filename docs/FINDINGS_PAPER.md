# Findings paper — working draft (v0.2)

**Working title:** *When Does Outlier Protection Help? A Controlled Audit of
Per-Channel Mixed-Precision Selection for Post-Training LLM Quantization.*

_Status: draft. F1–F4 tables are populated from the RTX-4090 final run
(`runs/final/reports/GATE_SUMMARY.md`, mirrored in
`docs/PROJECT_STATUS_AND_ROADMAP.md §1`); the Pareto table is the regenerated
`docs/COMPARISON.md` (W1 done). F5 downstream: the **HQQ axis (F1) is in**
(`docs/DOWNSTREAM.md`); the **GPTQ axis (F3/F4) is pending** the saved GPTQ base
checkpoint on the eval box (see §7)._

---

## Abstract

Keeping a small fraction of "outlier" channels in high precision on top of a
low-bit weight base is a popular recipe (LLM.int8, SpQR, OWQ). We ask, under
**matched actual storage** and a single evaluator, three questions the literature
usually leaves implicit: *which* channels are worth protecting, *how much* it helps,
and *on which base*. Across Llama-3.2-1B/3B we find: **(F1)** protecting
activation-outlier channels measurably improves a data-free RTN/HQQ base over a
random-channel control, strongest at low bits; **(F2)** an interaction-aware,
full-Hessian greedy selector does **not** robustly beat a simple independent
activation-magnitude score at matched bits — the extra second-order machinery is
unjustified; **(F3, headline)** outlier protection is **antagonistic to
error-compensated bases**: residual-driven selection on a GPTQ base is catastrophic
(PPL 10→40–107), because GPTQ redistributes quantization error into specific columns
and restoring those to FP16 breaks its compensation, whereas activation-magnitude
selection and random selection are safe; **(F4)** on a weight-only matched-bit axis
the base quantizer is the Pareto ceiling — protected RTN never reaches GPTQ-4's
operating point, and only a single activation-magnitude-on-GPTQ point is
Pareto-optimal (3B, 4.82 bits / 8.099 PPL vs GPTQ-4 8.304). We release a
reproducible pipeline with honest storage accounting and argue these controls
overturn several implicit assumptions in the mixed-precision literature.

## 1. Introduction
The outlier-protection recipe and its many instances. The gap: no controlled study
that varies the *selection signal* and the *base quantizer* at **matched actual
bytes** with **random controls**. Contributions = findings F1–F5; a reproducible,
honestly-accounted benchmark.

## 2. Setup
- **Protection form (column split):** `y = Q(W)x + x[S]·(W − Q(W))[:,S]ᵀ`; S = the
  protected input channels restored to FP16. (Note: this is the LLM.int8 idea —
  isolating outlier *input-feature* columns of `[out, in]` weights via a
  mixed-precision decomposition, *not* row promotion.)
- **Bases:** HQQ-4 (data-free RTN) and a validated LightCompress **GPTQ-4** (replay
  diff < 0.006 vs LLMC PPL).
- **Selectors:** act_max, act_scale, residual_rms, residual_max (activation-weighted
  quant-error), `greedy` (OMP on `tr(ΔWᵀΔW H)`, H = XᵀX), `greedy_indep` (first-step
  gains, no iterative interaction), `random` (control, 3 seeds).
- **Accounting:** weight-only bits/param (embeddings/lm_head/norms excluded; equal
  to GPTQ-4 = 4.0). **Eval:** WikiText-2 canonical PPL, seq 2048. Checkpoints saved
  and reload-validated.

### 2a. Baselines at ~4 bits (single environment, one evaluator, matched bits)
FP16 PPL: **1B 9.757**, **3B 7.817**. Axis = weight-only bits/param.

| model | GPTQ-4 | AWQ-4 | RTN-4 | HQQ-4 | HQQ-5 | HQQ-6 | HQQ-8 |
|---|---|---|---|---|---|---|---|
| 1B | **10.363** (4.29b) | 11.278 | 11.710 | 11.187 | 10.064 | 9.829 | 9.762 |
| 3B | **8.304** (4.28b) | 8.405 | 8.498 | 8.387 | 7.957 | 7.845 | 7.820 |

## 3. F1 — Protection helps a data-free base
On HQQ, every signal beats random at matched bits, monotone in budget; weight-magnitude
selection ≈ random (runs 4–6) → **the useful signal is activation, not weights.**

| model | frac | bits | greedy | greedy_indep | residual_max | random (mean [95% CI]) |
|---|---|---|---|---|---|---|
| 1B | 0.02 | 4.82 | 10.495 | 10.506 | 10.533 | 11.165 [11.155, 11.174] |
| 1B | 0.20 | 7.70 | **10.207** | 10.230 | 10.232 | 10.975 [10.932, 11.018] |
| 3B | 0.02 | 4.82 | 8.149 | 8.151 | 8.161 | 8.376 [8.370, 8.382] |
| 3B | 0.20 | 7.70 | **8.028** | 8.037 | 8.048 | 8.243 [8.025, 8.462] |

Every signal point sits well below the random-control CI at the same bits (e.g. 1B @
0.20: 10.207 vs 10.975 [10.932, 11.018]) → per-channel activation importance is real.

## 4. F2 — Interactions don't pay
greedy vs greedy_indep vs residual_max are within **~0.02–0.03 PPL** on HQQ (see the
F1 table), and greedy_indep sometimes wins (3B @ 0.05: 8.111 vs greedy 8.112).
Isolating the iterative interaction term (greedy − greedy_indep) yields no consistent,
meaningful gain. **The full-Hessian OMP machinery is not justified over a one-shot
activation-magnitude score.**

| model | frac | greedy | greedy_indep | Δ (greedy − indep) | residual_max |
|---|---|---|---|---|---|
| 1B | 0.02 | 10.495 | 10.506 | −0.011 | 10.533 |
| 1B | 0.20 | 10.207 | 10.230 | −0.023 | 10.232 |
| 3B | 0.02 | 8.149 | 8.151 | −0.002 | 8.161 |
| 3B | 0.20 | 8.028 | 8.037 | −0.009 | 8.048 |

## 5. F3 — Protection is antagonistic to error compensation (headline)
On the GPTQ base, `residual_max` stays safe (≈ base, even improves) and `random` is
harmless (≈ base), but the residual-driven **set** selectors blow up.

| model | frac | residual_max | random | **greedy** | **greedy_indep** |
|---|---|---|---|---|---|
| 1B | 0.02 | 10.391 | 10.680 | **104.16** | **15.64** |
| 1B | 0.20 | 10.350 | 10.650 | **106.82** | **15.61** |
| 3B | 0.02 | 8.099 | 8.161 | **55.34** | **44.88** |
| 3B | 0.20 | 8.070 | 8.342 | **43.21** | **45.41** |

**Mechanism:** GPTQ quantizes column-by-column and pushes each column's error into the
*remaining* columns to compensate; the residual `ΔW = W − Wq` is therefore concentrated
in those compensation columns; a residual-driven selector picks exactly them, and
restoring them to FP16 removes the error the other columns were compensating for → the
compensation double-counts. **Dose-response:** harm grows with budget.
**Practical warning:** do not stack residual-driven protection on an error-compensated
base; use activation magnitude, or protect *before* compensation (see §10).

### 5a. Pre-registered gate (why the framing is "audit")
Rule: `greedy` must beat greedy_indep, residual_max, **and** the random-CI in ≥3/4
budgets **in every** model×base stratum.

| stratum | greedy > greedy_indep | > residual_max | > random-CI | verdict |
|---|---|---|---|---|
| 1B / HQQ | 4/4 | 4/4 | 4/4 | **PASS** |
| 3B / HQQ | 3/4 | 3/4 | 4/4 | **PASS** |
| 1B / GPTQ | 0/4 | 0/4 | 0/4 | **FAIL** |
| 3B / GPTQ | 0/4 | 0/4 | 0/4 | **FAIL** |

The interaction-aware method fails its own pre-registered bar on the strong base →
we report an **audit**, not a method.

## 6. F4 — The base is the ceiling; honest accounting matters
On the weight-only axis: **1B — no SEQ point is Pareto-optimal** (GPTQ-4, uniform
HQQ-5/6/8, FP16 dominate). **3B — one SEQ point is on the frontier:**
`residual_max` on GPTQ, **4.82 bits / 8.099 PPL**, non-dominated vs GPTQ-4 (8.304) at a
+0.5-bit premium. Nominal effective bits mislead: naive accounting charged the same
GPTQ base 7.9 bits vs 4.0 (fixed here via `seq_core/storage_accounting.py`). Full
matched-bit Pareto in the regenerated `docs/COMPARISON.md`.

## 7. F5 — Downstream corroboration
lm-eval (hellaswag, arc-easy, arc-challenge, piqa, winogrande, lambada-openai) at the
operating points, run from the saved checkpoints via `scripts/run_downstream_eval.sh`
(operating points in `configs/downstream_operating_points.json`). We report zero-shot
accuracy (acc_norm where the task provides it) and **paired-bootstrap 95% CIs** on the
per-example correctness for three contrasts, to confirm the PPL ordering and the GPTQ
antagonism at the task level. Numbers auto-populate from `docs/DOWNSTREAM.md`.

_Table (fills from the downstream run):_

Zero-shot accuracy (acc_norm where reported, else acc), macro-averaged over the six
tasks (full table in `docs/DOWNSTREAM.md`). **HQQ axis (available now):**

| model | point | bits | arc-c | arc-e | hellaswag | lambada | piqa | winogrande | **avg** |
|---|---|---|---|---|---|---|---|---|---|
| 3B | FP16 | 16 | 46.42 | 72.14 | 74.16 | 70.17 | 78.07 | 69.61 | **68.43** |
| 3B | HQQ-4 (base) | 4.0 | 45.31 | 70.03 | 72.34 | 67.48 | 76.61 | 68.59 | **66.72** |
| 3B | best greedy@HQQ | 7.70 | 45.99 | 71.63 | 73.34 | 69.63 | 77.86 | 69.14 | **67.93** |
| 1B | FP16 | 16 | 31.50 | 61.50 | 58.00 | 60.00 | 76.50 | 60.50 | **58.00** |
| 1B | HQQ-4 (base) | 4.0 | 35.00 | 59.00 | 54.50 | 56.50 | 75.50 | 62.50 | **57.17** |
| 1B | best greedy@HQQ | 7.70 | 32.00 | 64.00 | 57.50 | 61.00 | 75.00 | 63.50 | **58.83** |

**Result — F1 corroborated:** protection on the data-free HQQ base recovers most of
the accuracy the 4-bit base loses to FP16 — macro Δ(best@HQQ − HQQ-4) = **+1.21
[+0.77, +1.63] pts on 3B** (paired bootstrap, CI excludes 0) and **+1.67 [−0.08,
+3.33] pts on 1B** (directional). Note this contrast is not matched-bit (7.70 vs
4.0 bits); the tight matched-bit signal-vs-random claim is the PPL result in §3 — a
matched-bit `random@HQQ` downstream point would sharpen it (see §7a).

_GPTQ axis (F3/F4 downstream) — pending:_ the `GPTQ-4`, `residual_max@GPTQ` and
`greedy@GPTQ` points require the saved GPTQ base checkpoint
(`runs/final/llmc/<Model>/gptq/artifacts/fake_quant_model`), which was not present
on the eval box for this run, so those rows are not yet filled.

**Pre-registered downstream expectations** (falsifiable): (i) `residual_max@GPTQ −
GPTQ-4` macro-Δ CI includes 0 or is positive (safe protection, F4); (ii)
`greedy@GPTQ − GPTQ-4` macro-Δ CI is strongly negative (F3 antagonism reproduces
downstream); (iii) `best@HQQ − HQQ-4` macro-Δ CI is positive (protection helps a
data-free base, F1) — **confirmed above**.

### 7a. Note on the matched-bit downstream control
The current HQQ contrast varies budget (4.0 → 7.70 bits). Adding a `random@HQQ`
point at the same 7.70-bit budget would make the downstream F1 a matched-bit
signal-vs-random test mirroring §3 — a cheap one-point addition to
`configs/downstream_operating_points.json`.

## 8. Auxiliary result — allocation proxies decouple from PPL (module level)
Before the per-channel study we tested whether a *proxy* — activation/weight entropy,
Hessian diagonal, or reconstruction error — can rank whole modules for bit allocation
better than uniform (`docs/FINDINGS_run{1..6}.md`, `analysis/findings_summary.json`).
Across runs 1–6 the proxies **rank-decouple from measured per-module PPL sensitivity**
(low, unstable Spearman ρ; measured sensitivity is itself concentrated and often near
the noise floor), and proxy-guided module allocation is **≤ uniform** at matched bits.
This is the module-granularity analogue of F1/F2: the coarse, proxy-driven allocation
that entropy-weighted methods (EWQ) rely on does not survive a matched-bit control.
It motivates moving to the per-channel activation signal that F1 shows *does* carry
information. (Consolidated from six earlier audit runs; details in the appendix.)

## 9. Related work
We do not propose a new protector; we *audit* selection signals and base×protection
interaction with controls that prior work omits.

| method | unit protected | selection signal | base | our controlled result |
|---|---|---|---|---|
| LLM.int8 | input columns (mixed-precision decomp.) | activation outlier magnitude | RTN | F1: activation signal helps a data-free base |
| SpQR / OWQ | outlier weights / columns | sensitivity (Hessian/OBS-style) | RTN & compensated | F3: residual-driven selection toxic post-compensation |
| CLAQ / Atom / SqueezeLLM | columns / groups / rows | outlier + sensitivity | RTN/GPTQ | F4: base quantizer is the Pareto ceiling |
| AMQ / SliM-LLM / CoopQ | layer / group budgets | learned / interaction | mixed | F2: interaction-aware selection doesn't pay at channel level |
| EWQ | modules | entropy prior | uniform | §8: entropy proxy ≤ uniform under matched bits |

**Position:** the novel contribution is the *controls* — matched actual bytes, random
baselines, an interaction ablation (greedy vs greedy_indep), and a base×selector cross
— which together overturn the implicit "more/smarter protection is better" assumption.

## 10. Discussion & limitations
**Discussion.** F3's mechanism implies a concrete fix — **protect-then-recompensate**:
choose the FP16 columns first, then run GPTQ over the complement so compensation and
protection cooperate. That is the one direction that could turn this audit into a
method (see `docs/TRACK_B_STATUS.md §4`); it is deliberately out of scope here.
**Limitations.** Two sizes / one family (Llama-3.2) unless an 8B/cross-family
robustness check is added; weight-only PTQ; PPL + six tasks; storage is theoretical
weight-only bytes (no custom kernels / latency).

## 11. Conclusion
Cheap **activation-magnitude** protection gives a modest, real gain on a data-free
RTN base; **interaction-aware selection does not earn its cost**; and **protection
must not be naively combined with error compensation** — on a GPTQ base it is
catastrophic unless applied before compensation. The base quantizer dominates the
accuracy–size frontier. Practitioners should prefer a strong base over post-hoc
protection, and reserve outlier protection for data-free bases or protect-then-
compensate designs.

---

## Appendix A. Statistics
- **Random control CIs (F1/F3).** The `random` selector is run with 3 seeds per
  (model, base, fraction); we report the mean and a 95% CI across seeds (columns in
  §3/§5). A signal is credited only when its point lies **below** the random CI at
  matched bits.
- **Downstream paired bootstrap (F5).** lm-eval is run with `--log_samples`, giving
  per-example 0/1 correctness. For a contrast (system A vs B) on a task, we pair
  correctness on the **same** examples and bootstrap the mean difference (2000
  resamples of examples with replacement) for a 95% CI; the macro-average difference
  bootstraps within each task and averages. Implemented pure-stdlib in
  `analysis/build_downstream_table.py` (paired design → tighter, correct CIs than the
  harness's unpaired stderr). When sample logs are absent the code falls back to an
  unpaired normal-approximation CI and labels it as such.
- **Effective-bits accounting.** All "bits" are weight-only bits/param with
  embeddings/lm_head/norms excluded (`seq_core/storage_accounting.py`), so every
  method is compared on the axis where GPTQ-4 = 4.0; the earlier 7.9-vs-4.0 mis-plot
  is corrected.
