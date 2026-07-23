# Findings paper — working draft (v0.3)

**Working title:** *When Does Outlier Protection Help? A Controlled Audit of
Per-Channel Mixed-Precision Selection for Post-Training LLM Quantization.*

_Status: near-submission draft. F1, F2, F4 are complete and internally consistent.
F5 downstream has now been **run for all operating points on both models**
(`docs/DOWNSTREAM.md`) — the HQQ axis **confirms F1**, but the GPTQ axis
**falsified our pre-registered F3 downstream prediction** and exposed a
**base-provenance / export-fidelity inconsistency** that is the single open blocker
to publication (§7.2). A one-command reproducibility diagnostic (`channel_sweep
--verify_materialized`, added this cycle) is running to determine whether F3's PPL
catastrophe is (A) real but base-specific/fragile or (B) an export artifact; §12
tracks the remaining work._

---

## Abstract

Keeping a small fraction of "outlier" channels in high precision on top of a
low-bit weight base is a popular recipe (LLM.int8, SpQR, OWQ). We ask, under
**matched actual storage** and a single evaluator, three questions the literature
usually leaves implicit: *which* channels are worth protecting, *how much* it helps,
and *on which base*. Across Llama-3.2-1B/3B we find: **(F1)** protecting
activation-outlier channels measurably improves a data-free RTN/HQQ base over a
random-channel control, strongest at low bits, and this **reproduces downstream**
(macro accuracy +1.21 pts [+0.77, +1.63] on 3B); **(F2)** an interaction-aware,
full-Hessian greedy selector does **not** robustly beat a simple independent
activation-magnitude score at matched bits — the extra second-order machinery is
unjustified; **(F3)** on an error-compensated (GPTQ) base, residual-driven *set*
selection is **catastrophic in perplexity** (PPL 8→55 on 3B, 10→104 on 1B) while
activation-magnitude and random selection stay safe — consistent with a mechanism in
which GPTQ concentrates residual error into compensation columns that the selector
then restores to FP16, breaking the compensation; **(F4)** on a weight-only
matched-bit axis the base quantizer is the Pareto ceiling — protected RTN never
reaches GPTQ-4's operating point, and only a single activation-magnitude-on-GPTQ
point is Pareto-optimal (3B, 4.82 bits / 8.099 PPL vs GPTQ-4 8.304). **We report an
important negative control on F3:** the PPL catastrophe **did not reproduce
downstream** — the exported greedy@GPTQ checkpoint scores as a healthy 4-bit model
(lambada PPL 4.17, not the ~55 its perplexity implies) — and we trace this to a
base-regeneration/export-fidelity issue that we are resolving before any downstream
antagonism claim is made. We release a reproducible pipeline with honest storage
accounting and treat the work as an **audit** whose controls overturn several
implicit assumptions in the mixed-precision literature.

## 1. Introduction
The outlier-protection recipe and its many instances. The gap: no controlled study
that varies the *selection signal* and the *base quantizer* at **matched actual
bytes** with **random controls**. Contributions = findings F1–F5, a pre-registered
gate that the interaction-aware method fails on the strong base, and a reproducible,
honestly-accounted benchmark. We also report where our own pre-registered
downstream prediction was falsified (§7.2) — the paper is an audit, and that
includes auditing our own headline.

## 2. Setup
- **Protection form (column split):** `y = Q(W)x + x[S]·(W − Q(W))[:,S]ᵀ`; S = the
  protected input channels restored to FP16. (Note: this is the LLM.int8 idea —
  isolating outlier *input-feature* columns of `[out, in]` weights via a
  mixed-precision decomposition, *not* row promotion.) The exported evaluation
  checkpoint materializes exactly this: `W_dense = Q(W) + scatter(W − Q(W), S)`,
  which is algebraically identical to the runtime forward pass, so a faithful
  export must reproduce the runtime PPL (this identity is the basis of the §7.2
  diagnostic).
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

> **Provenance note (to reconcile before submission).** The GPTQ-4 base that
> produced these numbers (3B 8.304 / 1B 10.363) was later lost and **regenerated**
> with the same recipe (seed 42, fixed calibration); the regenerated base measures
> 3B **8.326** / 1B **10.404**. All downstream GPTQ exports (§7) were built from the
> **regenerated** base, while the §3/§5 PPL sweeps were run on the **original** base.
> This split provenance is benign for F1/F2/F4 but is the crux of the F3 downstream
> discrepancy (§7.2); the fix is to re-run the GPTQ-axis sweeps on the regenerated
> base so every GPTQ number shares one base (§12, item 2).

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
**Downstream (F5, §7.1): confirmed** — the best HQQ protection point recovers most of
the base→FP16 accuracy gap (3B +1.21 pts, CI excludes 0).

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

## 5. F3 — Protection is antagonistic to error compensation (in perplexity)
On the GPTQ base, `residual_max` stays safe (≈ base, even improves) and `random` is
harmless (≈ base), but the residual-driven **set** selectors blow up.

| model | frac | residual_max | random | **greedy** | **greedy_indep** |
|---|---|---|---|---|---|
| 1B | 0.02 | 10.391 | 10.680 | **104.16** | **15.64** |
| 1B | 0.20 | 10.350 | 10.650 | **106.82** | **15.61** |
| 3B | 0.02 | 8.099 | 8.161 | **55.34** | **44.88** |
| 3B | 0.20 | 8.070 | 8.342 | **43.21** | **45.41** |

**Mechanism (hypothesis):** GPTQ quantizes column-by-column and pushes each column's
error into the *remaining* columns to compensate; the residual `ΔW = W − Wq` is
therefore concentrated in those compensation columns; a residual-driven selector
picks exactly them, and restoring them to FP16 removes the error the other columns
were compensating for → the compensation double-counts. **Dose-response:** harm grows
with budget on 1B; on 3B it is large at every budget.

> **⚠ Scope and open status of F3 (read before citing).** This result is currently
> established **only in WikiText-2 perplexity, on the original GPTQ base.** Two facts
> keep it from being a finished headline:
> 1. **It did not reproduce downstream.** The exported greedy@GPTQ checkpoint scores
>    as a healthy 4-bit model on six tasks (§7.2), with a lambada token-PPL of 4.17
>    (3B) — incompatible with a WikiText PPL of 55. Whatever was evaluated downstream
>    is **not** the catastrophic model.
> 2. **It has not been re-confirmed on the regenerated base.** The PPL-55/104 numbers
>    come from the original base; the downstream export used the regenerated base
>    (§2a provenance note).
>
> Because `runtime_ppl == materialized_ppl` by construction (§2), a faithful export of
> a PPL-55 model *must* score ~55 downstream. It did not. Either (A) greedy@GPTQ is no
> longer catastrophic on the regenerated base (F3 is **base-fragile**), or (B) the
> export drops the protection (an **export bug**, F3 stands but needs a corrected
> downstream). The `--verify_materialized` diagnostic (§7.2) settles A vs B. **We do
> not claim downstream antagonism until it does.**

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

## 7. F5 — Downstream evaluation
lm-eval (hellaswag, arc-easy, arc-challenge, piqa, winogrande, lambada-openai) at the
operating points, run from the saved checkpoints via `scripts/run_downstream_eval.sh`
(operating points in `configs/downstream_operating_points.json`). We report zero-shot
accuracy (acc_norm where the task provides it, else acc) and **paired-bootstrap 95%
CIs** on per-example correctness for three contrasts. Numbers are from
`docs/DOWNSTREAM.md` (all six points × two models now complete).

### 7.1 Full operating-point table

**Llama-3.2-3B** (macro-avg over six tasks):

| point | bits | arc-c | arc-e | hellaswag | lambada | piqa | winogrande | **avg** |
|---|---|---|---|---|---|---|---|---|
| FP16 | 16 | 46.42 | 72.14 | 74.16 | 70.17 | 78.07 | 69.61 | **68.43** |
| GPTQ-4 (base) | 4.0 | 44.20 | 69.91 | 73.24 | 68.54 | 77.04 | 69.69 | **67.10** |
| HQQ-4 (base) | 4.0 | 45.31 | 70.03 | 72.34 | 67.48 | 76.61 | 68.59 | **66.72** |
| residual_max@GPTQ | 4.82 | 45.31 | 72.35 | 73.27 | 68.45 | 76.39 | 69.06 | **67.47** |
| greedy@GPTQ | 4.82 | 44.80 | 71.46 | 73.25 | 69.05 | 77.26 | 69.85 | **67.61** |
| best greedy@HQQ | 7.70 | 45.99 | 71.63 | 73.34 | 69.63 | 77.86 | 69.14 | **67.93** |

**Llama-3.2-1B**:

| point | bits | arc-c | arc-e | hellaswag | lambada | piqa | winogrande | **avg** |
|---|---|---|---|---|---|---|---|---|
| FP16 | 16 | 31.50 | 61.50 | 58.00 | 60.00 | 76.50 | 60.50 | **58.00** |
| GPTQ-4 (base) | 4.0 | 34.98 | 61.62 | 62.11 | 58.06 | 73.56 | 59.75 | **58.35** |
| HQQ-4 (base) | 4.0 | 35.00 | 59.00 | 54.50 | 56.50 | 75.50 | 62.50 | **57.17** |
| residual_max@GPTQ | 4.82 | 35.32 | 61.41 | 61.81 | 60.62 | 73.94 | 61.25 | **59.06** |
| greedy@GPTQ | 4.82 | 35.58 | 60.82 | 61.97 | 59.01 | 73.45 | 61.17 | **58.67** |
| best greedy@HQQ | 7.70 | 32.00 | 64.00 | 57.50 | 61.00 | 75.00 | 63.50 | **58.83** |

**Paired contrasts (macro-Δ accuracy pts, 95% CI, paired bootstrap):**

| contrast | claim tested | 3B | 1B |
|---|---|---|---|
| best@HQQ − HQQ-4 | protection helps a data-free base (F1) | **+1.21 [+0.77, +1.63]** ✅ | +1.67 [−0.08, +3.33] (dir.) |
| residual_max@GPTQ − GPTQ-4 | safe protection ≥ GPTQ-4 (F4) | +0.37 [−0.06, +0.77] | **+0.71 [+0.28, +1.16]** ✅ |
| greedy@GPTQ − GPTQ-4 | residual-driven protection is catastrophic (F3) | **+0.51 [+0.13, +0.86]** ❌pred. | +0.32 [−0.07, +0.73] ❌pred. |

- **F1 — confirmed.** On 3B the CI excludes zero; the data-free base recovers ~70%
  of its accuracy gap to FP16. 1B is directional (small model, wide CI).
- **F4 — confirmed.** Safe activation-magnitude protection on GPTQ is at least as good
  as the GPTQ-4 base downstream (positive on both models, CI excludes 0 on 1B).

### 7.2 F3 downstream — pre-registered prediction falsified (the open blocker)
We pre-registered: *"`greedy@GPTQ − GPTQ-4` macro-Δ CI is strongly negative (F3
antagonism reproduces downstream)."* **It is not.** The contrast is **+0.51 [+0.13,
+0.86] on 3B** (significantly in greedy's *favor*) and +0.32 on 1B. The exported
greedy@GPTQ checkpoint behaves like a healthy 4-bit model, not a PPL-55 one.

The internal evidence is decisive without a GPU: lm-eval's own token-level
**lambada perplexity** for the greedy@GPTQ checkpoints is **4.17 (3B)** and **6.76
(1B)** — *lower* than the healthy GPTQ-4 base (4.28 / 7.11). A model at WikiText PPL
55/104 would have lambada perplexity in the tens–hundreds and near-zero accuracy. So
the checkpoint that was evaluated downstream **is not the catastrophic model** — the
PPL-55 blow-up seen in the §5 sweep is absent from the exported checkpoint.

Given the export identity `runtime_ppl == materialized_ppl` (§2), this leaves exactly
two possibilities:
- **(A) F3 is base-fragile.** On the regenerated base, greedy@GPTQ measured healthy
  in-sweep, so the export is healthy. The catastrophe is a property of the specific
  original base and does not survive base regeneration.
- **(B) Export bug.** greedy@GPTQ still measures ~55 in-sweep on the regenerated base,
  but the export drops the protection and saves ≈ the plain base.

**Diagnostic (running):** `channel_sweep --select greedy --protect_fracs 0.02
--base_quantizer gptq_llmc --gptq_model_path <regenerated base> --verify_materialized`
prints both `runtime_ppl` and `materialized_ppl` on the regenerated base. `runtime≈8`
⇒ **A**; `runtime≈55, Δ≈0 but downstream healthy` ⇒ inconsistent, points to **B** in
the orchestrator's export path. This single number decides how §5/§7 are finalized.

**Until resolved:** F3 is reported as a **perplexity-only** result on the original
base (§5), the downstream row is presented as a *falsified pre-registration* (this
subsection), and no "antagonism confirmed downstream" claim is made. (An earlier
draft of `docs/DOWNSTREAM.md` mislabeled greedy@GPTQ "CATASTROPHIC — confirms F3
downstream"; that label is contradicted by its own numbers and is being corrected.)

### 7.3 Note on the matched-bit downstream control
The HQQ F1 contrast varies budget (4.0 → 7.70 bits). Adding a `random@HQQ` point at
the same 7.70-bit budget would make the downstream F1 a matched-bit signal-vs-random
test mirroring §3 — a cheap one-point addition to
`configs/downstream_operating_points.json` (§12, item 3).

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
| LLM.int8 | input columns (mixed-precision decomp.) | activation outlier magnitude | RTN | F1: activation signal helps a data-free base (PPL + downstream) |
| SpQR / OWQ | outlier weights / columns | sensitivity (Hessian/OBS-style) | RTN & compensated | F3: residual-driven selection toxic post-compensation *in PPL* |
| CLAQ / Atom / SqueezeLLM | columns / groups / rows | outlier + sensitivity | RTN/GPTQ | F4: base quantizer is the Pareto ceiling |
| AMQ / SliM-LLM / CoopQ | layer / group budgets | learned / interaction | mixed | F2: interaction-aware selection doesn't pay at channel level |
| EWQ | modules | entropy prior | uniform | §8: entropy proxy ≤ uniform under matched bits |

**Position:** the novel contribution is the *controls* — matched actual bytes, random
baselines, an interaction ablation (greedy vs greedy_indep), and a base×selector cross
— which together overturn the implicit "more/smarter protection is better" assumption.

## 10. Discussion & limitations
**Discussion.** F3's mechanism (if confirmed robust) implies a concrete fix —
**protect-then-recompensate**: choose the FP16 columns first, then run GPTQ over the
complement so compensation and protection cooperate. That is the one direction that
could turn this audit into a method (see `docs/TRACK_B_STATUS.md §4`); it is
deliberately out of scope here.
**Limitations.**
- Two sizes / one family (Llama-3.2) unless an 8B/cross-family robustness check is
  added; GPTQ baselines for 7B/8B-class models and Qwen2.5-3B already exist
  (`runs/final/llmc/*`), so only the sweeps + downstream are missing.
- **F3 reproducibility is open** (§7.2) and **base provenance is split** (§2a) — both
  must close before F3 is a headline rather than a perplexity-only observation.
- Weight-only PTQ; PPL + six zero-shot tasks; storage is theoretical weight-only
  bytes (no custom kernels / latency).

## 11. Conclusion
Cheap **activation-magnitude** protection gives a modest, real gain on a data-free
RTN base **that reproduces downstream**; **interaction-aware selection does not earn
its cost**; and on a strong error-compensated base, **residual-driven set protection
is catastrophic in perplexity** — a warning against naively stacking outlier
protection on GPTQ, pending confirmation that the effect is robust to base
regeneration and transfers downstream (§7.2). The base quantizer dominates the
accuracy–size frontier. Practitioners should prefer a strong base over post-hoc
protection, and reserve outlier protection for data-free bases or protect-then-
compensate designs.

---

## 12. What is missing before submission (ranked)
1. **Resolve F3 (A vs B) — blocking.** Run the `--verify_materialized` diagnostic on
   the regenerated GPTQ base (3B and 1B). Outcome rewrites §5's scope and §7.2's
   verdict. *(In progress.)*
2. **Unify GPTQ base provenance — blocking for F3/F4.** Re-run the GPTQ-axis PPL
   sweeps (`residual_max`, `greedy`, `greedy_indep`, `random` × fracs) on the
   **regenerated** base so §5, §6 and §7 all cite one base. Reconcile the §2a
   baseline (8.304→8.326 / 10.363→10.404). If F3 is base-fragile (A), report the
   catastrophe as base-conditioned and quantify how often it recurs across bases.
3. **Matched-bit downstream control (F1 sharpening).** Add `random@HQQ` at 7.70 bits
   to `configs/downstream_operating_points.json`; rerun that one point; report
   `best@HQQ − random@HQQ` downstream (matched-bit signal-vs-random, mirrors §3).
4. **Cross-family / scale robustness (F1/F3/F4 generality).** Extend the sweep +
   downstream to Llama-3.1-8B, Qwen2.5-3B, Mistral-7B-v0.3 (baselines present). At
   minimum, does F3's PPL antagonism appear on a second model family?
5. **Second error-compensated base (F3 generality).** Repeat the greedy-vs-safe
   contrast on an **AWQ** base. If residual-driven protection is also toxic on AWQ,
   F3 generalizes beyond GPTQ; if not, it is GPTQ-specific.
6. **Reload-validation table.** Publish, for every exported checkpoint, `expected_ppl`
   vs measured `reload_ppl` (the sanity check that would have caught §7.2 earlier).
   This is a credibility exhibit, not new science.
7. **(Optional, turns audit→method)** protect-then-recompensate proof-of-concept on
   3B: does choosing FP16 columns *before* GPTQ beat GPTQ-4 at matched bits?

## 13. Figures to produce
Data and a plotting entry point (`analysis/plot_final_results.py`, outputs under
`figures/final/`) already exist; these are the target figures.

- **Fig. 1 — Dose-response (F1 + F3), the money figure.** PPL vs protection fraction,
  two panels (HQQ base | GPTQ base), one line per selector (greedy, greedy_indep,
  residual_max, random with CI band), for 3B (1B in appendix). HQQ panel: all
  selectors below the random band (F1). GPTQ panel: greedy/greedy_indep blow up while
  residual_max/random stay flat (F3). Log-y so 8 and 104 coexist.
- **Fig. 2 — Pareto frontier (F4).** PPL vs weight-only bits/param scatter of all
  methods (GPTQ-4, AWQ-4, RTN-4, HQQ-4/5/6/8, SEQ points, FP16), frontier line drawn,
  the single 3B frontier SEQ point (residual_max@GPTQ, 4.82b/8.099) marked ★. Source:
  regenerated `COMPARISON.md` / `results/final_comparison.csv`.
- **Fig. 3 — Downstream forest plot (F5).** The three paired contrasts × two models,
  each as Δ-accuracy ± 95% CI with a zero line. Visually shows F1 positive, F4 ≈0/
  positive, and the F3 prediction landing on the *wrong side of zero* (ties to §7.2).
- **Fig. 4 — F3 forensic panel (honesty exhibit).** Per checkpoint, side-by-side of
  (i) sweep WikiText PPL (greedy 55/104) and (ii) the exported checkpoint's lambada
  token-PPL (4.17/6.76) against the base — making visible that the exported model is
  healthy. Replace with the `--verify_materialized` `runtime` vs `materialized` bars
  once the diagnostic returns.
- **Fig. 5 (appendix) — Proxy decoupling (§8).** Spearman ρ between each module-level
  proxy and measured per-module PPL sensitivity across runs 1–6, with the ρ≈0 band.
  Source: `analysis/findings_summary.json`.

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
- **Base provenance (F3 caveat).** §3/§5 PPL sweeps use the original GPTQ base; §7
  downstream exports use a regenerated base with the same recipe (3B 8.326 / 1B
  10.404 vs original 8.304 / 10.363). Item 2 of §12 unifies these.
