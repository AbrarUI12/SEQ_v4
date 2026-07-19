# Findings paper — working draft (v0)

**Working title:** *When Does Outlier Protection Help? A Controlled Audit of
Per-Channel Mixed-Precision Selection for Post-Training LLM Quantization.*

_Status: draft for understanding. All numbers are from the RTX-4090 final run
(`runs/final/reports/GATE_SUMMARY.md`); the Pareto claims assume the regenerated
`docs/COMPARISON.md` (see PROJECT_STATUS_AND_ROADMAP.md §3 W1)._

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

## 3. F1 — Protection helps a data-free base
On HQQ, every signal beats random at matched bits, monotone in budget: 1B @ 20%
(7.70b) greedy 10.207 vs random 10.975; 3B @ 20% 8.028 vs 8.243. From runs 4–6,
weight-magnitude selection ≈ random → **the useful signal is activation, not
weights.** [Table: HQQ selector curves, all fractions, both models.]

## 4. F2 — Interactions don't pay
greedy vs greedy_indep vs residual_max are within **~0.02–0.03 PPL** on HQQ, and
greedy_indep sometimes wins (3B @ 0.05: 8.111 vs greedy 8.112). Isolating the
iterative interaction term (greedy − greedy_indep) yields no consistent, meaningful
gain. **The full-Hessian OMP machinery is not justified over a one-shot activation-
magnitude score.** [Table: greedy − greedy_indep − residual_max deltas.]

## 5. F3 — Protection is antagonistic to error compensation (headline)
On the GPTQ base, `residual_max` stays safe (≈ base, even improves) and `random` is
harmless (≈ base), but the residual-driven **set** selectors blow up: greedy
43–107, greedy_indep 15–47 PPL. **Mechanism:** GPTQ quantizes column-by-column and
pushes each column's error into the *remaining* columns to compensate; the residual
`ΔW = W − Wq` is therefore concentrated in those compensation columns; a
residual-driven selector picks exactly them, and restoring them to FP16 removes the
error the other columns were compensating for → the compensation double-counts.
**Dose-response:** harm grows with budget. **Takeaway / practical warning:** do not
stack residual-driven protection on an error-compensated base; use activation
magnitude, or protect *before* compensation (see Discussion). [Figure: PPL vs budget
for greedy/greedy_indep/residual_max/random on GPTQ, both models.]

## 6. F4 — The base is the ceiling; honest accounting matters
On the weight-only axis: **1B — no SEQ point is Pareto-optimal** (GPTQ-4, uniform
HQQ-5/6/8, FP16 dominate). **3B — one SEQ point is on the frontier:**
`residual_max` on GPTQ, 4.82 bits / 8.099 PPL, non-dominated vs GPTQ-4 (8.304) at a
+0.5-bit premium. Nominal effective bits mislead: naive accounting charged the same
GPTQ base 7.9 bits vs 4.0 (fixed here). [Table: full matched-bit Pareto, both models.]

## 7. F5 — Downstream corroboration [pending, W2]
lm-eval (hellaswag/arc/piqa/winogrande) at the operating points to confirm the PPL
ordering and check the GPTQ-antagonism at the task level. Paired bootstrap CIs.

## 8. Related work
Outlier / mixed-precision: LLM.int8 (column mixed-precision decomposition), SpQR,
OWQ, CLAQ, Atom, SqueezeLLM. Allocation: AMQ, SliM-LLM, CoopQ (layer-granularity
interactions), LieQ. Entropy priors: EWQ. **Position:** we do not propose a new
protector; we *audit* selection signals and base×protection interaction with
controls (matched actual bytes, random baselines, an interaction ablation) that
prior work omits.

## 9. Limitations
Two sizes / one family (Llama-3.2) unless W4 adds 8B/cross-family; weight-only PTQ;
PPL + five tasks; storage is theoretical weight-only bytes (no custom kernels /
latency).

## 10. Conclusion
Cheap **activation-magnitude** protection gives a modest, real gain on a data-free
RTN base; **interaction-aware selection does not earn its cost**; and **protection
must not be naively combined with error compensation** — on a GPTQ base it is
catastrophic unless applied before compensation. The base quantizer dominates the
accuracy–size frontier. Practitioners should prefer a strong base over post-hoc
protection, and reserve outlier protection for data-free bases or protect-then-
compensate designs.
