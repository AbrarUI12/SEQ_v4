# When Outlier Protection Backfires: A Controlled Audit of Post-Hoc Mixed-Precision Channel Selection on Error-Compensated LLM Quantization

_Draft v1.0 (2026-07-26) — ACL long-paper format. All perplexity numbers in this draft come
from the corrected sweep matrix (`skip_lm_head=true`, base group size 128), traceable to
committed `runs/final/sweeps/**/channel_pareto.json`. Two items are explicitly marked
**pending**: the corrected downstream evaluation (§8) and the causal mechanism test (§9)._

---

## Abstract

Keeping a small fraction of "outlier" input channels in high precision on top of a low-bit
weight base is a widely used post-training quantization recipe. We audit it under matched
theoretical weight-only storage with random controls and a single evaluator, varying the
selection signal, the base quantizer, and the model. On a data-free RTN/HQQ base, cheap
activation-magnitude protection gives a small but real gain over a matched-bit random control
(Llama-3.2-3B: 8.11 vs 8.32 perplexity at 2% protection). On an error-compensated GPTQ base
the picture inverts: protection is at best neutral, and *residual-driven set selection applied
post hoc* is catastrophic — perplexity rises from 8.17 to 38–55 on Llama-3.2-3B and from 10.56
to 64–78 on Llama-3.2-1B — while activation-magnitude and random selection of the *same* number
of channels remain safe (~8.1–8.5). The failure requires both a residual-driven selection rule
and post-hoc restoration, is faithful through export and reload, and is model-dependent: it does
not appear on Llama-2-7B or Qwen2.5-3B. We further find that interaction-aware iterative
selection does not outperform one-shot full-Hessian selection, and that the base quantizer
dominates the accuracy–size frontier. Post-hoc outlier protection must not be assumed safe on a
compensated base.

_(≤200 words)_

---

## 1 Introduction

Post-training quantization (PTQ) of large language models routinely combines a low-bit weight
base with a small high-precision "escape hatch": a few input channels, columns, or weights kept
in FP16 because they carry outlier activations. LLM.int8() isolates outlier feature dimensions;
OWQ and SpQR retain sensitivity-selected columns or weights at higher precision; CMPQ assigns
per-channel precision from activation distributions. The recipe is intuitive and widely adopted.

What is rarely tested is whether the recipe *composes*. Two design choices are usually made
implicitly: **which signal** selects the protected channels, and **which base** the protection
sits on. A third — **when** protection is applied relative to error compensation — is almost
never varied, because most methods integrate the two by construction.

We run that audit. We hold storage fixed (matched theoretical weight-only bits per parameter),
include a random-selection control at every budget, use one evaluator for every number, and vary
the selection signal (activation magnitude, activation scale, quantization-residual magnitude,
residual RMS, random, and two Hessian-based set selectors), the base (data-free HQQ-4 vs an
error-compensated GPTQ-4), the protection budget (2–20% of input channels), and the model
(Llama-3.2-1B/3B, Llama-2-7B, Qwen2.5-3B).

The headline result is a failure mode. On a compensated GPTQ base, selecting channels by
quantization *residual* using a Hessian-driven set objective and then restoring them to FP16
**post hoc** does not merely fail to help — it destroys the model, raising perplexity by
30–68 points, while random selection of the same number of channels at the same storage is
harmless. To our knowledge this specific composition failure has not been reported. It is not
an artifact: it survives base regeneration, it reproduces under a faithful export/reload
round-trip, and it is stable across protection budgets. It is, however, **model-dependent** —
absent on Llama-2-7B and Qwen2.5-3B — which is itself informative and which we do not yet fully
explain.

**Contributions.**
1. A controlled, matched-storage audit of outlier-channel protection across selection signal ×
   base quantizer × budget × model, with random controls and a single evaluator (§3–§7).
2. The main finding: **post-hoc residual-driven set protection can catastrophically damage an
   already error-compensated base**, while magnitude/random protection at identical storage is
   safe; the effect is model-dependent (§5).
3. Two supporting negative results: interaction-aware iterative selection does not earn its
   cost where protection helps (§6), and the base quantizer, not the protection, sets the
   accuracy–size frontier (§7).
4. A mechanism hypothesis with the falsification test that would confirm it, reported honestly
   as open (§9), and a reproducible pipeline with explicit storage accounting.

**What is not novel.** Protecting activation-outlier channels is not new (LLM.int8, CMPQ);
Hessian/salience-driven column selection is not new (OWQ, SpQR); and *integrating* protection
into compensation — selecting columns before GPTQ and compensating over the complement — is
exactly OWQ's design, not ours. Our contribution is the controlled demonstration that the
*post-hoc* composition, which the literature does not endorse but which is the natural thing to
try when a compensated checkpoint already exists, is unsafe and model-dependent.

## 2 Related work

**Outlier-aware mixed precision.** LLM.int8() decomposes matrix multiplication so that outlier
feature dimensions are computed in FP16. OWQ selects sensitive input columns with a
Hessian-weighted quantization-error score, keeps them in FP16, and — crucially — *orders them
last inside the OPTQ/GPTQ pass* so that compensation happens with knowledge of which columns
stay exact. SpQR isolates outlier weights in a sparse high-precision structure. Atom, CLAQ and
SqueezeLLM apply related group/column/row schemes. CMPQ assigns per-channel precision from
activation distributions. All of these *co-design* protection with the base; none reports what
happens when protection is applied to a finished compensated base.

**Bit allocation and interactions.** SliM-LLM and CoopQ allocate bit budgets by
salience/interaction structure; AMQ searches allocations under memory constraints; EWQ uses an
entropy prior over layers. These motivate our F2 ablation (does modelling interaction between
protected channels pay?) and our matched-bit random controls, which some of this literature
includes (SliM-LLM) and some does not.

**Compensation error.** Recent work analyses residual/compensation error in GPTQ-style methods
and its accumulation. Our finding is complementary and empirical: it identifies a composition
that *breaks* compensation rather than improving its numerics.

**Positioning.** We do not propose a new protector. We audit the recipe's composition and report
a failure mode plus two negative controls. The closest prior work is OWQ; our post-hoc arm is
precisely the configuration OWQ's design avoids, and our results give an empirical reason why
that design choice matters more than it may appear.

## 3 Method

**Protection form.** For a linear layer with weight `W ∈ R^{out×in}` and protected input-channel
set `S`, the protected forward pass is

```
y = Q(W)x + x[S] · (W − Q(W))[:, S]ᵀ
```

i.e. the base quantizer `Q` is applied to all columns and the protected columns are corrected
back to full precision. Materialising this gives the dense weight
`W_dense = Q(W) + scatter(W − Q(W), S)`, which is algebraically identical to the forward pass —
so a faithful export must reproduce the runtime perplexity exactly. We use this identity as an
integrity check throughout (§5.3).

Note this is *post-hoc* protection: `Q` has already been computed (including, for GPTQ, its error
compensation) when `S` is restored. The alternative — choosing `S` first and compensating over
the complement — is OWQ's design and is the causal comparison we discuss in §9.

**Bases.** (i) **HQQ-4**, a data-free round-to-nearest base; (ii) **GPTQ-4**, an
error-compensated base produced by LightCompress at W4 group size 128 with a fixed calibration
set. The GPTQ base is imported as fake-quantized weights and re-evaluated inside our evaluator.

**Selectors.** Per layer, given the base residual `ΔW = W − Q(W)` and input Hessian `H = XᵀX`
from calibration data:
- `act_max`, `act_scale` — activation-magnitude statistics per input channel (data-driven, cheap).
- `residual_max`, `residual_rms` — magnitude statistics of `ΔW` per column.
- `random` — uniformly random channels (control), 3 seeds.
- `greedy` — an OMP-style **iterative** set selector maximising residual-energy reduction
  `tr(ΔWᵀΔW H)`, re-scoring after each pick.
- `greedy_indep` — the **one-shot full-Hessian** ablation: the same marginal-gain objective
  evaluated once, with no iterative re-scoring. (It is *not* an activation-magnitude score; it
  uses `ΔW`, `H` and `ΔW H`. This distinction matters for §6.)

**Storage accounting.** We report **matched theoretical weight-only storage**: bits per
quantized-linear parameter, counting the low-bit values, group scales/zeros at *the base's true
group size* (128 for the imported GPTQ base), the FP16 correction columns, and the channel index
overhead. Embeddings, `lm_head` and norms are FP16 in every method and excluded from the axis;
SEQ runs therefore exclude `lm_head` from quantization so that the quantized scope matches the
GPTQ baseline exactly. These are theoretical weight bytes: no kernel, runtime-metadata or
latency effects are modelled.

**Evaluation.** WikiText-2 perplexity, sequence length 2048, canonical non-overlapping chunking,
one evaluator for every number in this paper. Where a checkpoint is exported, the *reloaded*
checkpoint is what we evaluate. We note that the imported GPTQ base evaluated in our evaluator
differs from the LightCompress-reported value by 0.19 PPL on Llama-3.2-1B (10.557 vs 10.363);
we therefore use the reloaded `k=0` checkpoint as the operational base for every comparison
rather than the externally reported number, so all contrasts are within one evaluator.

## 4 Protection helps a data-free base

On HQQ-4, every informative signal beats the matched-bit random control, and the gain grows with
budget. Llama-3.2-3B (FP16 7.817, HQQ-4 base 8.328) and Llama-3.2-1B (FP16 9.757, HQQ-4 base
11.072), perplexity at 2% and 20% protection:

| model | budget | greedy | greedy_indep | act_max | act_scale | residual_rms | **random** |
|---|---|---|---|---|---|---|---|
| 3B | 2% | **8.100** | — | 8.108 | 8.146 | 8.141 | 8.319 |
| 3B | 20% | **7.994** | — | 8.007 | 8.046 | 8.040 | 8.247 |
| 1B | 2% | **10.398** | — | 10.434 | 10.521 | 10.512 | 11.051 |
| 1B | 20% | **10.128** | — | 10.158 | 10.309 | 10.308 | 10.907 |

Every signal sits clearly below random at the same storage, so per-channel activation importance
carries real information on a data-free base. The margin over random (~0.2 PPL at 3B, ~0.65 at
1B) is modest but consistent, and the ordering of signals is stable across budgets.

## 5 Post-hoc residual-driven protection breaks a compensated base

### 5.1 The failure

On the GPTQ-4 base the same protection machinery, at the same storage, produces a qualitatively
different outcome. Llama-3.2-3B (FP16 7.817, GPTQ-4 base 8.172 at 4.25 bits):

| selector | 2% | 5% | 10% | 20% |
|---|---|---|---|---|
| **greedy** (iterative) | **51.68** | **55.05** | **38.39** | **40.65** |
| **greedy_indep** (one-shot full-Hessian) | **41.50** | **43.30** | **44.83** | **43.04** |
| residual_max | 8.100 | 8.100 | 8.101 | 8.075 |
| residual_rms | 8.146 | 8.153 | 8.160 | 8.139 |
| act_max | 8.181 | 8.173 | 8.136 | 8.102 |
| act_scale | 8.586 | 8.204 | 8.187 | 8.154 |
| random (control) | 8.181 | 8.221 | 9.048 | 8.487 |

Actual storage is identical across a column (4.57 bits at 2%, 7.45 at 20%), so this is not a
budget effect. Llama-3.2-1B (FP16 9.757, GPTQ-4 base 10.557) shows the same collapse for the
iterative selector, with an important asymmetry:

| selector | 2% | 5% | 10% | 20% |
|---|---|---|---|---|
| **greedy** (iterative) | **63.95** | **66.84** | **77.99** | **65.00** |
| greedy_indep (one-shot) | 11.28 | — | — | 11.39 |
| residual_max | 10.42 | — | — | 10.36 |
| act_max | 10.71 | — | — | 10.38 |
| random (control) | 10.41 | 10.41 | 10.43 | 10.58 |

Two observations. First, **the harm requires a residual-driven *set* objective**: magnitude-based
and random selection of the same number of channels leave the model intact, so simply restoring
2–20% of columns to FP16 is not what breaks it. Second, **the iterative selector is the
consistent offender**: `greedy` collapses on both models at every budget, whereas the one-shot
ablation `greedy_indep` collapses on 3B but is only mildly harmful on 1B (11.28 vs base 10.56).
Iterative re-scoring concentrates the selection on precisely the columns whose restoration is
most damaging.

### 5.2 Model dependence

The failure is not universal. Running the identical configuration (greedy, 2%, GPTQ-4 base,
matched scope and storage) across four models:

| model | FP16 | GPTQ-4 base | greedy@GPTQ 2% | verdict |
|---|---|---|---|---|
| Llama-3.2-1B | 9.757 | 10.557 | **63.95** | catastrophic |
| Llama-3.2-3B | 7.817 | 8.172 | **51.68** (55.35 in the standalone diagnostic run) | catastrophic |
| Llama-2-7B | 5.469 | — | 5.571 | safe |
| Qwen2.5-3B | 8.030 | 8.290 | 8.335 | safe |

On Qwen2.5-3B the full selector panel is benign (greedy 8.34, greedy_indep 8.36, random 8.30 at
2%). So the failure mode is real and reproducible on one model family and absent on two others.
This is a genuine boundary condition rather than a fragile artifact, and it constrains any
mechanism: whatever breaks must be a property of the Llama-3.2 GPTQ residual structure, not of
the protection form itself.

### 5.3 Integrity checks

Three checks rule out the obvious artifacts.
- **Base regeneration.** The original GPTQ base was lost and regenerated with the same recipe;
  the catastrophe reproduces on the regenerated base, so it is not a property of one artifact.
- **Export/reload fidelity.** For every catastrophic configuration, materialising the protected
  model to dense weights and re-measuring gives the same perplexity (3B 55.35 → 55.19; 1B 63.95
  → 63.82; |Δ| < 0.13), and reloading the saved checkpoint from disk reproduces it again
  (3B 51.76, 1B 63.82). The collapse is a property of the weights, not of a runtime path.
- **Budget exhaustion.** The selector is by default forced to spend the full budget. Re-running
  with early stopping (stop when the next marginal gain is non-positive) at 2% yields the
  identical perplexity, i.e. at this budget every selected channel had positive predicted gain —
  the catastrophe is not caused by being forced to protect channels the objective itself deems
  useless.

## 6 Interaction-aware selection does not pay

Where protection helps (HQQ), the iterative selector and the one-shot full-Hessian ablation are
within ~0.03 PPL of each other at every budget (§4), i.e. modelling interactions between
protected channels does not buy accuracy over evaluating the same objective once. Where
protection hurts (GPTQ), the iterative selector is *worse* (§5.1): on 1B it collapses to 63.95
while the one-shot ablation stays at 11.28.

The defensible statement is therefore: **iterative re-scoring does not improve on one-shot
full-Hessian selection, and on a compensated base it actively concentrates harm.** We do not
claim that second-order machinery is unnecessary relative to activation magnitude — that would
require an equivalence test against `act_max` with a pre-declared margin, which we report as
future work rather than assert from a ~0.03 PPL difference.

## 7 The base quantizer is the ceiling

On the matched weight-only storage axis, the base quantizer dominates. The GPTQ-4 base at 4.25
bits (3B: 8.172) is not reached by any protected HQQ configuration at comparable storage, and
protection on GPTQ buys at most ~0.07–0.10 PPL (residual_max, 8.100 at 4.57 bits vs base 8.172 at
4.25 bits) for a +0.32-bit premium. Correcting the storage accounting matters here: charging the
imported GPTQ base at its true group size (128) rather than the internal default (64) moves the
2% operating point from 4.82 to **4.57** bits. We therefore state F4 as an observation on this
grid — the base dominates, and protection is a small correction on top — not as a general law.

## 8 Downstream evaluation

**Pending (corrected run).** Downstream zero-shot evaluation (hellaswag, arc-easy, arc-challenge,
piqa, winogrande, lambada-openai via lm-eval) is being re-run under the corrected scope, at
uniform full scale, with a multi-seed matched-bit random control. Earlier downstream numbers in
this repository are superseded: they were produced from a stale exported checkpoint (a resume
guard reused an lm-eval directory from an earlier, healthy export), which we detected because the
recorded `expected_ppl` sidecar and the measured accuracy were mutually inconsistent. Because the
export/reload identity is now verified (§5.3), a faithful downstream evaluation of a
perplexity-55 checkpoint must show correspondingly degraded accuracy; we report the measured
values here rather than predict them.

## 9 Mechanism: hypothesis and the test that would confirm it

**Hypothesis.** GPTQ quantizes column by column and pushes each column's error into the
*remaining* columns. The residual `ΔW = W − Q(W)` is therefore not arbitrary: it encodes the
compensation. A residual-driven selector picks the columns carrying the most of that encoded
correction, and restoring them to FP16 post hoc removes the error the other columns were
compensating *for*, leaving the complement mis-corrected. Random and magnitude selectors do not
preferentially pick compensation-carrying columns, which is consistent with their safety.

**Evidence and its limits.** The hypothesis is consistent with the selector asymmetry (only
residual-driven objectives are harmful) and with the iterative/one-shot gap (§5.1, §6). However,
we measured the overlap between the selected set and the columns of highest residual energy at
only 2–3%, so a simple "it picks the biggest-residual columns" account is *not* supported; the
iterative objective selects an interaction-structured set, not the top-energy columns. We
therefore report the mechanism as an open hypothesis.

**The decisive test.** Hold the selected set `S`, the model, the calibration data, the budget and
the evaluator fixed, and vary only *when* protection is applied: (A) quantize all columns with
GPTQ, then restore `S` to FP16 (post hoc, our failing configuration); (B) keep `S` exact
*throughout* compensation so the complement is compensated knowing `S` is exact (OWQ-style). If
A collapses while B stays healthy on the same `S`, the compensation-breaking account is
established causally. We have implemented both arms; our first execution used a one-shot GPTQ
implementation whose base was itself degenerate (perplexity > 3000 for all arms, including the
unprotected base), so it is uninformative and we exclude it. The experiment is being re-run on a
sequential GPTQ implementation. **We make no causal claim until it returns.**

## 10 Limitations

- **Model coverage.** The failure is established on Llama-3.2-1B/3B and shown absent on
  Llama-2-7B and Qwen2.5-3B. We do not yet know which property of a model or its GPTQ residual
  predicts susceptibility, and we cannot claim it generalises to families we did not test.
- **Mechanism.** Open (§9). The causal experiment is specified and implemented but not yet
  validly executed.
- **Downstream.** Pending (§8); all conclusions in this draft are perplexity-based.
- **Base coverage.** One compensated base (GPTQ). Whether the failure appears on other
  compensated bases (e.g. AWQ) determines whether it is GPTQ-specific.
- **Storage is theoretical.** Weight-only bytes; no kernels, packing overhead or latency.
- **Selection cost.** We do not report wall-clock/memory cost of each selector; the Hessian-based
  selectors are substantially more expensive than the magnitude ones, which strengthens the
  practical case against them given §6.

## 11 Conclusion

Outlier-channel protection is not a universally safe add-on. On a data-free base it delivers a
small, real gain over a matched-bit random control. On an error-compensated base it delivers
almost nothing — and if the channels are chosen by a residual-driven set objective and restored
*post hoc*, it can destroy the model, raising perplexity from 8 to 40–55 while a random control
at identical storage is harmless. The effect is faithful through export and reload, stable across
budgets, and model-dependent. Practitioners should prefer a strong base over post-hoc protection,
and where protection is wanted on a compensated base, integrate it into compensation (as OWQ
does) rather than applying it afterwards — and validate on the specific model, because whether
the failure appears depends on it.

---

## Appendix A. Reproducibility and statistics

- **Provenance.** Every perplexity in this draft is read from a committed
  `runs/final/sweeps/<base>/<model>/<selector>/seed-1234/channel_pareto.json` with
  `skip_lm_head=true` and `base_group_size=128`, or from `results/f3_*`/`results/f3check_*` for
  the model-dependence panel. The two 3B greedy@2% values (51.68 in the matrix sweep, 55.35 in
  the standalone diagnostic) come from separate runs of the same configuration; both are
  catastrophic and we report the matrix value in tables.
- **Random controls.** `random` is run with 3 seeds per (model, base, budget); tables report the
  seed-1234 value, with the multi-seed spread used for the control band in the dose-response
  figure.
- **Storage.** Weight-only bits/parameter, group scales/zeros at the base's true group size,
  FP16 correction columns and channel indices included; embeddings/`lm_head`/norms excluded and
  FP16 in all methods.
- **Evaluator.** One path for all numbers; exported checkpoints are evaluated after reload. The
  imported GPTQ base measures 10.557 (1B) in our evaluator vs 10.363 as reported by
  LightCompress; the reloaded value is used operationally.
