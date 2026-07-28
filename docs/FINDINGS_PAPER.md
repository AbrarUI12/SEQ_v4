# When Perplexity Lies: A Controlled Perplexity Collapse with No Downstream Degradation in Weight-Quantized LLMs

## Abstract

Post-training quantization of large language models is evaluated overwhelmingly by WikiText
perplexity, and regressions are routinely read as utility loss. We show this inference can fail
badly. Using outlier-channel protection — keeping a small fraction of input channels in FP16 on a
low-bit base — we construct a controlled generator of extreme perplexity damage: on an
error-compensated GPTQ-4 base, a selector ranking channels by the Hessian-weighted residual
objective raises Llama-3.2-3B WikiText-2 perplexity from 8.17 to 51.68 at matched storage. Disk
reload verifies the evaluated checkpoint is that model (51.76). Yet on six zero-shot tasks it
loses nothing: macro accuracy 67.61% versus 67.10% for its base, a paired-bootstrap difference of
+0.51 points whose interval excludes zero, and LAMBADA perplexity 4.17. The pattern replicates on
Llama-3.2-1B (10.56 to 63.83 reload-verified, +0.32 points). A per-token decomposition rules out
the obvious explanation: the damage is broad, not a tail artifact — 74.6% of tokens degrade and
excluding the worst 5% still leaves a 4x gap. The damage requires the selector's off-diagonal
Hessian coupling, and the natural mechanism is refuted: scored by its exact objective, the
harmful selector is *anti*-correlated with GPTQ's compensation while the safest selector tracks
it almost perfectly.

_(200 words)_

---

## 1 Introduction

Weight-only post-training quantization (PTQ) is the default way to shrink large language models,
and the field's default report is a WikiText perplexity table. Methods are accepted or rejected on
perplexity deltas that are often far smaller than one point, and deployment gates are frequently
written in the same terms. Implicit in this practice is an assumption: that perplexity movement
tracks the model's usefulness.

We show that assumption can fail by a wide margin, using a construction that is entirely within
standard PTQ practice. *Outlier-channel protection* — keeping a small fraction of input channels
in FP16 on top of a low-bit base — is a common recipe (LLM.int8, OWQ, SpQR). Its selection signal
is usually presented as a design detail. We find that one standard choice of signal, applied to an
error-compensated base, produces a catastrophic perplexity collapse:

> On Llama-3.2-3B with a GPTQ-4 base, protecting 2% of input channels chosen by the
> Hessian-weighted residual objective moves WikiText-2 perplexity from **8.17 to 51.68**, at
> matched theoretical weight-only storage, while random selection of the same number of channels
> leaves it at 8.18.

A 6.3× perplexity increase would normally end the discussion. We instead treat it as an
instrument. We export that exact configuration to a dense checkpoint, verify by in-memory
materialization and by disk reload that the checkpoint *is* the perplexity-51.76 model, and then
evaluate it on six standard zero-shot tasks. It shows no degradation at all: macro accuracy
**67.61%** against **67.10%** for its own unprotected base and **68.43%** for FP16, a paired
bootstrap difference of **+0.51 points [+0.13, +0.86]** — statistically *favourable* — with
LAMBADA perplexity **4.17**, comparable to the base's 4.28.

**Contributions.**
1. A reload-verified demonstration that a **~6× WikiText-2 perplexity collapse can carry zero
   downstream cost** on six tasks, on **two models**, including a token-level task on a second
   corpus (§4.1–4.2).
2. A **per-token decomposition falsifying the tail explanation**: the degradation is broad
   (74.6% of tokens worse, median ΔNLL +0.287), so the decoupling is not an artifact of a few
   catastrophic tokens (§4.3).
3. A characterization of *what generates* the perplexity damage: it requires the selector's
   **off-diagonal Hessian coupling**. Selectors using no Hessian, only its diagonal, only residual
   magnitude, or random selection are all benign at every budget we tested (§5).
4. A **refuted mechanism**, measured on the selector's exact objective: the harmful selector is
   anti-correlated with compensation magnitude (ρ = −0.23 / −0.14, positive in under a third of
   layers) while the safest selector is almost perfectly aligned with it (ρ ≈ 0.99 in every
   layer). Protecting compensation-bearing columns is neither sufficient nor necessary for the
   collapse (§6).
5. A quantification of how much the effect depends on the **selector's own Hessian estimate**,
   which is large on one model and negligible on another (§5.3), and a **model-scope boundary**:
   two of four checkpoints exhibit the collapse, and on Qwen2.5-3B no selector is harmful at all
   (§5.4).
6. Supporting audits under matched storage with random controls, including a matched-bit
   downstream signal-versus-random test (§7), plus code, pinned environment and all result JSON.

**Scope of the claim.** We do not claim perplexity is uninformative, nor that this configuration
is one practitioners would deliberately choose. We claim that a perplexity gap of this magnitude,
produced by an ordinary PTQ manipulation, is not sufficient evidence of utility loss — and that
quantization studies and regression gates that rely on perplexity alone can therefore be
badly miscalibrated.

**What is not new.** Activation-outlier protection (LLM.int8, CMPQ), Hessian- and error-based
column selection (OWQ, SpQR), and integrating protected-column selection into the compensation
pass (OWQ) are all prior work. Our contribution is the controlled decoupling demonstration, the
structural condition on the selector, and the negative mechanism result.

## 2 Related work

**Outlier-aware mixed precision.** LLM.int8() (Dettmers et al., 2022) decomposes the matmul so
outlier feature dimensions stay in FP16. OWQ (Lee et al., 2023) scores input columns by
Hessian-weighted quantization error, keeps the top ones in FP16, and orders them last inside the
OPTQ pass so compensation proceeds knowing which columns remain exact. SpQR (Dettmers et al.,
2023) isolates outlier weights in a sparse high-precision format. Atom (Zhao et al., 2024),
CLAQ and SqueezeLLM (Kim et al., 2023) use related group/column/row schemes; CMPQ sets per-channel
precision from activation statistics. Our selectors are drawn from this family; we vary the signal
and the base rather than proposing a new protector.

**Error compensation.** GPTQ (Frantar et al., 2023), building on OBQ (Frantar & Alistarh, 2022),
quantizes column by column and propagates each column's rounding error into the not-yet-quantized
columns, minimizing ‖ΔW X‖². The residual it leaves is therefore structured, which is what makes
the base×selector interaction we study non-trivial. HQQ (Badri & Shaji, 2023) is calibration-free
but is *not* plain round-to-nearest: it optimizes quantization parameters with a half-quadratic
splitting procedure under a robust reconstruction loss.

**Bit allocation and interaction modelling.** SliM-LLM and CoopQ allocate budgets using salience
and interaction structure; AMQ searches allocations under memory constraints; EWQ uses an entropy
prior across layers. These motivate our interaction ablation and our matched-bit random controls.

**Evaluating quantized models.** Broad benchmark studies (e.g. Jin et al., 2024) report downstream
task results for quantized LLMs, and several works note that perplexity and task metrics can move
differently. Our contribution is sharper and adversarial: rather than observing correlation
strength across methods, we *construct* a checkpoint with an extreme perplexity gap and verify
that the downstream cost is nil, which bounds how much a perplexity delta can be trusted.

## 3 Setup

**Protection form.** For a linear layer with weight `W ∈ R^{out×in}` and protected input-channel
set `S`, the protected forward pass is `y = Q(W)x + x[S]·(W − Q(W))[:,S]ᵀ`. Materializing gives
`W_dense = Q(W) + scatter(W − Q(W), S)`, algebraically identical to the forward pass, so a faithful
export must reproduce the runtime perplexity. We use this identity as an integrity check (§4.2).
Storage is additive: the low-bit base is retained and FP16 correction columns are stored on top,
rather than replacing base columns.

**Bases.** (i) **HQQ-4**, calibration-free (half-quadratic parameter optimization, no error
compensation). (ii) **GPTQ-4**, an error-compensated base produced by LightCompress at W4 group
size 128 with a fixed WikiText-2 calibration set (128 × 2048 tokens), imported as fake-quantized
weights and re-evaluated in our own evaluator.

**Selectors.** With residual `ΔW = W − Q(W)` and input Hessian `H = XᵀX`:

| selector | score | ΔW | H diag | H off-diag |
|---|---|---|---|---|
| `random` (control, 3 seeds) | uniform | ✗ | ✗ | ✗ |
| `act_max`, `act_scale` | activation magnitude / scale | ✗ | ✗ | ✗ |
| `residual_max`, `residual_rms` | per-column magnitude of ΔW | ✓ | ✗ | ✗ |
| `hessian_diag` | `H_jj` | ✗ | ✓ | ✗ |
| `greedy_indep` | `2⟨ΔW_j,(ΔW H)_j⟩ − ‖ΔW_j‖²H_jj`, evaluated once | ✓ | ✓ | ✓ |
| `greedy` | same objective, re-scored iteratively (OMP-style) | ✓ | ✓ | ✓ |

`greedy_indep` is a **one-shot full-Hessian** selector, not an activation-magnitude score. The last
two maximize the exact reduction in `‖ΔW X‖² = tr(ΔW H ΔWᵀ)` obtainable by restoring columns.

**Storage.** We report **matched theoretical weight-only storage**: bits per quantized-linear
parameter, counting low-bit values, group scales/zeros at the base's true group size (128 for the
imported GPTQ base), FP16 correction columns and channel indices. Embeddings, `lm_head` and norms
are FP16 in every method and excluded from the axis; all runs exclude `lm_head` from quantization
so the quantized scope matches the baseline exactly. No kernels, packing or latency are modelled.

**Evaluation.** WikiText-2 perplexity, sequence length 2048, canonical non-overlapping chunking,
one evaluator for every perplexity in this paper; exported checkpoints are evaluated after reload.
Downstream: lm-eval-harness v0.4.12, six zero-shot tasks (hellaswag, arc-easy, arc-challenge,
piqa, winogrande, lambada-openai), full test sets (no example limit), with paired-bootstrap 95%
CIs computed on per-example correctness. The imported GPTQ base measures 10.557 on Llama-3.2-1B in
our evaluator versus 10.363 as reported by the producing toolkit, so we use the **reloaded k=0
checkpoint** as the operational base for every contrast, keeping all comparisons inside one
evaluator.

**Scalar-signal calibration used padded prompts.** The activation statistics behind
`act_max`, `act_scale`, `residual_max` and `residual_rms` were collected with prompts padded to
the full sequence length, and the hooks accumulate every position, so with ~51 short prompts those
statistics are dominated by pad/EOS states. The Hessian path used by `greedy`/`greedy_indep`
deliberately avoids padding for exactly this reason, so the two calibration paths are not
comparable. This weakens the description of the scalar selectors as "informed"; note it does not
threaten §5.1's conclusion, since a selector closer to random would if anything reinforce the
finding that only objective-coupled selectors do harm. A `--no_pad_calibration` path now exists
and the robustness re-measurement is reported in §9.

**Calibration provenance.** The base quantizer's Hessian came from 128 × 2048 ≈ 262k WikiText-2
tokens. Our selectors' Hessian, by default, came from a 51-prompt instruction set of ≈500 tokens —
rank-deficient for layers with 3072–8192 input channels, and a different distribution. We treat
this as an experimental variable rather than a fixed detail, and measure its effect in §5.3.

## 4 A perplexity collapse with no downstream cost

### 4.1 The two measurements

On Llama-3.2-3B with the GPTQ-4 base (8.172 in our evaluator, 4.25 bits), protecting 2% of input
channels selected by `greedy` gives **51.68** perplexity at 4.57 bits. All entries below share
identical storage at a given budget.

| point | WikiText-2 PPL | macro acc (6 tasks) | LAMBADA PPL |
|---|---|---|---|
| FP16 | 7.817 | 68.43% | — |
| GPTQ-4 (base) | 8.172 | 67.10% | 4.28 |
| **greedy@GPTQ, 2%** | **51.68** | **67.61%** | **4.17** |
| residual_max@GPTQ, 2% | 8.100 | 67.47% | — |
| HQQ-4 (base) | 8.328 | 66.86% | — |
| best greedy@HQQ, 20% | 7.994 | 68.14% | — |
| random@HQQ, 20% | 8.247 | 67.21% | — |

Accuracy is `acc_norm` where the task provides it and `acc` otherwise, macro-averaged over the six
tasks; all values are read from `results/downstream.json`, the same artifact the confidence
intervals below are computed from.

Paired bootstrap, macro-Δ accuracy with 95% CI:

| contrast | Δ (pts) | 95% CI |
|---|---|---|
| **greedy@GPTQ − GPTQ-4** | **+0.51** | **[+0.13, +0.86]** |
| resmax@GPTQ − GPTQ-4 | +0.37 | [−0.06, +0.77] |
| best@HQQ − HQQ-4 | +1.28 | [+0.84, +1.68] |
| best@HQQ − random@HQQ | +0.92 | [+0.50, +1.34] |

The model whose perplexity rose 6.3× is **not** worse downstream; its confidence interval for the
contrast against its own base excludes zero on the favourable side. Its LAMBADA perplexity — a
token-level metric on a different corpus — is 4.17, slightly better than the base's 4.28.

### 4.2 The checkpoint is the same model

Because this is the crux, we establish checkpoint identity explicitly. The protection form
guarantees `runtime_ppl == materialized_ppl` algebraically (§3), and we confirm it empirically at
each stage of the chain that produced the downstream number:

| stage | measurement | value |
|---|---|---|
| in-sweep runtime (forward pass) | WikiText-2 PPL | 51.68 |
| in-memory materialization to dense weights | WikiText-2 PPL | 55.19 → 51.76¹ |
| the exported checkpoint, reloaded from disk | WikiText-2 PPL | **51.76** (`PASS`, tol 0.5) |
| the same directory, scored by lm-eval-harness | macro accuracy | **67.61%** |

¹ Two standalone diagnostic runs of the same configuration recorded 55.35/55.19; the matrix sweep
records 51.68 and the disk reload 51.76. All are catastrophic; the appendix provenance table
reconciles them. The reload was run on the exact directory lm-eval loaded, after the downstream
evaluation, and passed against the sweep value.

So the downstream evaluation scored a checkpoint whose WikiText-2 perplexity is 51.76. The
decoupling is a property of the weights, not an artifact of an evaluation path.

**It replicates on a second model.** Llama-3.2-1B reproduces the pattern under the same
verification discipline: the exported `greedy@GPTQ` checkpoint reloads at **63.83** against a
sweep value of 63.95 (`PASS`, Δ = 0.124), a **6.0×** increase over its 10.557 base, and that
checkpoint scores macro **58.67%** against the base's **58.35%** — a contrast of **+0.32 pts
[−0.07, +0.73]**. The interval includes zero, so on 1B we claim no degradation rather than an
improvement; on 3B the interval excludes zero on the favourable side. Both models show a ~6×
perplexity collapse with no measurable downstream cost.

| model | base PPL | greedy@GPTQ PPL (reload-verified) | ratio | macro Δ vs base |
|---|---|---|---|---|
| Llama-3.2-3B | 8.172 | 51.76 (`PASS`) | 6.3× | **+0.51 [+0.13, +0.86]** |
| Llama-3.2-1B | 10.557 | 63.83 (`PASS`) | 6.0× | +0.32 [−0.07, +0.73] |

### 4.3 The damage is broad, not a tail artifact

The obvious explanation is that perplexity, being `exp` of a *mean* negative log-likelihood, is
dominated by its worst tokens: if a small minority of tokens became catastrophically improbable,
perplexity would explode while the bulk of the model's behaviour — and therefore task accuracy —
stayed intact. We pre-registered this hypothesis and tested it directly, scoring both checkpoints
on the identical token stream (141 windows, 288,627 supervised targets) and decomposing the
per-token negative log-likelihood.

**The hypothesis is false.** The degradation is broad:

| statistic | value |
|---|---|
| tokens with higher NLL | **74.6%** |
| tokens with lower NLL | 25.4% |
| median ΔNLL | **+0.287** |
| share of total increase carried by the worst 1% of tokens | 6.9% |
| share carried by the worst 10% | 46.3% |

Perplexity after excluding the worst-k% of tokens for both models:

| excluded | base | greedy@GPTQ |
|---|---|---|
| 0% | 8.17 | 51.76 |
| 1% | 8.30 | 46.85 |
| 5% | 8.59 | **34.56** |

Removing the worst 5% of tokens — fifty times more than a tail explanation would require — still
leaves a 4× gap. The *median* token is meaningfully worse, and three quarters of all tokens
degrade. This is not a small set of outliers dragging up a mean; the model is genuinely and
broadly worse at next-token prediction on WikiText-2.

That makes the result more surprising, not less, and it removes the most comfortable way to
dismiss it. A model can be substantially and pervasively worse at modelling a corpus while
answering six downstream benchmarks exactly as well as before. Whatever the tasks measure, a broad
degradation in next-token likelihood on WikiText-2 is not sufficient to disturb it. We do not have
a positive account of why, and we do not offer one.

## 5 What generates the perplexity damage

### 5.1 The damage requires off-diagonal Hessian coupling

All selectors at matched storage on the GPTQ-4 base (Llama-3.2-3B base 8.172; Llama-3.2-1B base
10.557), protection fraction 2%:

| selector | Hessian use | 3B | 1B |
|---|---|---|---|
| `random` (seed 1234) | none | 8.181 | 10.41 |
| `act_max` | none | 8.181 | 10.71 |
| `act_scale` | none | 8.586 | 10.93 |
| `residual_max` | none | 8.100 | 10.42 |
| `residual_rms` | none | 8.146 | 10.43 |
| **`hessian_diag`** | **diagonal only** | **8.153** | **10.441** |
| `greedy_indep` | full, one-shot | 41.50 | 11.28 |
| `greedy` | full, iterative | **51.68** | **63.95** |

`hessian_diag` — which uses the Hessian, but only its diagonal — is benign at every budget we
tested (3B: 8.153/8.163/8.156/8.147 at 2/5/10/20%; 1B: 10.441/10.462/10.475/10.450). Only the two
selectors that use the **off-diagonal, cross-column** terms cause damage. This rules out both
"reads the quantization residual" and "uses the Hessian" as the operative property and isolates the
interaction structure.

We avoid absolutes: the benign selectors are not exactly neutral. `act_scale` reaches 8.586 at 2%
on 3B and `random` reaches 9.048 at the 10% budget — moderate movements, one to two orders of
magnitude smaller than the collapse. Throughout we call a change **indistinguishable** if within
±0.1 PPL of the base, **moderate** up to ~1 PPL, and **catastrophic** above 5× the base.

**The ordering does not depend on pad tokens.** The scalar statistics above were collected with
prompts padded to the full sequence length (§3), so we re-measured them on real tokens only
(`--no_pad_calibration`, Llama-3.2-3B). Three of the four scalars are unmoved — `act_max`
8.181→8.178, `residual_max` 8.100→8.100, `residual_rms` 8.146→8.146 — and only `act_scale` shifts
materially, 8.586→8.142, so its apparent weakness was a padding artifact rather than a property of
the signal. Under honest calibration all four scalars sit within 8.10–8.15 against a base of
8.172, which sharpens rather than weakens the finding: every Hessian-free selector is benign, and
the gap to the coupled selectors is unchanged. The HQQ result of §7 also survives, with informed
selectors at 8.100–8.141 against a random control at 8.319.

### 5.2 Severity within the coupled selectors

Among the two coupled selectors, iterative re-scoring is at least as harmful as one-shot
evaluation, but the separation is model-dependent and not monotone in budget. On 1B the ordering is
clean at 2% (`greedy` 63.95 vs `greedy_indep` 11.28); on 3B both collapse (51.68 vs 41.50) and
`greedy` is worse at 2% and 5% but *less* harmful at 10% and 20% (38.39, 40.65 vs 44.83, 43.04). We
therefore report that coupling is necessary for harm, and that harm saturates rather than scaling
smoothly once present.

### 5.3 The selector's own Hessian estimate matters, model-dependently

Our default selector Hessian used ≈500 tokens of out-of-distribution prompts, against 262k
in-distribution tokens for the base. Re-running `greedy` with the selector Hessian estimated from a WikiText-2 sample of the same
size as the base's (128 × 2048 tokens):

| model | selector H ≈500 tokens | selector H = 262,144 tokens |
|---|---|---|
| Llama-3.2-3B | 51.68 | **58.24** |
| Llama-3.2-1B | 63.95 | **11.39** |

On 3B the collapse persists and is slightly worse, so it is **not** an artifact of an
under-estimated Hessian. On 1B most of the collapse disappears (11.39 against a 10.557 base — a
moderate, not catastrophic, movement), so on that model the estimate quality was doing much of the
work. Any study using Hessian-based selection should therefore report the selector's calibration
size and distribution; we did not initially, and it materially changes one of our two models.

**This is a size-matched, not a sample-matched, comparison.** The base was calibrated by
LightCompress with its own preprocessing and seed (`wikitext2_gptq`, seed 0) while our selector
draws an independent sample (seed 1234). The two therefore share a corpus and a token budget but
not the same windows. This experiment consequently bounds the effect of *estimator quality* and
does **not** eliminate calibration-sample mismatch as a confound; a sample-matched test requires
persisting the base's exact calibration token ids and reusing them, which we did not do.

### 5.4 On which models the collapse occurs at all

The perplexity collapse is not a universal property of the recipe. Running the identical
configuration (`greedy`, 2%, GPTQ-4 base, matched scope and storage) across four checkpoints:

| model | FP16 | GPTQ-4 base | greedy@GPTQ 2% | Δ vs base |
|---|---|---|---|---|
| Llama-3.2-1B | 9.757 | 10.557 | **63.95** | +53.4 |
| Llama-3.2-3B | 7.817 | 8.172 | **51.68** | +43.5 |
| Qwen2.5-3B | 8.030 | 8.290 | 8.335 | +0.045 |
| Llama-2-7B | 5.469 | —¹ | 5.571 | — |

¹ The unprotected base was not measured in the Llama-2-7B run, so we compare only against FP16
and treat that row as weaker evidence.

On Qwen2.5-3B the *entire* selector panel is benign at 2% — `greedy` 8.335, `greedy_indep` 8.357,
`residual_rms` 8.339, `residual_max` 8.351, `act_max` 8.366, `act_scale` 8.372, `random` 8.300,
against a base of 8.290. The spread across every selector is under 0.09 perplexity, so on this
model the choice of selection signal is immaterial and no configuration we tested is harmful.

The collapse therefore requires something beyond the structural condition of §5.1: the selector
must use off-diagonal coupling **and** the model must be susceptible. Two of four checkpoints are,
both from the same family. We have no predictor for susceptibility, and identifying one is the
clearest route to turning this from an observed boundary into a diagnostic.

**These models cannot supply the decoupling replication.** The decoupling claim of §4 requires a
checkpoint that is simultaneously catastrophic in perplexity and intact downstream. Qwen2.5-3B and
Llama-2-7B are not catastrophic in perplexity at all, so there is nothing to decouple: evaluating
them downstream would confirm that a healthy model scores healthily, which is uninformative. A
second *decoupling* datapoint can only come from a model that exhibits the collapse — i.e.
Llama-3.2-1B, whose export failed reload validation (§9).

## 6 The compensation account, tested directly and refuted

The natural explanation for §5 is that the coupled selectors identify columns carrying GPTQ's
deposited error correction, and that restoring those columns to FP16 discards compensation the
remaining columns were tuned around. This predicts that harmful selectors should rank
compensation-bearing columns highly.

We tested it directly. During a sequential GPTQ pass we recorded, per input channel, the
compensation magnitude `comp_j = ‖W_pre_quant[:,j] − W_orig[:,j]‖₂` — how far compensation moved
that column before it was itself quantized — and rank-correlated each selector's score against it.
Crucially, the harmful selector is scored by its **exact first-step gain**
`2⟨ΔW_j,(ΔW H)_j⟩ − ‖ΔW_j‖²H_jj`, computed in-pass against the same Hessian the selector itself
receives (Spearman, median over 56 layers per model):

| selector | 3B median ρ | 1B median ρ | layers with ρ>0 | perplexity verdict |
|---|---|---|---|---|
| `residual_rms` | **+0.988** | **+0.981** | 100% | benign |
| `residual_max` | +0.548 | +0.615 | 100% | benign |
| `hessian_diag` | −0.133 | −0.092 | ~33% | benign |
| **`greedy` (true objective)** | **−0.230** | **−0.142** | **29%** | **catastrophic** |

**The prediction fails, and fails in the strongest possible direction.** The selector most nearly
*identical* to compensation magnitude (`residual_rms`, ρ ≈ 0.99 in every single layer) is among the
safest, while the catastrophic selector is *anti*-correlated with compensation and ranks it
positively in fewer than a third of layers. Protecting compensation-bearing columns is therefore
neither sufficient nor necessary for the collapse; if anything the harmful selector systematically
*avoids* them. Both models agree.

We report this as a negative result and do **not** substitute a new mechanism for it: what §5
establishes is a structural condition (off-diagonal coupling), not a causal account.

> **Correction.** An earlier version of this section scored the harmful selector as
> `‖ΔW_j‖²·H_jj` — only the subtracted term of the gain above, with the off-diagonal coupling
> dropped — and reported ρ = +0.390 (3B) / +0.386 (1B). That expression disagrees with the true
> ranking on the majority of layers, so it did not describe the selector it was attributed to. The
> table above uses the selector's actual objective; the superseded column reproduces at +0.388 /
> +0.384 and is retained in the released artifacts as `diag_proxy`. The corrected measurement
> **strengthens** the refutation rather than reversing it. `greedy_select.first_step_gains` is now
> the single definition shared by selector and analysis, pinned by `tests/test_first_step_gains.py`.

The clean causal test remains the ordering intervention — hold the protected set fixed and vary
only whether it is made exact before or after compensation. We implemented it but do not report a
result: our internal sequential GPTQ produced a degenerate base at 3B, so all arms were
uninformative (§9).

## 7 Supporting audits

**Protection helps a calibration-free base.** On HQQ-4, every informative signal beats the
matched-bit random control at both budgets and both sizes (3B at 2%: `greedy` 8.100, `act_max`
8.108, `residual_rms` 8.141 vs `random` 8.318). Downstream this reproduces at matched storage:
best@HQQ − random@HQQ = **+0.92 pts [+0.50, +1.34]**, so the gain is a selection effect and not a
budget effect.

**Interaction modelling does not pay where protection helps.** On HQQ, iterative and one-shot
selection are within ≈0.03 PPL at every budget — we observed no material difference, which is not
an equivalence result — while on the compensated base the iterative variant is strictly worse.

**The base dominates the frontier.** At matched weight-only storage the base quantizer sets the
ceiling: the GPTQ-4 base at 4.25 bits (3B 8.172) is not reached by any protected HQQ configuration
at comparable storage, and the best protection on GPTQ buys ≈0.07 PPL (`residual_max`, 8.100 at
4.57 bits). We state this as an observation on this grid, not a general law.

## 8 Conclusion

We constructed, verified and evaluated checkpoints whose WikiText-2 perplexity is ~6× their
base's and whose downstream accuracy is not worse — on 3B slightly better, with a confidence
interval excluding zero — across six zero-shot tasks, on two models, with a token-level metric on
a second corpus also unharmed. The construction uses only standard PTQ components. A per-token
decomposition shows the perplexity damage is broad rather than a tail artifact: three quarters of
tokens degrade and the median token worsens, so the models really are pervasively worse at
modelling the corpus while answering the benchmarks exactly as well. We further show that
producing this damage requires the selector's off-diagonal Hessian coupling. The obvious mechanism — destroying
error compensation — is refuted by direct measurement against the selector's exact objective: the
harmful selector avoids compensation-bearing columns rather than targeting them (§6).

The practical consequence is narrow and concrete. Perplexity remains a useful diagnostic, but a
perplexity delta, even a very large one, is not by itself evidence of utility loss for a quantized
model. Quantization studies should report task metrics alongside perplexity, and perplexity-only
regression gates should be expected to reject models that are downstream-equivalent.

## 9 Limitations

- **Scale and family.** The decoupling is verified on both susceptible checkpoints we have,
  Llama-3.2-3B and 1B, but both come from one model family. Of the four checkpoints tested only
  these two exhibit the perplexity collapse at all (§5.4), so Qwen2.5-3B and Llama-2-7B cannot
  supply further replication — a model healthy in perplexity has no decoupling to demonstrate.
  Broadening the claim requires finding further *susceptible* models, which in turn requires a
  predictor for susceptibility that we do not have.
- **A known export failure.** The Llama-3.2-1B `greedy@GPTQ` checkpoint reloads at 203.72 against
  an expected 63.95 (`FAIL`). It is a third distinct value for that configuration, so we exclude
  the 1B `greedy@GPTQ` downstream row entirely rather than report it. The 3B chain, by contrast,
  passed reload validation against the sweep value.
- **A published analysis error, corrected.** The §6 probe originally scored the greedy selector
  with a diagonal expression rather than its true first-step gain. It has been corrected, re-run on
  both models, and the refutation is stronger on the right quantity (§6). The error was confined to
  the analysis probe; the selector, the sweeps and every perplexity here are unaffected. The two
  definitions are now shared and regression-tested.
- **Scalar-signal calibration was padded.** See §3. Re-measuring on real tokens leaves the ordering
  intact on Llama-3.2-3B (§5.1); we have not repeated it on 1B.
- **§5.3 is size-matched, not sample-matched.** It does not establish token identity with the
  base's calibration set. The released code can now consume a saved token set
  (`--selector_calib_tokens`) so future bases can be made token-identical, but the base used here
  predates that and its calibration tokens were never persisted.
- **No causal mechanism, and no positive account of the decoupling.** §6 falsifies the
  compensation explanation and §4.3 falsifies the tail explanation; §5 gives a structural
  condition only. We can say what the decoupling is *not* caused by more confidently than what it
  is caused by, and a numerical-conditioning explanation for the collapse is not excluded.
- **The ordering intervention did not produce a usable result.** Our internal sequential GPTQ
  yields a degenerate base at 3B (perplexity 3437 unprotected), so all three arms are
  uninterpretable and we exclude the experiment entirely rather than report its arms. The causal
  test of whether ordering matters therefore remains open, and would need a working
  reimplementation or an instrumented external GPTQ.
- **Task battery.** Six zero-shot tasks, five of them multiple-choice. A perturbation invisible
  here could still harm long-form generation, instruction following or reasoning chains; our claim
  is about the *inference from perplexity*, not a guarantee of preserved capability.
- **Selector calibration.** Our default selector Hessian was small and out-of-distribution; §5.3
  shows this materially changes the 1B result. Numbers in §5.1 use that default.
- **Storage is theoretical.** Weight-only bytes, additive representation, no kernels or latency.
- **Statistics.** Downstream CIs are paired bootstraps over examples; they do not capture variance
  over random channel-selection sets, for which we ran a single seed downstream (three in the
  perplexity sweeps).

## 10 Societal impact

Reliable quantization lowers the memory, energy and hardware cost of deploying language models,
widening access. The failure mode we document cuts both ways: a perplexity-only gate may reject a
usable model, wasting effort, and — more importantly — the converse inference is equally unsafe,
so small perplexity movements should not be read as evidence that a quantized model is safe to
deploy. Practitioners with limited evaluation budgets are the most likely to rely on a single
perplexity number, and are therefore the most exposed. Our results are benchmark findings and are
not guarantees of behaviour in safety-critical deployments.

---

## Appendix A. Provenance

**Compute.** All experiments ran on a single consumer GPU (NVIDIA RTX 4090/5090, 24–32 GB) under
WSL2. Model sizes are 1.24B, 3.09B, 3.21B and 6.74B parameters. The twelve baseline quantizations
total 2.2 GPU-hours, recorded per run as `duration_sec` in `runs/final/llmc/**/summary.json`; the
complete matrix of sweeps, checkpoint exports, reload validations and downstream evaluations is on
the order of 60–80 GPU-hours across the project, including the discarded and re-run experiments
documented below. No model was trained or fine-tuned: every experiment is post-training
quantization and inference.

Every perplexity is read from a committed JSON artifact carrying its own `skip_lm_head`,
`base_group_size`, `seed`, `selector_calib_source` and `selector_hessian_tokens` fields. The five
recorded values for Llama-3.2-3B `greedy@GPTQ` at 2% arise from distinct runs:

| value | run | source |
|---|---|---|
| 51.68 | corrected matrix sweep (skip_lm_head, g128) | `runs/final/sweeps/gptq_llmc/Llama-3.2-3B/greedy/seed-1234/` |
| 51.76 | disk reload of the exported checkpoint | `results/reload_after_e4_3B.log` |
| 55.35 / 55.19 | standalone diagnostic, runtime / materialized | `results/f3_Llama-3.2-3B/` |
| 58.24 | selector Hessian matched to the base (262k tokens) | `results/e5_matchedcalib_Llama-3.2-3B/` |

All are catastrophic relative to the 8.172 base. Tables in §4–§5 use the matrix-sweep value except
§5.3, which reports the matched-calibration run explicitly.

**Llama-3.2-1B `greedy@GPTQ`.** An earlier export of this point reloaded at 203.72 against a sweep
value of 63.95 and was excluded. It was re-exported from scratch and now reloads at **63.83**
(`PASS`, Δ = 0.124); §4.2 uses the repaired checkpoint, and its downstream row is the one reported
in §4.2. The discarded export is retained in the artifacts for audit.

**Excluded experiment.** The ordering intervention (§9) ran to completion but on a degenerate
internal base: unprotected sequential-GPTQ perplexity 3437.3 against an FP16 reference of 7.82.
Its three arms (3437.3 / 3434.8 / 3448.6) are mutually indistinguishable because all are broken,
not because ordering is irrelevant, so no conclusion is drawn from them.

**Environment.** Pinned in `requirements.txt`; `run_manifest.json` records the git commit,
hardware, package versions and the resolved revision of every model. lm-eval-harness v0.4.12.

## Appendix B. Statistics

- **Random controls.** `random` is run with 3 seeds per (model, base, budget) in the perplexity
  sweeps; tables report the seed-1234 value. Downstream, `random@HQQ` was evaluated at one seed;
  additional seeds are configured but not yet evaluated.
- **Downstream CIs.** lm-eval is run with `--log_samples`, giving per-example correctness. For a
  contrast we pair correctness on the same examples and bootstrap the mean difference (2000
  resamples); the macro difference bootstraps within each task and averages.
- **Storage formula.** Weight-only bits per quantized-linear parameter: low-bit values + group
  scales/zeros at the base's true group size + FP16 correction columns + channel indices;
  embeddings, `lm_head` and norms excluded and FP16 in every method.

## References

Badri, H. and Shaji, A. (2023). *Half-Quadratic Quantization of Large Machine Learning Models.*

Dettmers, T., Lewis, M., Belkada, Y. and Zettlemoyer, L. (2022). *LLM.int8(): 8-bit Matrix
Multiplication for Transformers at Scale.* NeurIPS.

Dettmers, T., Svirschevski, R., Egiazarian, V., Kuznedelev, D., Frantar, E., Ashkboos, S., Borzunov,
A., Hoefler, T. and Alistarh, D. (2023). *SpQR: A Sparse-Quantized Representation for Near-Lossless
LLM Weight Compression.*

Frantar, E. and Alistarh, D. (2022). *Optimal Brain Compression: A Framework for Accurate
Post-Training Quantization and Pruning.* NeurIPS.

Frantar, E., Ashkboos, S., Hoefler, T. and Alistarh, D. (2023). *GPTQ: Accurate Post-Training
Quantization for Generative Pre-trained Transformers.* ICLR.

Gao, L. et al. (2024). *A Framework for Few-Shot Language Model Evaluation* (lm-evaluation-harness).

Jin, R. et al. (2024). *A Comprehensive Evaluation of Quantization Strategies for Large Language
Models.*

Kim, S. et al. (2023). *SqueezeLLM: Dense-and-Sparse Quantization.*

Lee, C., Jin, J., Kim, T., Kim, H. and Park, E. (2023). *OWQ: Outlier-Aware Weight Quantization for
Efficient Fine-Tuning and Inference of Large Language Models.* AAAI.

Lin, J., Tang, J., Tang, H., Yang, S., Dang, X. and Han, S. (2024). *AWQ: Activation-aware Weight
Quantization for LLM Compression and Acceleration.* MLSys.

Merity, S., Xiong, C., Bradbury, J. and Socher, R. (2017). *Pointer Sentinel Mixture Models*
(WikiText-2). ICLR.

Paperno, D. et al. (2016). *The LAMBADA Dataset: Word Prediction Requiring a Broad Discourse
Context.* ACL.

Zhao, Y. et al. (2024). *Atom: Low-bit Quantization for Efficient and Accurate LLM Serving.* MLSys.
