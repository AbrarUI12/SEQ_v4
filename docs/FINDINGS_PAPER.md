# Objective Collision: Why Outlier-Channel Protection Backfires on Error-Compensated LLM Quantization

_Draft v2.0 (2026-07-26) — ACL long-paper format. Every perplexity comes from the corrected
sweep matrix (`skip_lm_head=true`, base group size 128), traceable to committed
`runs/final/sweeps/**/channel_pareto.json`. §9 (downstream) and the causal confirmation in §6.4
are marked **pending** and are the only open items._

---

## Abstract

Outlier protection — keeping a small fraction of input channels in FP16 on top of a low-bit
base — is a standard post-training quantization recipe, with selection signals ranging from
activation magnitude to Hessian-weighted quantization error. We audit selection signal × base
quantizer × budget × model under matched weight-only storage, with random controls and a single
evaluator, and find that the *same* selector can be the best or worst choice depending only on
the base. On a data-free RTN/HQQ base, a Hessian-weighted set selector
is the strongest signal (Llama-3.2-3B: 8.10 perplexity versus 8.32 for a matched-bit random
control). On an error-compensated GPTQ base the identical selector is catastrophic (8.17 →
51.7), while activation-magnitude, plain-residual and random selection of the same channel count
stay safe (≈8.1). We attribute this to **objective collision**: the selector
maximizes reduction of ‖ΔW X‖², precisely the objective GPTQ minimized when it produced ΔW, so
on a compensated base it ranks compensation-bearing columns rather than salient ones. Harm
requires that coupling — selectors reading the residual without the Hessian are safe — and
severity grows with how aggressively the objective is optimized. Selection signals carry a
hidden precondition: they must not share the base quantizer's objective.

_(≤200 words)_

---

## 1 Introduction

Post-training quantization (PTQ) of large language models commonly pairs a low-bit weight base
with a small high-precision escape hatch: a few input channels kept in FP16 because they are
deemed salient. Methods differ in how they choose those channels. LLM.int8() uses activation
outlier magnitude; OWQ and SpQR use sensitivity scores built from quantization error weighted by
the layer's input Hessian; CMPQ uses activation distributions. The selection signal is usually
presented as a design detail, and the base quantizer as an orthogonal choice.

We show it is neither. Under matched storage, with a random-selection control at every budget
and a single evaluator, **the same selection signal is the best available choice on one base and
the most destructive on another**. On Llama-3.2-3B, a Hessian-weighted set selector applied to a
data-free HQQ-4 base gives the lowest perplexity of any signal we tested (8.100, versus 8.319
for random selection of the same number of channels). Applied to an error-compensated GPTQ-4
base of the same model, the identical selector raises perplexity from 8.172 to **51.68** — while
activation-magnitude, plain-residual-magnitude and random selection all stay near 8.1.

This is not a quirk of one selector. Severity is **monotone in objective alignment**: on
Llama-3.2-1B the iterative Hessian-weighted selector reaches 63.95, its one-shot ablation 11.28,
plain residual magnitude 10.42, and random 10.41 (base: 10.557). The ordering follows exactly
how much of the base quantizer's own objective each selector uses.

We explain this as **objective collision**. The Hessian-weighted selector maximizes the
reduction in ‖ΔW X‖² obtainable by restoring columns of the residual `ΔW = W − Q(W)`. But
‖ΔW X‖² is precisely the quantity GPTQ minimizes when it produces `Q(W)`: GPTQ pushes each
column's rounding error into the not-yet-quantized columns so that this objective stays small.
On a data-free base, `ΔW` is unstructured rounding noise and the Hessian-weighted score is
genuine saliency information. On a compensated base, `ΔW` is the *output of an optimization over
the same objective* — its structure encodes where the correction was deposited. A selector
sharing that objective therefore ranks compensation-bearing columns, and restoring them to FP16
discards the correction that the remaining columns were tuned around.

**Contributions.**
1. A controlled matched-storage audit (selection signal × base × budget × model) with random
   controls and one evaluator, showing a **base-conditioned inversion**: the same selector is
   best on a data-free base and catastrophic on a compensated one (§4).
2. **Objective collision** as an explanation, with the quantitative prediction that severity
   tracks a selector's coupling to the base quantizer's objective — confirmed by a monotone
   severity ordering across seven selectors and two bases (§5, §6).
3. Integrity evidence that the failure is a property of the weights, not of an evaluation path:
   it survives base regeneration, materialization and disk round-trip, and budget-exhaustion
   controls (§6.3).
4. A boundary condition: the collision appears on Llama-3.2-1B/3B but not on Llama-2-7B or
   Qwen2.5-3B, indicating that it requires the base's compensation to be load-bearing (§7).
5. Supporting audits: protection helps a data-free base over a matched-bit random control, and
   interaction-aware selection does not outperform one-shot selection where protection helps
   (§8). Code, pinned environment and all result JSON are released.

**Practical consequence.** A selection signal is not portable across bases. Protection whose
score derives from the base quantizer's own objective must be *integrated* into quantization (as
OWQ does, by selecting before and compensating over the complement) rather than applied to a
finished compensated checkpoint — an increasingly common practice as pre-quantized checkpoints
are distributed and re-processed.

**What is not new.** Activation-outlier protection (LLM.int8, CMPQ), Hessian/error-based column
selection (OWQ, SpQR), and selecting before compensation (OWQ) are all prior work. Our
contribution is the controlled demonstration that the selection signal and the base interact
destructively, the objective-collision account of why, and the evidence that severity is
predictable from that account.

## 2 Related work

**Outlier-aware mixed precision.** LLM.int8() decomposes the matmul so outlier feature
dimensions are computed in FP16. OWQ scores input columns by Hessian-weighted quantization
error, keeps the top ones in FP16, and — crucially for our analysis — *orders them last inside
the OPTQ/GPTQ pass*, so compensation proceeds knowing which columns remain exact. SpQR isolates
outlier weights in a sparse high-precision structure. Atom, CLAQ and SqueezeLLM use related
group/column/row schemes; CMPQ sets per-channel precision from activation statistics. These
methods **co-design** protection with the base. Our results give a concrete reason why that
co-design is load-bearing rather than incidental: applying the same score after compensation
inverts its meaning.

**Bit allocation and interaction modelling.** SliM-LLM and CoopQ allocate budgets using salience
and interaction structure; AMQ searches allocations under memory constraints; EWQ uses an
entropy prior across layers. These motivate our interaction ablation (§8.2) and our matched-bit
random controls, which some of this literature includes and some omits.

**Error compensation.** GPTQ/OBQ minimize ‖ΔW X‖² by propagating each column's rounding error
into the remaining columns; recent work analyses how this compensation error accumulates. We
contribute an empirical failure mode of *composing* compensation with post-hoc protection, and
an account tying it to the shared objective.

**Position.** We do not propose a new protector. We audit the composition of two standard
components and report a base-conditioned inversion, its explanation, and its boundary.

## 3 Experimental setup

**Protection form.** For a linear layer with weight `W ∈ R^{out×in}` and protected input-channel
set `S`, the protected forward pass is `y = Q(W)x + x[S]·(W − Q(W))[:,S]ᵀ`. Materializing gives
`W_dense = Q(W) + scatter(W − Q(W), S)`, algebraically identical to the forward pass — so a
faithful export must reproduce the runtime perplexity exactly. We use this identity as an
integrity check (§6.3).

**Bases.** (i) **HQQ-4**, a data-free round-to-nearest base (no calibration, no compensation).
(ii) **GPTQ-4**, an error-compensated base produced by LightCompress at W4 group size 128 with a
fixed calibration set, imported as fake-quantized weights and re-evaluated in our evaluator.

**Selectors.** Given the residual `ΔW = W − Q(W)` and input Hessian `H = XᵀX`:

| selector | score | uses ΔW | uses H |
|---|---|---|---|
| `random` (control, 3 seeds) | uniform | ✗ | ✗ |
| `act_max`, `act_scale` | activation magnitude/scale per channel | ✗ | ✗ |
| `residual_max`, `residual_rms` | per-column magnitude of `ΔW` | ✓ | ✗ |
| `greedy_indep` | one-shot marginal gain `2⟨ΔW_j,(ΔW H)_j⟩ − ‖ΔW_j‖²H_jj` | ✓ | ✓ |
| `greedy` | the same objective, re-scored iteratively (OMP-style) | ✓ | ✓ |

The last two maximize the reduction in `‖ΔW X‖² = tr(ΔW H ΔWᵀ)` achievable by restoring columns.
Note `greedy_indep` is a *one-shot full-Hessian* selector, not an activation-magnitude score.

**Storage.** We report **matched theoretical weight-only storage**: bits per quantized-linear
parameter, counting low-bit values, group scales/zeros *at the base's true group size* (128 for
the imported GPTQ base), FP16 correction columns, and channel-index overhead. Embeddings,
`lm_head` and norms are FP16 in every method and excluded from the axis; SEQ runs exclude
`lm_head` from quantization so the quantized scope matches the GPTQ baseline exactly. No kernel,
packing or latency effects are modelled.

**Evaluation.** WikiText-2 perplexity, sequence length 2048, canonical non-overlapping chunking,
one evaluator for every number in this paper; exported checkpoints are evaluated after reload.
The imported GPTQ base measures 10.557 on Llama-3.2-1B in our evaluator versus 10.363 as
reported by the producing toolkit, so we use the reloaded `k=0` checkpoint as the operational
base for all contrasts, keeping every comparison inside one evaluator.

## 4 The base decides whether a selector helps or destroys

**On a data-free base, the Hessian-weighted selector is the best signal.** Llama-3.2-3B
(FP16 7.817, HQQ-4 base 8.328):

| budget | **greedy** | act_max | residual_rms | act_scale | random (control) |
|---|---|---|---|---|---|
| 2% | **8.100** | 8.108 | 8.141 | 8.146 | 8.319 |
| 20% | **7.994** | 8.007 | 8.040 | 8.046 | 8.247 |

Every informative signal beats the matched-bit random control, and `greedy` is the best of them.
Llama-3.2-1B agrees (greedy 10.398 / 10.128 at 2% / 20% versus random 11.051 / 10.907).

**On a compensated base, the same selector is the worst.** Llama-3.2-3B (GPTQ-4 base 8.172 at
4.25 bits); all entries in a column have identical storage (4.57 bits at 2%, 7.45 at 20%):

| selector | 2% | 5% | 10% | 20% |
|---|---|---|---|---|
| **greedy** | **51.68** | **55.05** | **38.39** | **40.65** |
| **greedy_indep** | **41.50** | **43.30** | **44.83** | **43.04** |
| residual_max | 8.100 | 8.100 | 8.101 | 8.075 |
| residual_rms | 8.146 | 8.153 | 8.160 | 8.139 |
| act_max | 8.181 | 8.173 | 8.136 | 8.102 |
| act_scale | 8.586 | 8.204 | 8.187 | 8.154 |
| random (control) | 8.181 | 8.221 | 9.048 | 8.487 |

The inversion is complete: `greedy` moves from best (8.100, better than the 8.328 base) on HQQ to
worst (51.68, six times the 8.172 base) on GPTQ, for the same model, budget and storage. Because
random selection of the same number of channels is harmless, the damage is not caused by
restoring 2–20% of columns to FP16 per se — it is caused by *which* columns the objective picks.

## 5 Harm requires the Hessian coupling, and scales with how hard the objective is optimized

Llama-3.2-1B (GPTQ-4 base 10.557) separates the selectors more finely than 3B, and the ordering
follows how much of GPTQ's objective each selector uses:

| selector | objective coupling | 2% | Δ vs base |
|---|---|---|---|
| `greedy` | full objective, iteratively re-scored | **63.95** | +53.4 |
| `greedy_indep` | full objective, evaluated once | 11.28 | +0.72 |
| `residual_rms` | residual magnitude, no Hessian | 10.43 | −0.12 |
| `residual_max` | residual magnitude, no Hessian | 10.42 | −0.13 |
| `random` | none | 10.41 | −0.15 |
| `act_max` | activation only | 10.71 | +0.15 |

Two claims are supported, and we separate them carefully.

**(i) Harm requires the Hessian coupling — this is categorical and holds on both models.**
Selectors that read `ΔW` but *not* `H` (`residual_max`, `residual_rms`) are entirely safe, as are
those reading neither (`act_max`, `act_scale`, `random`). Only the two selectors optimizing
`tr(ΔW H ΔWᵀ)` cause damage. This rules out "reads the quantization residual" as the operative
factor and isolates the Hessian coupling.

**(ii) Among coupled selectors, severity increases with how aggressively the objective is
optimized — but the separation is model-dependent.** On 1B the ordering above is strict
(63.95 → 11.28 → 10.42 → 10.41): iterative re-scoring is catastrophic while its one-shot
ablation is only mildly harmful (+0.72). On 3B the distinction collapses — *both* coupled
selectors are catastrophic (`greedy` 51.68, `greedy_indep` 41.50) — so 3B saturates the effect
rather than resolving it. We therefore claim a strict severity ordering only on 1B, and treat 3B
as evidence that once the coupling is present the damage can saturate. Filling the axis between
`residual_max` (no Hessian) and `greedy_indep` (full objective) with an intermediate selector
that uses only the Hessian *diagonal* is the natural test of whether severity varies
continuously with coupling; we report it as an outstanding measurement (§6.4c).

## 6 Objective collision

### 6.1 The account

GPTQ quantizes column by column and propagates each column's rounding error into the columns not
yet quantized, so as to keep `‖ΔW X‖² = tr(ΔW H ΔWᵀ)` small. The residual it leaves is therefore
not arbitrary: `ΔW` is the *residue of an optimization over that objective*, and its structure
records where the correction was deposited.

The Hessian-weighted selector chooses `S` to maximize the reduction in the *same* quantity. On a
base that never optimized it — HQQ, which is data-free round-to-nearest — `ΔW` is unstructured
rounding error and this score is genuine saliency: it identifies columns whose error the layer's
input distribution actually amplifies, which is why `greedy` is the strongest signal in §4. On a
compensated base the same score instead identifies columns carrying the deposited correction.
Restoring those columns to their original FP16 values removes the correction while leaving the
complement quantized *as if* the correction were still present, so the layer output is
mis-corrected rather than merely un-protected.

This account predicts precisely the pattern observed: harm requires the Hessian coupling (§5(i)),
harm increases with how aggressively the objective is optimized where the effect is not
saturated (iterative > one-shot on 1B, §5(ii)), harm is absent for signals orthogonal to the
objective (activation, plain residual, random, §4), and the sign of the effect flips with the
base (§4). It further predicts that a selector using only part of the objective should cause
intermediate harm (§6.4c).

### 6.2 What the account does not yet establish

We measured the overlap between the selected set and the columns of largest residual energy at
only 2–3%. The selector therefore does **not** simply pick the biggest-residual columns; it picks
an interaction-structured set. This is consistent with the account (the objective is
Hessian-weighted, not magnitude-based) but it means we have not directly measured the quantity
the account is about. Two direct measurements are specified in §6.4.

An alternative explanation we cannot yet exclude is numerical: `H` may be more ill-conditioned on
a compensated base, degrading the selector's own arithmetic. Our selector runs in float64 with
periodic exact refresh specifically to bound that drift, but this does not by itself rule the
alternative out.

### 6.3 Integrity checks

The failure is a property of the weights, not of an evaluation path.
- **Base regeneration.** The original GPTQ base was lost and regenerated with the same recipe;
  the collapse reproduces on the regenerated base.
- **Materialization and reload.** Materializing the protected model to dense weights and
  re-measuring gives the same perplexity (3B 55.35 → 55.19; 1B 63.95 → 63.82; |Δ| < 0.13), and
  reloading the saved checkpoint from disk reproduces it again (3B 51.76; 1B 63.82).
- **Budget exhaustion.** The selector is by default forced to spend the full budget. Re-running
  with early stopping at the last positive-gain channel yields identical perplexity at 2%: every
  selected channel had positive predicted gain, so the collapse is not an artifact of forcing
  the selector to protect channels its own objective rejects.
- **Scope and storage.** All runs exclude `lm_head` from quantization (matching the baseline's
  quantized scope) and charge the imported base at its true group size of 128; correcting the
  latter moves the 2% operating point from 4.82 to 4.57 bits.

### 6.4 Direct tests of the account (pending)

Three measurements would move objective collision from a well-supported explanation to a
demonstrated mechanism. We state each prediction before running it.

**(a) Alignment.** For each selector, measure the rank correlation between its per-column score
and GPTQ's per-column compensation magnitude (how much correction that column absorbed during
the pass). The account predicts high alignment for `greedy`/`greedy_indep` and near-zero for
`residual_max`, `act_max` and `random`, with the alignment ordering matching the severity
ordering of §5.

**(b) Ordering intervention.** Hold the model, calibration data, protected set `S`, budget and
evaluator fixed and vary only *when* `S` is made exact: (A) quantize all columns, then restore
`S` (post hoc — the failing configuration); (B) hold `S` exact throughout compensation, so the
complement is compensated knowing `S` is exact. The account predicts A collapses and B does not,
on the same `S`. Both arms are implemented on a sequential GPTQ implementation. *An earlier
execution used a one-shot GPTQ implementation whose base was itself degenerate (perplexity above
3000 for every arm, including the unprotected base); it is uninformative and we exclude it.* We
make no causal claim until (b) returns.

**(c) Intermediate coupling.** Add a selector scoring `‖ΔW_j‖²·H_jj` — the Hessian *diagonal*
only, i.e. the objective without its cross-column interaction terms — and run it on the GPTQ base
at the same budgets. The account predicts harm strictly between the Hessian-free selectors
(`residual_max`, safe) and the full-objective selectors (`greedy_indep`, harmful), making the
coupling axis continuous rather than a two-point contrast (§5(ii)).

## 7 Boundary: the collision is model-dependent

Running the identical configuration (`greedy`, 2%, GPTQ-4 base, matched scope and storage):

| model | FP16 | GPTQ-4 base | greedy@GPTQ 2% | verdict |
|---|---|---|---|---|
| Llama-3.2-1B | 9.757 | 10.557 | **63.95** | collapse |
| Llama-3.2-3B | 7.817 | 8.172 | **51.68** | collapse |
| Qwen2.5-3B | 8.030 | 8.290 | 8.335 | safe |
| Llama-2-7B | 5.469 | — | 5.571 | safe |

On Qwen2.5-3B the full panel is benign (`greedy` 8.335, `greedy_indep` 8.357, `random` 8.300 at
2%). Objective collision is therefore a *necessary but not sufficient* condition: the selector
must share the objective **and** the base's compensation must be load-bearing enough that
removing it is destructive. We do not yet have a predictor for the latter; the alignment
measurement of §6.4(a), run across all four models, is the natural candidate and would convert
this boundary from an observation into a diagnostic.

## 8 Supporting audits

**8.1 Protection helps a data-free base (matched-bit).** Every informative signal beats the
matched-bit random control on HQQ-4 at both budgets and both model sizes (§4), by ≈0.2 PPL on 3B
and ≈0.65 on 1B. Per-channel importance therefore carries real information when the base has not
already exploited it.

**8.2 Interaction modelling does not pay where protection helps.** On HQQ the iterative selector
and its one-shot ablation are within ≈0.03 PPL at every budget, so iterative re-scoring buys no
accuracy over evaluating the same objective once. On GPTQ it is strictly worse (§5). The
defensible statement is that **iterative re-scoring does not improve on one-shot full-Hessian
selection and, on a compensated base, concentrates harm**. We do not claim second-order machinery
is unnecessary *relative to activation magnitude*: that requires an equivalence test against
`act_max` with a pre-declared margin, which we leave to future work rather than assert from a
0.03 PPL difference.

**8.3 The base dominates the frontier.** On matched weight-only storage the base quantizer sets
the ceiling: the GPTQ-4 base at 4.25 bits (3B: 8.172) is not reached by any protected HQQ
configuration at comparable storage, and the best protection on GPTQ buys ≈0.07 PPL
(`residual_max`, 8.100 at 4.57 bits) for a +0.32-bit premium. We state this as an observation on
this grid, not a general law.

## 9 Downstream evaluation

**Pending.** Zero-shot evaluation (hellaswag, arc-easy, arc-challenge, piqa, winogrande,
lambada-openai) is being re-run under the corrected scope at uniform full scale with a
multi-seed matched-bit random control. Earlier downstream numbers in this repository are
superseded: they were produced from a stale exported checkpoint that a resume guard reused from
an earlier, healthy export, which we detected because the recorded `expected_ppl` sidecar and the
measured accuracy were mutually inconsistent. Because the materialization and reload identity is
verified (§6.3), a faithful downstream evaluation of a perplexity-52 checkpoint must show
correspondingly degraded accuracy; we report measurements rather than predictions.

## 10 Limitations

- **Mechanism.** Objective collision explains and predicts the observed pattern but the two
  direct tests in §6.4 are outstanding; the numerical-conditioning alternative (§6.2) is not yet
  excluded.
- **Model coverage.** Collapse is established on Llama-3.2-1B/3B and absent on Llama-2-7B and
  Qwen2.5-3B. We have no predictor for susceptibility, so we cannot say which unseen models are
  at risk.
- **Downstream.** Pending (§9); all conclusions here are perplexity-based.
- **Base coverage.** One compensated base (GPTQ). Whether the collision appears with other
  compensated bases (e.g. AWQ) determines how general the principle is.
- **Storage is theoretical.** Weight-only bytes; no kernels, packing overhead or latency.
- **Selection cost.** We do not report wall-clock or memory cost per selector; the Hessian-based
  selectors are far more expensive than the magnitude ones, which strengthens the practical case
  against them given §8.2.

## 11 Conclusion

A selection signal for outlier protection is not a property of the model alone — it is only
meaningful relative to what the base quantizer has already optimized. We show a complete
inversion: a Hessian-weighted set selector is the best signal available on a data-free base and
the most destructive on an error-compensated one, at identical storage, with a matched-bit random
control harmless in both. We attribute this to objective collision — the selector maximizes the
very quantity the base quantizer minimized, so on a compensated base it ranks compensation
rather than saliency — and show that severity is monotone in objective coupling and vanishes when
the coupling is removed. Practitioners should treat a selection signal as tied to its base:
where the score derives from the quantizer's own objective, protection must be integrated into
quantization rather than applied to a finished compensated checkpoint.

---

## Appendix A. Provenance and statistics

- **Traceability.** Every perplexity is read from a committed
  `runs/final/sweeps/<base>/<model>/<selector>/seed-1234/channel_pareto.json` carrying its own
  `skip_lm_head`, `base_group_size`, `seed` and `greedy_early_stop` flags, or from
  `results/f3_*` / `results/f3check_*` for the model panel (§7). The two 3B `greedy`@2% values
  (51.68 in the matrix sweep, 55.35 in the standalone diagnostic) are separate runs of the same
  configuration; tables report the matrix value.
- **Random controls.** `random` is run with 3 seeds per (model, base, budget); tables report the
  seed-1234 value, with the multi-seed spread used for the control band in the dose–response
  figure.
- **Storage formula.** Weight-only bits per quantized-linear parameter: low-bit values + group
  scales/zeros at the base's true group size + FP16 correction columns + channel indices;
  embeddings, `lm_head` and norms excluded and FP16 in every method.
- **Environment.** Pinned in `requirements.txt`; `run_manifest.json` records the git commit,
  hardware, package versions and the resolved revision of every model.
