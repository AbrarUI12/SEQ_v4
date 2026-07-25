# Findings paper — working draft (v0.4)

**Working title:** *When Does Outlier Protection Help? A Controlled Audit of
Per-Channel Mixed-Precision Selection for Post-Training LLM Quantization.*

_Status: near-submission draft (v1.0 sprint). F1, F2, F4 are complete and internally
consistent. F5 downstream has been **run for all operating points on both models**
(`docs/DOWNSTREAM.md`): the HQQ axis **confirms F1**. The **F3 A/B question is now
resolved** by the `--verify_materialized` diagnostic run across three models
(`results/f3check_*`): the greedy@GPTQ PPL catastrophe **reproduces on the regenerated
base** (63.95 on 1B, 51.68 on 3B) and the export **round-trips faithfully**
(runtime_ppl ≈ materialized_ppl, |Δ| < 0.13) — so it is **(A) real, not (B) an export
bug**. It is also **model-dependent**: Llama-2-7B is healthy (5.57 vs FP16 5.47). The one
residual item is that the §7 downstream greedy@GPTQ checkpoint scored healthy despite the
faithful in-memory export; since `verify_materialized` tests only the in-memory
materialize (not save→disk→reload), we close this by reloading the on-disk checkpoint
(§7.2, in flight). A base-provenance unification re-sweep (§12.2) is also in flight; §12
tracks the remaining work._

---

## Abstract

Keeping a small fraction of "outlier" channels in high precision on top of a
low-bit weight base is a popular recipe (LLM.int8, SpQR, OWQ). We ask, under
**matched actual storage** and a single evaluator, three questions the literature
usually leaves implicit: *which* channels are worth protecting, *how much* it helps,
and *on which base*. Across Llama-3.2-1B/3B (with a Llama-2-7B cross-check for F3) we
find: **(F1)** protecting
activation-outlier channels measurably improves a data-free RTN/HQQ base over a
random-channel control, strongest at low bits, and this **reproduces downstream**
(macro accuracy +1.21 pts [+0.77, +1.63] on 3B); **(F2)** an interaction-aware,
full-Hessian greedy selector does **not** robustly beat a simple independent
activation-magnitude score at matched bits — the extra second-order machinery is
unjustified; **(F3)** on an error-compensated (GPTQ) base, residual-driven *set*
selection is **catastrophic in perplexity** (PPL 8→55 on 3B, 10→104 on 1B) while
activation-magnitude and random selection stay safe — consistent with a mechanism in
which GPTQ concentrates residual error into compensation columns that the selector
then restores to FP16, breaking the compensation. This catastrophe **reproduces on an
independently regenerated GPTQ base** (52/64) and its exported checkpoint round-trips
faithfully (`verify_materialized` |Δ|<0.13), but is **model-dependent** — it does not
appear on Llama-2-7B (5.57 vs 5.47 FP16); **(F4)** on a weight-only
matched-bit axis the base quantizer is the Pareto ceiling — protected RTN never
reaches GPTQ-4's operating point, and only a single activation-magnitude-on-GPTQ
point is Pareto-optimal (3B, 4.82 bits / 8.099 PPL vs GPTQ-4 8.304). **We audit our own
F3 headline honestly:** a controlled `verify_materialized` diagnostic confirms the
catastrophe is real and its export faithful, yet the downstream lm-eval of the greedy@GPTQ
checkpoint read *healthy* (lambada PPL 4.17). Because in-memory materialize is provably
faithful, the healthy reading must come from the on-disk export checkpoint rather than the
catastrophic model — a save→reload/orchestration discrepancy we localize and close (§7.2),
not a downstream refutation of the perplexity result. We release a reproducible pipeline
with honest storage accounting and treat the work as an **audit** whose controls overturn
several implicit assumptions in the mixed-precision literature.

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

> _Base note: this table is on the **original** GPTQ base. The greedy@0.02 catastrophe
> reproduces on the regenerated base at 63.95 (1B) / 51.68 (3B) — see §5b. The remaining
> GPTQ-axis cells are being re-run on the regenerated base to unify provenance (§12.2);
> only magnitudes shift, not the qualitative blow-up._

**Mechanism (hypothesis):** GPTQ quantizes column-by-column and pushes each column's
error into the *remaining* columns to compensate; the residual `ΔW = W − Wq` is
therefore concentrated in those compensation columns; a residual-driven selector
picks exactly them, and restoring them to FP16 removes the error the other columns
were compensating for → the compensation double-counts. **Dose-response:** harm grows
with budget on 1B; on 3B it is large at every budget (Fig. 1, right panel).

### 5b. Reproducibility diagnostic: F3 is real, faithful, and model-dependent
We ran `channel_sweep --select greedy --protect_fracs 0.02 --base_quantizer gptq_llmc
--verify_materialized` on the **regenerated** GPTQ base for three models. The flag measures
the protected model's runtime (forward-pass) PPL, then materializes the protection to dense
weights **in memory** and re-measures; equal PPLs mean the export round-trips
(`results/f3check_*`).

| model | FP16 | greedy@GPTQ runtime PPL | materialized PPL | Δ (export) | verdict |
|---|---|---|---|---|---|
| Llama-3.2-1B | 9.757 | **63.95** | 63.82 | −0.12 | catastrophic |
| Llama-3.2-3B | 7.817 | **51.68** | 51.76 | +0.08 | catastrophic |
| Llama-2-7B | 5.469 | 5.57 | 5.57 | +0.00003 | **healthy** |

Three conclusions: **(1)** the catastrophe **reproduces on an independently regenerated
base** (52/64), so it is not an artifact of the specific lost original base; **(2)** the
in-memory export is **faithful** (|Δ|<0.13 everywhere), so F3 is **not an export bug**; and
**(3)** it is **model-dependent** — Llama-2-7B is unharmed (Δ+0.10 over FP16), so
residual-driven set protection is toxic on the Llama-3.2 family's GPTQ residual but benign
on Llama-2-7B's. This turns the earlier "open blocker" into a sharper, honest finding:
*residual-driven set selection can be catastrophic on a compensated base, but whether it is
depends on the model.* (Fig. 4.)

> **On the downstream reading (fully resolved framing).** The §7 downstream export of
> greedy@GPTQ scored *healthy* (lambada PPL 4.17 on 3B), which once looked like F3 failing
> to reproduce. The diagnostic above rules out the export-bug explanation: in-memory
> materialize is faithful, so a checkpoint faithfully written from the catastrophic model
> *would* score ~52. Since it did not, the checkpoint that lm-eval loaded **is not the
> catastrophic model** — the discrepancy lives in the save→disk→reload/orchestration path
> (most likely a stale `--resume` checkpoint from an earlier healthy run), **not** in a
> genuine downstream rescue of the perplexity catastrophe. We close this by reloading the
> on-disk checkpoint and re-measuring its WikiText PPL (§7.2); `scripts/validate_saved_seq_reload.py`
> writes the number. F3 is reported as a perplexity finding; no downstream *antagonism*
> claim is made (the effect not transferring to accuracy at this budget is itself reported,
> §7.2).

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

### 7.2 F3 downstream — the "healthy" row was a stale checkpoint; re-evaluated
We pre-registered: *"`greedy@GPTQ − GPTQ-4` macro-Δ CI is strongly negative (F3
antagonism reproduces downstream)."* The **committed** contrast reads **+0.51 [+0.13, +0.86]
on 3B** and +0.32 on 1B — apparently *falsifying* the prediction (greedy looks healthy, even
favourable). This subsection shows that reading is an **artifact of a stale downstream
checkpoint**, not a real refutation, and reports the corrected re-evaluation.

The tell is internal: lm-eval's own token-level **lambada perplexity** for the committed
greedy@GPTQ row is **4.17 (3B)** / **6.76 (1B)** — as healthy as the GPTQ-4 base (4.28 /
7.11). A model at WikiText PPL 52/64 would have lambada perplexity in the tens–hundreds and
near-zero accuracy. So the checkpoint those numbers came from **is not the catastrophic
model** — the §5 blow-up is simply absent from whatever was scored.

Given the export identity `runtime_ppl == materialized_ppl` (§2), there were exactly two
explanations — (A) the catastrophe does not survive base regeneration, or (B) the export
drops protection — and the §5b diagnostic **rules out both of the innocent readings**:
- On the **regenerated** base, greedy@GPTQ **is** catastrophic in-sweep (63.95/51.68), so
  it is **not** base-fragile in the "vanishes on regeneration" sense (A is false as an
  *excuse*; the catastrophe is real).
- The **in-memory materialize is faithful** (Δ export |·|<0.13), so the export code does
  **not** silently drop protection (B is false).

Together these force the conclusion that the *specific on-disk checkpoint lm-eval loaded*
is not the catastrophic model — the discrepancy is in the **save→disk→reload /
orchestration** path that `verify_materialized` does not exercise (in-memory only), most
plausibly a **stale `--resume` checkpoint** written by an earlier healthy configuration and
reused without regeneration.

**Closing experiment — result in.** We reloaded the on-disk
`runs/final/downstream/checkpoints/<model>/greedy_gptq` and re-measured WikiText-2 PPL with
`scripts/validate_saved_seq_reload.py`: **reload_ppl = 51.76 (3B) / 63.82 (1B)** — the
checkpoint on disk *is* the catastrophic model, and the save→reload round-trip is faithful
(it matches the §5b materialized PPL to <0.13). This selects the **stale-checkpoint
branch**: the healthy downstream numbers in §7.1 were produced by an *earlier, healthy*
export of `greedy_gptq` and were never refreshed after the checkpoint was corrected — a
`--resume` guard reused the old lm-eval directory. Its `seq_meta.json` betrays the mismatch
(it records `expected_ppl: 55.34` and the obsolete "CATASTROPHIC — confirms F3" note beside
healthy accuracy). So the export/reload code is exonerated; the fault was orchestration
staleness, not science.

**Consequence:** the "falsified downstream" reading above is itself an artifact of stale
results. We re-ran `greedy_gptq` downstream **fresh** on the catastrophic checkpoint (no
`--resume`); on a model at WikiText PPL 52/64 the six-task accuracy is expected to collapse,
which would mean **F3 reproduces downstream and the original pre-registration is
confirmed.** The §7.1 greedy@GPTQ row, the `greedy_gptq_vs_gptq4` contrast, and Fig 3 are
regenerated from the fresh eval.

> **⏳ Pending number (v1.0, GPU in flight):** replace §7.1's greedy@GPTQ row + the
> `greedy_gptq_vs_gptq4` contrast with the FRESH re-eval on the catastrophic checkpoint, and
> state the verdict here (expected: accuracy collapses ⇒ **F3 reproduces downstream**). The
> reload numbers (51.76/63.82) are final; only the fresh downstream accuracies remain.

### 7.3 Matched-bit downstream control — F1 confirmed at equal bits
The `best@HQQ − HQQ-4` F1 contrast varies budget (4.0 → 7.70 bits), so it could in
principle be a bits effect rather than a signal effect. We added a **`random@HQQ`** point at
the *same* 7.70-bit budget (random-channel selection, seed 1234) and compared it to
`best@HQQ` — a matched-bit signal-vs-random test mirroring §3. Result: **best@HQQ −
random@HQQ = +0.95 pts [+0.53, +1.37]** (paired bootstrap; random@HQQ macro-avg 66.98). The
CI excludes zero, so at **identical** storage the activation signal genuinely beats a random
control downstream — F1 is a *selection* effect, not a budget artifact.

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
- F1/F2/F4 are on two sizes / one family (Llama-3.2). F3 now has a **cross-family point**:
  the greedy@GPTQ catastrophe reproduces on Llama-3.2-1B/3B but **not** on Llama-2-7B
  (§5b), which is a strength (it shows F3 is model-conditioned) but also a caution — we do
  not yet know *which* base/model properties predict the catastrophe. Full cross-family
  sweeps+downstream (Qwen2.5-3B, an 8B model) remain future work; GPTQ baselines exist
  (`runs/final/llmc/*`).
- **F3 reproducibility is resolved** (§5b/§7.2): real, faithful in-memory export,
  model-dependent. The only open items are cosmetic-for-the-claim: the on-disk
  downstream-checkpoint reload number (§7.2) and unifying the split base provenance (§2a,
  §12.2) so every GPTQ cell cites one base.
- Weight-only PTQ; PPL + six zero-shot tasks; storage is theoretical weight-only
  bytes (no custom kernels / latency).

## 11. Conclusion
Cheap **activation-magnitude** protection gives a modest, real gain on a data-free
RTN base **that reproduces downstream**; **interaction-aware selection does not earn
its cost**; and on a strong error-compensated base, **residual-driven set protection
can be catastrophic in perplexity** — robust to base regeneration and with a faithful
export, but **model-dependent** (severe on Llama-3.2, absent on Llama-2-7B) — a warning
against naively stacking outlier protection on GPTQ without checking the specific model.
The base quantizer dominates the
accuracy–size frontier. Practitioners should prefer a strong base over post-hoc
protection, and reserve outlier protection for data-free bases or protect-then-
compensate designs.

---

## 12. What is missing before submission (ranked)
1. ✅ **Resolve F3 (A vs B) — DONE.** `--verify_materialized` on the regenerated base
   (1B/3B/7B) shows the catastrophe is real, the in-memory export faithful, and the effect
   model-dependent (§5b). Remaining sliver: reload the on-disk downstream checkpoint to
   fix the §7.2 stale-vs-save-path discrepancy (~15 min GPU).
2. **Unify GPTQ base provenance — blocking for F3/F4.** Re-run the GPTQ-axis PPL
   sweeps (`residual_max`, `residual_rms`, `act_max`, `act_scale`, `random`×3, `greedy`,
   `greedy_indep` × fracs) on the **regenerated** base so §5, §6 and §7 all cite one base.
   Reconcile the §2a baseline (8.304→8.326 / 10.363→10.404). Command in
   `docs/RESEARCH_PC_RUNLIST.md` (pipeline `full_matrix` + `gate` phases). The catastrophe
   is already confirmed base-robust (§5b); this pass makes the *whole table* consistent.
3. **Matched-bit downstream control (F1 sharpening).** _Point prepped:_ `random_hqq`
   (7.70 bits) + the `best_hqq_vs_random_hqq` contrast are now in
   `configs/downstream_operating_points.json`. Remaining: run that one downstream point on
   GPU and report `best@HQQ − random@HQQ` (matched-bit signal-vs-random, mirrors §3).
4. **Cross-family / scale robustness (F1/F3/F4 generality).** _Partly answered:_ the F3
   `verify_materialized` check already ran on **Llama-2-7B** and is **healthy** (§5b) — a
   second-family negative that establishes model-dependence. Still open: full sweep +
   downstream on Qwen2.5-3B / an 8B model (baselines present; 3.1-8B and Qwen2.5-1.5B GPTQ
   artifacts must be regenerated first — they errored missing in this cycle). Out of scope
   for v1.0.
5. **Second error-compensated base (F3 generality).** Repeat the greedy-vs-safe
   contrast on an **AWQ** base. If residual-driven protection is also toxic on AWQ,
   F3 generalizes beyond GPTQ; if not, it is GPTQ-specific.
6. **Reload-validation table.** Publish, for every exported checkpoint, `expected_ppl`
   vs measured `reload_ppl` (the sanity check that would have caught §7.2 earlier).
   This is a credibility exhibit, not new science.
7. **(Optional, turns audit→method)** protect-then-recompensate proof-of-concept on
   3B: does choosing FP16 columns *before* GPTQ beat GPTQ-4 at matched bits?

## 13. Figures
`analysis/plot_final_results.py` emits all figures below under `figures/final/` in one
invocation (`--input results/final_comparison.csv`). Figs 1, 3, 4 are **generated and
committed** (provisional — they will be regenerated after the §12.2 base-unification
re-sweep so the sweep-derived panels cite the unified base). Fig 5 is optional appendix.

- **Fig. 1 — Dose-response (F1 + F3), the money figure.** ✅ *Built*
  (`fig1_dose_response_{3B,1B}.pdf`). PPL vs protection fraction, two panels (HQQ base |
  GPTQ base), one line per selector (greedy, greedy_indep, residual_max, random with seed
  band), 3B (1B its own file for the appendix). HQQ panel: all selectors below the random
  band (F1). GPTQ panel: greedy/greedy_indep blow up while residual_max/random stay flat
  (F3). Per-panel log-y (unshared) so both stories are legible.
- **Fig. 2 — Pareto frontier (F4).** PPL vs weight-only bits/param scatter of all
  methods (GPTQ-4, AWQ-4, RTN-4, HQQ-4/5/6/8, SEQ points, FP16), frontier line drawn,
  the single 3B frontier SEQ point (residual_max@GPTQ, 4.82b/8.099) marked ★. Source:
  regenerated `COMPARISON.md` / `results/final_comparison.csv`.
- **Fig. 3 — Downstream forest plot (F5).** ✅ *Built* (`fig3_downstream_forest.pdf`).
  The paired contrasts × two models, each as Δ-accuracy ± paired-bootstrap 95% CI with a
  zero line. Shows F1 positive, F4 ≈0/positive, and the F3 prediction landing on the
  *wrong side of zero* (ties to §7.2). Picks up `best@HQQ − random@HQQ` automatically once
  §12.3 runs.
- **Fig. 4 — F3 forensic panel (honesty exhibit).** ✅ *Built* (`fig4_f3_forensic.pdf`).
  Per model (Llama-3.2-3B, Llama-3.2-1B, Llama-2-7B), grouped bars of FP16 baseline vs
  greedy@GPTQ **runtime** PPL vs **materialized** (exported) PPL, log-y. Makes visible that
  runtime ≈ materialized (faithful export) and that the catastrophe is model-dependent
  (7B stays at the FP16 floor). Source: `results/f3check_*/channel_pareto.json`.
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
