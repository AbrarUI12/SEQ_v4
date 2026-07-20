# Track B — Improving SEQ as a *method*: everything done, and what's left

_Companion to `docs/PROJECT_STATUS_AND_ROADMAP.md` (§4) and the audit paper
`docs/FINDINGS_PAPER.md`. Track A = the findings/audit paper (the realistic
publication). **Track B = can SEQ be turned into a method that beats a strong
baseline at matched bits?** This doc is the single place that records every Track B
attempt, its verdict, and the one lead still worth running._

**One-line status:** Track B is **mostly closed by our own data**. Three families of
ideas were tried; two are dead and one (interaction-aware selection) is not just
dead but *toxic* on a strong base. The single surviving positive is cheap
**activation-magnitude protection on a GPTQ base** (a Pareto point on 3B), and the
only unexplored idea with real headroom is **protect-then-recompensate**.

---

## 0. What "improving SEQ" has meant, in three eras

SEQ = keep a small fraction of "important" weights/channels in higher precision on
top of a low-bit base. Track B asked, era by era, *which unit to protect, chosen by
what signal, on which base* — always trying to beat the base at matched storage.

| Era | Unit protected | Signal / policy | Base | Where | Verdict |
|---|---|---|---|---|---|
| **A. Variants v0–v5** | modules / projections | entropy percentiles, tier policies, risk score | bitsandbytes NF4 | `seq_variants/` | superseded — heuristic, not matched-bit-controlled |
| **B. Module-level allocation (runs 1–6)** | whole modules | entropy / Hessian / reconstruction proxies | HQQ/RTN | `docs/FINDINGS_run{1..6}.md` | **dead** — proxies decouple from PPL; ≤ uniform (EWQ prior art) |
| **C. Per-channel + greedy set selection** | input channels | act_max/scale, residual_rms/max, greedy OMP | HQQ **and** GPTQ | `seq_core/channel_sweep.py` | **audited** — F1 helps on HQQ; F2 interactions don't pay; **F3 toxic on GPTQ** |

Everything below expands these three and states exactly what is DONE vs LEFT.

---

## 1. Era A — `seq_variants/` v0–v5 (DONE; superseded)

Six hand-designed policies over an NF4 bitsandbytes base, run through
`run_compare_matrix.py --experiments_file experiments_seq_variants.yaml`
(`seq_variants/README.md`, `seq_variants/smoke_test_variants.py`):

- **v0** (`seq`/`seq_v0`) — baseline SEQ: weight+activation entropy, hard AND/XOR
  tier policy, 0.90/0.90 thresholds, protection floors (lm_head, first/last blocks,
  attn out-proj).
- **v1** — adds `attention_mask` masking of padded positions in the entropy passes.
- **v2** — v1 with 0.80/0.80 entropy thresholds (more protection).
- **v3** — v2 + all attention projections ≥ INT8.
- **v4** — v3 + MLP gate/down projections ≥ INT8.
- **v5** — risk-score policy: `risk = max(weight_rank, act_rank)`; FP16 ≥ 0.90,
  INT8 ≥ 0.65, INT4 otherwise.

**Verdict:** these established the SEQ machinery and the tier vocabulary, but they
are **percentile heuristics without matched-bit controls or a random baseline** —
so they cannot support a method claim. Era B/C replaced them with controlled,
matched-storage experiments. Kept in-repo for provenance and the smoke test; **not**
a basis for further work.

## 2. Era B — module-level entropy/proxy allocation, runs 1–6 (DONE; dead)

Systematic study of whether a *proxy* (activation/weight entropy, Hessian diagonal,
reconstruction error) can rank whole modules for bit allocation better than uniform.
Consolidated evidence in `docs/FINDINGS_run{1..6}.md` and
`analysis/findings_summary.json`.

**Verdict (dead):** the proxies **decouple from measured PPL**, and proxy-guided
module allocation is **≤ uniform** at matched bits. This is also essentially **EWQ**
(entropy-weighted quantization) prior art. → folded into the Track A paper as the
"proxies mislead" section (W3); **do not revisit** as a method.

## 3. Era C — per-channel protection + interaction-aware greedy (DONE; audited)

The current SEQ core (`seq_core/channel_sweep.py`, `channel_protect.py`,
`greedy_select.py`, `signals.py`). Protect the top-k% **input channels** (LLM.int8
column split: `y = Q(W)x + x[S]·(W−Q(W))[:,S]ᵀ`), choosing S by a signal or a
full-Hessian greedy OMP, on both a data-free **HQQ** base and a validated
LightCompress **GPTQ-4** base, at matched weight-only bytes with a random control.
Numbers: `docs/PROJECT_STATUS_AND_ROADMAP.md §1b–1d`.

- **F1 (positive):** on HQQ, every signal beats random, monotone in budget → the
  useful signal is **activation, not weights**.
- **F2 (negative):** `greedy` beats `greedy_indep`/`residual_max` by only
  ~0.02–0.03 PPL, and `greedy_indep` sometimes wins → **the interaction-aware
  full-Hessian machinery does not earn its cost.** The interaction-aware greedy
  selector — the thing Track B was *most* betting on — is dead.
- **F3 (toxic, headline):** on the GPTQ base, residual-driven set selectors blow up
  (greedy 43–107, greedy_indep 15–47 PPL) while activation-magnitude `residual_max`
  stays safe. **Protection is antagonistic to error compensation.**

**Net for Track B:** the one surviving positive is **cheap activation-magnitude
protection (`residual_max`/`act_max`) on the GPTQ base** — a single Pareto-frontier
point on 3B (4.82 bits / 8.099 PPL vs GPTQ-4 8.304; roadmap §1e). Today's downstream
staging (`scripts/run_downstream_eval.sh`, points `resmax_gptq` vs `gptq4`) is what
confirms this holds at the *task* level, not just PPL.

---

## 4. The one live lead — **protect-then-recompensate** (LEFT; the only path to a method paper)

F3 is *caused* by protecting **after** GPTQ has already pushed its quantization error
into the surviving columns. The fix is to reverse the order so protection and
compensation cooperate:

1. **Pick the protected set S per layer first** (act_max, or residual on the FP16
   residual — *not* on the post-GPTQ residual).
2. **Run GPTQ error-compensation over the complement only**, keeping S in FP16 (feed
   LightCompress a per-layer column mask, or a custom GPTQ loop that skips S).
3. **Compare at matched weight bits** to plain GPTQ-4 and to naive post-hoc
   protection (the F3 catastrophe).

**Hypothesis:** matched-bit PPL/accuracy **≥ GPTQ-4**, with the F3 blow-up gone
because no compensated column is later removed.

**Status:** not started. **Risk:** lands near SpQR/OWQ territory; the novelty is the
*ordering* claim + the matched-bit audit that shows post-hoc protection fails and
protect-first does not. **Keep it out of the Track A paper** — Track A is an audit,
this would be a separate method paper.

**How to run** (GPU box; extends the existing plumbing rather than new infra):
- `seq_core/gptq_llmc_base.py` / `channel_sweep.py --base_quantizer gptq_llmc` is the
  hook point — add a "mask these input channels to FP16 before GPTQ compensates"
  path, then reuse the same sweep/gate/comparison pipeline for the matched-bit
  contrast against `gptq4` and the naive `resmax_gptq`/`greedy_gptq` points already
  defined in `configs/downstream_operating_points.json`.

---

## 5. Explicitly rejected — do NOT revisit

- **Module-level entropy allocation** (Era B, runs 1–6): proxies decouple from PPL;
  ≤ uniform; EWQ prior art.
- **Interaction-aware greedy set selection** (Era C, F2): no robust win over a
  one-shot activation-magnitude score; the second-order machinery is unjustified.
- **Naive stacking of FP16 protection on a GPTQ base** (Era C, F3): catastrophic
  (PPL 10→40–107). Only ever protect a data-free base, or protect *before*
  compensation (§4).
- **Percentile/tier heuristics without matched-bit + random controls** (Era A).

---

## 6. What is LEFT for Track B (bottom line)

1. **Confirm the one positive at task level** — the downstream eval staged today
   (`resmax_gptq` vs `gptq4`, `best_hqq` vs `hqq4`). If the F4 Pareto point survives
   downstream, that is Track B's contribution *to the audit paper* (F5), not a
   method paper.
2. **Optionally, the protect-then-recompensate experiment (§4)** — the only thing
   that could upgrade the audit into a method paper. One focused GPU study; separate
   paper; separate branch.
3. Otherwise Track B is **closed**. The honest science is in Track A.

**Cross-links:** results/numbers → `PROJECT_STATUS_AND_ROADMAP.md §1,§4`; audit
write-up → `FINDINGS_PAPER.md` (F1–F5); today's GPU tasks → `GPU_RUNBOOK_TODAY.md`.
