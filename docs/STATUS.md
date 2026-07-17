# SEQ — status, what was rejected, and publication plan

Date: 2026-07-15. Target: August ARR cycle (deadline ~Aug 15), then ≥5 days to write.

## 0. Update 2026-07-17 — principled selectors + a working GPTQ base

Three changes land the method side of the project and fix a measurement bug.

**(a) Baseline mislabel bug — FIXED.** In `channel_sweep.py` the FP16 baseline was
measured *after* the GPTQ precompute; sequential GPTQ mutates the decoder weights
in place, so `baseline_fp16_ppl` was silently recording the GPTQ-base PPL. The
baseline is now measured on the unmodified FP16 model *before* any precompute, and
the uniform-base PPL is recorded separately as `baseline_base_ppl` (the k=0 gate).

**(b) From-scratch GPTQ — SHELVED; LightCompress is the base.** The hand-written
`seq_core/gptq.py` never produced a working full-model base (≈5 attempts; even the
sequential path gives k=0 ≈ 4900, not ~10.4) and cannot be debugged in the no-GPU
dev environment. The per-layer math is provably correct (`scripts/diag_gptq.py`),
so the failure is in full-model application, not the algorithm — but chasing it
further is not worth the deadline risk. The strong-base path now **loads a saved
LightCompress (LLMC) fake-quant model** (its GPTQ works: W4g128 ≈ 10.39) via
`--base_quantizer gptq_llmc --gptq_model_path <dir>` (loader: `seq_core/gptq_llmc_base.py`).
`--base_quantizer gptq` is kept for diagnostics only.

**(c) Two principled, interaction-aware selectors** replace the (exhausted) scalar-
signal search — the direction from the "IMPROVING HQQ SEQ" note:
- **Residual-aware signals** `residual_rms` = `E[x_j²]·‖ΔW_:,j‖²` and `residual_max`
  = `(max_t|x_{t,j}|)²·‖ΔW_:,j‖²` (`--signals residual_rms,residual_max`). Unlike
  `act_max`, these weight each channel by the *real* quantization error it carries
  against the built base (`ΔW = W − Wq`). (`seq_core/recon_sensitivity.py`.)
- **Greedy OMP selector** (`--select greedy`): picks the *set* of channels that most
  reduces the layer output residual `‖ΔW X‖²_F = tr(ΔWᵀΔW H)`, via
  `G_j = 2⟨ΔW_:,j,(ΔW H)_:,j⟩ − ‖ΔW_:,j‖²·H_jj` with a rank-1 update each step. This
  captures cross-channel interactions that any independent scalar score misses —
  the audit's core finding. (`seq_core/greedy_select.py`.)
- **Value-based tier allocation** (`--tier_alloc value`): spends a bit budget across
  {base, 8, 16} by error-per-byte `(D_t − D_{t+1})/(cost_{t+1} − cost_t)` instead of
  fixed `--protect_tiers` percentages. (`greedy_bit_alloc_by_value` in `channel_utils.py`.)

All pure logic is unit-tested (`tests/test_greedy_select.py`,
`tests/test_channel_utils.py`; 63 checks). **The decisive question is now runnable:**
does protection improve a GPTQ base over plain GPTQ-4 at matched actual bits?

**Run — new selectors on the HQQ base (does residual/greedy beat `act_max`?):**
```bash
python -m seq_core.channel_sweep --model meta-llama/Llama-3.2-1B --backend hqq \
  --base_bits 4 --protect_fracs 0,0.02,0.05,0.1,0.2 \
  --signals act_max,residual_rms,residual_max,random \
  --ppl_mode canonical --calibration_prompts calibration_prompts.json \
  --out_dir runs/seq10_residual/Llama-3.2-1B
# greedy (signal-agnostic): add --select greedy, out_dir runs/seq10_greedy/...
# value tiers: --tier_alloc value --protect_fracs 0.25,0.5,1.0,2.0 (fracs = extra bits/channel budget)
```

**Run — the decisive test (protection on the GPTQ base vs plain GPTQ-4):**
```bash
# 1) produce the LightCompress fake-quant W4 model (its GPTQ works ~10.39):
python run_compare_matrix.py --model meta-llama/Llama-3.2-1B \
  --methods gptq_llmc --llmc_save_mode fake --llmc_repo /path/to/LightCompress
#    -> note the saved fake-quant dir (llmc_save_path in the run's meta)
# 2) run SEQ protection on that base; the k=0 row MUST reproduce ~10.39:
python -m seq_core.channel_sweep --model meta-llama/Llama-3.2-1B --backend hqq \
  --base_bits 4 --base_quantizer gptq_llmc --gptq_model_path <SAVED_FAKE_QUANT_DIR> \
  --protect_fracs 0,0.02,0.05,0.1 --signals act_max,residual_rms,random \
  --ppl_mode canonical --calibration_prompts calibration_prompts.json \
  --out_dir runs/seq10_gptqbase/Llama-3.2-1B
#    strongest combo: add --select greedy (greedy selection on the GPTQ residual).
# 3) rebuild the comparison table (include the new signal labels):
python analysis/build_comparison.py \
  --sweeps runs/seq10_residual runs/seq10_greedy runs/seq10_gptqbase \
  --baselines baselines.json \
  --signals act_max,residual_rms,residual_max,greedy,tier_alloc,random,act_scale \
  --out docs/COMPARISON.md
```

Decision rule (unchanged): if GPTQ-base + protection beats plain GPTQ-4 at matched
**actual** bits → scenario-B method paper; else the audit is the paper and these
selectors are the interaction-aware method the audit motivates. Sections 5, 7
below predate this update and are kept for history.

## 1. Where the project stands (one line)

The original SEQ premise (entropy-guided module-level mixed precision) is
**falsified by our own data**; what survives is a rigorous **audit** of
quantization importance signals plus a **positive per-channel result**
(activation-outlier FP16 protection, `act_max` > AWQ's `act_scale`). The
competitive-Pareto piece (GPTQ base) is implemented, had a calibration bug
(now fixed), and is pending validation.

## 2. What we established (rigorous, committed — runs 1–6)

1. **Entropy is a poor signal.** Module-level weight entropy is weakly / anti-
   correlated with quantization sensitivity (runs 1–2).
2. **Reconstruction/Hessian proxies mislead.** They ace the local reconstruction
   correlation (ρ≈0.99) yet are **anti-correlated with end-to-end PPL**; the
   "principled" Hessian signal is the *worst* module-level allocator (run 3).
3. **Module-level mixed precision cannot beat uniform** at 3–7 bits; concentration
   is actively harmful (run 3).
4. **Per-channel protection works.** Keeping the top-k% activation-outlier input
   channels in FP16 beats a random-channel control on all models/bit-widths;
   weight magnitude ≈ random → the signal is *activation*, not weights (run 4).
5. **`act_max` (outlier magnitude) > `act_scale` (AWQ's mean magnitude)**, and the
   margin grows at low bits (−1.5 PPL at 3-bit on 1B); `act_entropy` ≈ random
   (runs 5–6).
6. **The base quantizer is the Pareto ceiling.** RTN (bitsandbytes/HQQ) cannot
   reach ≤ FP16 below ~8 bits regardless of the protection signal — this is the
   empirical answer to the original RQ3 (run 6).

## 3. What was rejected (honest negative results — these are contributions)

- Entropy-guided module mixed precision (the original SEQ) — does not beat uniform.
- Module-level allocation by **any** signal (entropy, magnitude, Hessian) — same.
- Reconstruction-error / Hessian as the importance proxy — misleading, anti-correlated.
- Per-channel **entropy**/information content as a protection signal — ≈ random.
- Sub-4-bit **RTN** base (bitsandbytes/HQQ) — unusable; dominated at matched bits.

## 4. What survives as the paper's contributions

- **Methodology:** a multi-granularity ground-truth audit (one-hot degrade,
  reconstruction, downstream Pareto) with **random controls**, showing popular
  importance proxies are decoupled from end-to-end quality.
- **Positive method:** per-channel activation-**outlier-magnitude** FP16
  protection, beating AWQ's mean-magnitude criterion, strongest at low bits.
- **Systems finding:** base-quantizer quality dominates; RTN is the ceiling
  (RQ3), motivating error-compensated base + outlier protection.

## 5. GPTQ base status

Implemented from scratch (faithful to auto-gptq), fake-quant output under the
same protection. The k=0 sanity check caught two calibration bugs:
1. Padding (~97% pad tokens) polluted the Hessian — fixed (real tokens).
2. **Root cause:** the calibration set is only ~614 real tokens, but GPTQ
   inverts a per-layer Hessian of size [in, in] (in up to ~14k) → rank-deficient
   → singular inverse → garbage base (k=0 PPL 15701). **Fixed:**
   `build_gptq_calibration` makes 128×2048 = 262k real tokens (standard GPTQ).
   The algorithm itself was fine — it was starved of data. **Pending: re-run 1B
   k=0, expect ~10–11.**

## 5b. Composite-signal result (run 8, HQQ 3-bit) — act_max is the best simple signal

Tested composites and the proper entropy direction against `act_max`. **Nothing
beats `act_max`** (gap vs act_max, 1B/3B/8B): `act_max*act_rms` +0.1–0.5,
`act_scale` +0.2–1.5, `neg_act_entropy` +0.4–2.2, `act_max*act_kurt` +0.3–2.0,
`act_entropy`/`random` +2–17. Conclusion: there is **no novel composite signal**;
per-channel **outlier magnitude alone** is the selection criterion. This closes
the "new signal" search and sharpens the message.

## 6. Two paper scenarios

- **A — Analysis/audit paper (SAFE, essentially done).** Runs 1–6 + writing.
  Realistic venue: **ACL/EMNLP Findings** or a strong efficiency workshop; main
  track possible but competitive for a mostly-negative + modest-positive result.
- **B — Competitive method paper (higher ceiling, riskier).** Needs GPTQ-4bit +
  `act_max` protection to reach ≤ FP16 at 5–7 effective bits **and** beat
  AWQ/GPTQ at equal bits. Gated on (i) GPTQ validating, (ii) the margin holding
  (uncertain — GPTQ-4bit is already near-FP16, so headroom is small).

**Recommendation:** write around **A** as the backbone and fold in **B** if the
GPTQ Pareto lands cleanly — so there is a submittable paper either way.

## 7. Remaining experiments (prioritized for the deadline)

1. **Validate GPTQ** — 1B `k=0` ≈ 10–11 (≈1 hr). Gate for scenario B.
2. **GPTQ + protection Pareto** on 1B/3B vs **AWQ/GPTQ baselines** (LightCompress,
   already integrated) at matched effective bits — the headline figure (~2–3 d).
3. **Per-channel sensitivity audit** (built, un-run) — shows `act_max` is
   near-optimal or finds headroom (~0.5 d).
4. **Downstream tasks** (lm-eval: hellaswag/arc/piqa/winogrande) at the operating
   point — reviewers want more than PPL (~1 d).
5. **Multi-seed / 1–2 more model families** for robustness of the headline (~1 d).

≈ 1 week of experiments + 5 days writing → fits in the ~4.5 weeks to Aug 15 with
buffer, **if GPTQ validates in the next day or two.**

## 7b. Making SEQ a distinctive method (not just AWQ+act_max)

Two channel-selection ideas were **never properly tested** and are where a novel
SEQ could live:

- **Composite signals** (now supported): `A*B` (min-max normalize each, product —
  favor channels high in *both*) or `A+B` (sum). Never tried; a composite that
  beats `act_max` alone would be the SEQ selection criterion.
- **`neg_act_entropy`** (protect *low*-entropy = outlier channels): the correct
  entropy direction, silently skipped in run 6.

Run (works on the validated HQQ base now; add `--base_quantizer gptq` after GPTQ
validates):

```bash
python -m seq_core.channel_sweep \
  --model meta-llama/Llama-3.2-1B --backend hqq --base_bits 3 \
  --protect_fracs 0,0.02,0.05,0.1,0.2 \
  --channel_entropy --entropy_bins 32 \
  --signals act_max,act_scale,random,neg_act_entropy,"act_max*act_kurt","act_max*neg_act_entropy","act_max*act_rms" \
  --ppl_mode canonical --calibration_prompts calibration_prompts.json \
  --out_dir runs/seq8_composite/Llama-3.2-1B
```

Decision rule: if any composite beats `act_max` by a consistent margin across
1B/3B, that is *SEQ's* selection score (the method's novelty). If not, SEQ =
`act_max` protection on a strong base, framed against AWQ/LLM.int8 + the audit.
(`3-bit base` chosen because the signal gaps are largest there.)

## 8. Honest bottom line

- **A submittable, honest paper by the deadline: yes, feasible.**
- **A\* main-track acceptance: not guaranteed** — it hinges on the scenario-B
  competitive result landing and the `act_max`-vs-AWQ novelty being framed well.
  The safe, high-probability outcome is a **Findings/workshop** paper; main track
  is the upside if GPTQ+protection beats the baselines cleanly.
- None of the work is wasted: the negative results and the audit are the spine,
  and they are real.
