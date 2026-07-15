# SEQ — status, what was rejected, and publication plan

Date: 2026-07-15. Target: August ARR cycle (deadline ~Aug 15), then ≥5 days to write.

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
same protection. **Bug found by the k=0 sanity check** (uniform GPTQ PPL 6717,
should be ~10–11): the Hessian was calibrated on padded sequences (~97% pad
tokens). **Fixed** (calibrate on real tokens, no padding). Pending: re-run the
1B k=0 point to confirm ~10–11.

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

## 8. Honest bottom line

- **A submittable, honest paper by the deadline: yes, feasible.**
- **A\* main-track acceptance: not guaranteed** — it hinges on the scenario-B
  competitive result landing and the `act_max`-vs-AWQ novelty being framed well.
  The safe, high-probability outcome is a **Findings/workshop** paper; main track
  is the upside if GPTQ+protection beats the baselines cleanly.
- None of the work is wasted: the negative results and the audit are the spine,
  and they are real.
