# Findings run 5 — can any channel signal beat activation magnitude?

Per-channel protection at 4-bit base, canonical PPL, 3 models. Question: does any
per-input-channel statistic pick better protected channels than `act_scale`
(mean|x|, AWQ's salience metric)?

## Result: outlier magnitude (`act_max`) > mean magnitude (`act_scale`)

PPL gap vs `act_scale` (negative = beats the AWQ metric):

| signal | 1B (k=1→10%) | 3B | 8B | reading |
|---|---|---|---|---|
| `act_max` | −0.05,−0.08,−0.15,−0.17 | −0.03…−0.06 | −0.03…−0.05 | **consistently best** |
| `act_rms` | ≈ 0 | ≈ 0 | ≈ 0 | tied with act_scale |
| `hessian_diag` | ≈ 0 / +0.02 | ≈ 0 | ≈ 0 | weight term adds nothing |
| `act_kurt` | +0.5 → +0.06 | +0.13 → +0.02 | +0.14 → +0.05 | weak (esp. tiny k) |
| `act_entropy` (high) | ≈ random | ≈ random | ≈ random | **information ≠ importance** |

Best recovery (1B, `act_max`, k=10%, 5.2 eff bits): uniform 4-bit 11.19 →
**10.33** (FP16 9.76) — recovers ~60% of the quantization loss.

## What this means for novelty — honest

- `act_max` (max|x| outlier magnitude) is essentially **LLM.int8()**'s
  outlier-feature criterion; `act_scale` (mean|x|) is **AWQ**'s. So the finding
  is *"max-based outlier selection beats mean-based salience for FP16 column
  protection."* That is a **real, clean, cross-model controlled result** — but
  both metrics are from prior work and the margin is modest (≤0.17 PPL). It is
  **not, by itself, an A\* method contribution.**
- `act_entropy ≈ random` and `hessian_diag ≈ act_scale` are useful *negative*
  findings: for channel protection, only outlier *magnitude* matters — not
  information content, not the weight-aware second-order term.

## Where the defensible paper is

The method delta over AWQ/LLM.int8 is thin. The **rigorous audit** is the
contribution, and it is genuinely unusual and honest:

1. Weight entropy is a poor / anti-correlated sensitivity signal (runs 1–2).
2. Local reconstruction error is near-perfectly predicted by the Hessian yet
   **anti-correlated with end-to-end quality**; module-level mixed precision
   **cannot beat uniform** and concentration is actively harmful (run 3).
3. Value exists **only per-channel**, and there it is driven by **outlier
   magnitude** (`act_max` > `act_scale` > weight-magnitude ≈ random; entropy ≈
   random) (runs 4–5).

Framing: *"What actually decides the bits? A controlled audit of LLM
quantization importance signals across granularities."* The headline —
**popular proxies (entropy, reconstruction error) are decoupled from end-to-end
quality, and the only thing that helps is per-channel outlier magnitude** — is a
clean empirical-study contribution (ACL/EMNLP Findings-tier; ICLR possible with
the two strengtheners below).

## Two strengtheners (recommended next)

1. **Lower base bits (run 6, no new code).** At 4-bit HQQ the gap to FP16 is
   small, so all margins compress. Protection matters most at 2–3-bit base — and
   the paper's "5–7 effective bits, ≤ FP16 PPL" target is best hit by a **2–3-bit
   base + a few FP16 outlier channels**. Rerun with `--base_bits 3` and
   `--base_bits 2`: expect much larger, more publishable separations
   (`act_max` vs `act_scale` vs `random`) and the real Pareto story.
2. **Per-channel true-sensitivity audit.** Measure the actual ΔPPL from
   protecting each channel/group and show *what `act_max` misses* — the honest
   ground truth that could motivate a signal beating it (and that turned runs
   1–3 from "reproduce heuristics" into surprising reversals).

### Run 6 recipe

```bash
for B in 3 2; do
  python -m seq_core.channel_sweep \
    --model meta-llama/Llama-3.2-1B --backend hqq \
    --base_bits $B --protect_fracs 0,0.02,0.05,0.1,0.2,0.3 \
    --signals act_max,act_scale,act_rms,random,neg_act_entropy \
    --ppl_mode canonical --calibration_prompts calibration_prompts.json \
    --out_dir runs/channel6_b$B/Llama-3.2-1B
done
```
