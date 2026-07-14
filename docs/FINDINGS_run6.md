# Findings run 6 — low-bit Pareto: the base quantizer is the bottleneck

Per-channel `act_max` protection on 2-/3-/4-bit HQQ bases, canonical PPL. This
run answers two things: (1) does the `act_max > act_scale` advantage grow at low
bits (yes), and (2) does a low base + protection hit "5–7 effective bits ≤ FP16"
(no — and why).

## 1. `act_max` beats `act_scale` more at low bits (novelty signal strengthens)

PPL gap `act_max − act_scale` (negative = act_max wins), 3-bit base:

| model | k=2% | k=5% | k=10% | k=20% |
|---|---|---|---|---|
| 1B | −1.14 | −1.49 | −1.52 | −1.28 |
| 3B | −0.20 | −0.31 | −0.36 | −0.30 |
| 8B | −0.21 | −0.26 | −0.22 | −0.18 |

At 4-bit the gap was ≤0.17; at 3-bit it's up to **−1.5 PPL (1B)**. And protection
vs `random` at 3-bit is enormous (1B: −16.9 PPL) — outlier channels are
*essential* at aggressive quantization. The per-channel outlier-magnitude story
is real and its margin over AWQ's mean-magnitude grows as bits shrink.

## 2. But the base quantizer dominates the Pareto (the important finding)

`act_max` PPL at matched **effective bits**, different HQQ base widths (1B, FP16 = 9.76):

| eff bits | PPL | base |
|---|---|---|
| 3.26 | 14.98 | 3-bit |
| 4.24 | **10.54** | 4-bit |
| 4.30 | 13.28 | 3-bit |
| 5.20 | **10.33** | 4-bit |
| 5.60 | 12.52 | 3-bit |
| 6.90 | 11.97 | 3-bit |

**The entire achievable frontier is 4-bit-base points.** A 4-bit base with *less*
protection always beats a 3-bit base with *more* protection at the same effective
bits. 2-bit HQQ is unusable (PPL 10³–10⁵, protection can't rescue it). So base
quantizer quality is the binding constraint, not the protection signal.

## 3. The target is not met with an RTN base — and that is RQ3's answer

The best point here is **5.2 eff bits → 10.33 PPL, still +0.57 above FP16 (9.76)**.
No HQQ-based config reaches "≤ FP16 at 5–7 effective bits." The reason is that
HQQ (like bitsandbytes) is **round-to-nearest**: it has no error compensation, so
4-bit already sits ~1.4 PPL above FP16 and sub-4-bit collapses. Per-channel
protection recovers part of that gap but cannot close it.

This is exactly the RQ3 question from day one — *"is bitsandbytes a limitation?"* —
now answered **empirically: yes.** RTN base quantization is the ceiling. To hit
≤ FP16 at 5–7 effective bits you need an **error-compensated base (GPTQ/AWQ)**,
with per-channel `act_max` protection on top.

## Consequence for the paper

Two coherent stories, both honest:

- **A (analysis + method, no new base):** *"Per-channel outlier-magnitude
  protection beats AWQ's mean-magnitude salience and is essential at low bits,
  but base-quantizer quality dominates the Pareto: RTN (bitsandbytes/HQQ) cannot
  reach FP16 below 8 bits regardless of the protection signal."* Strong, honest,
  fully supported by runs 1–6.
- **B (competitive method):** add a GPTQ/AWQ base under the same protection and
  show the combination reaches ≤ FP16 at 5–7 effective bits, beating each
  baseline. This needs a **calibration-aware backend** (GPTQ needs per-layer
  activations; the current `QuantBackend` is data-free) — a real but bounded
  build — and would make the Pareto competitive rather than "RTN is the ceiling."

## Recommended next

1. **Run the audit** (built, `seq_core/channel_audit.py`) — still the
   novelty-decider: is `act_max` optimal, or is there a signal that beats it?
2. **Decide A vs B.** If B, I build a calibration-aware backend + GPTQ base so
   `GPTQ-4bit + act_max protection` can be measured against the FP16 target.
