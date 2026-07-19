# Findings run 3 — downstream Pareto (does the signal beat random?)

End-to-end canonical PPL after allocating bits by each signal (native high→more-bits) and quantizing with HQQ. `random` is the chance control. **Lower is better; a positive gap-vs-random means the signal is WORSE than chance.**

## meta-llama/Llama-3.1-8B (FP16 PPL 6.238)

| signal | ~3.0b | ~3.5b | ~4.0b | ~5.0b | ~6.0b | ~7.0b | mean gap vs random |
|---|---|---|---|---|---|---|---|
| `entropy` | 9.73 | 8.31 | 6.95 | 6.80 | 6.72 | 6.55 | -0.09 ✅ |
| `random` | 9.73 | 8.52 | 7.23 | 7.00 | 6.72 | 6.39 | +0.00 |
| `magnitude` | 9.73 | 8.45 | 7.92 | 6.80 | 6.71 | 6.53 | +0.09 |
| `salience` | 9.73 | 8.59 | 8.01 | 7.61 | 7.47 | 7.01 | +0.47 ❌ |
| `hessian_diag` | 9.73 | 9.11 | 8.59 | 7.90 | 7.73 | 6.43 | +0.65 ❌ |

Concentration (share of params forced to the lowest level 3-bit) — high share ⇒ pathological allocation:

| signal | ~3.0b | ~3.5b | ~4.0b | ~5.0b | ~6.0b | ~7.0b |
|---|---|---|---|---|---|---|
| `entropy` | 100% | 64% | 14% | 7% | 7% | 7% |
| `hessian_diag` | 100% | 82% | 53% | 36% | 33% | 7% |
| `magnitude` | 100% | 76% | 42% | 7% | 7% | 7% |
| `random` | 100% | 67% | 35% | 20% | 13% | 3% |
| `salience` | 100% | 59% | 38% | 29% | 23% | 5% |

## meta-llama/Llama-3.2-3B (FP16 PPL 7.817)

| signal | ~3.0b | ~3.5b | ~4.0b | ~5.0b | ~6.0b | ~7.0b | mean gap vs random |
|---|---|---|---|---|---|---|---|
| `entropy` | 13.78 | 11.19 | 8.57 | 8.38 | 8.16 | 8.07 | -0.07 ✅ |
| `magnitude` | 13.78 | 11.16 | 8.64 | 8.38 | 8.31 | 8.09 | -0.04 |
| `random` | 13.78 | 11.21 | 8.85 | 8.57 | 8.24 | 7.93 | +0.00 |
| `salience` | 13.78 | 11.92 | 10.98 | 10.54 | 10.31 | 9.53 | +1.41 ❌ |
| `hessian_diag` | 13.78 | 13.35 | 12.51 | 10.94 | 8.89 | 8.03 | +1.49 ❌ |

Concentration (share of params forced to the lowest level 3-bit) — high share ⇒ pathological allocation:

| signal | ~3.0b | ~3.5b | ~4.0b | ~5.0b | ~6.0b | ~7.0b |
|---|---|---|---|---|---|---|
| `entropy` | 100% | 72% | 22% | 12% | 12% | 12% |
| `hessian_diag` | 100% | 88% | 60% | 33% | 28% | 7% |
| `magnitude` | 100% | 72% | 22% | 12% | 12% | 12% |
| `random` | 100% | 64% | 38% | 26% | 17% | 2% |
| `salience` | 100% | 61% | 34% | 26% | 22% | 6% |

## meta-llama/Llama-3.2-1B (FP16 PPL 9.757)

| signal | ~3.0b | ~3.5b | ~4.0b | ~5.0b | ~6.0b | ~7.0b | mean gap vs random |
|---|---|---|---|---|---|---|---|
| `entropy` | 32.42 | 13.20 | 11.51 | 11.06 | 10.38 | 9.86 | -0.19 ✅ |
| `random` | 32.42 | 13.72 | 12.02 | 10.98 | 10.54 | 9.89 | +0.00 |
| `magnitude` | 32.42 | 19.03 | 12.08 | 11.17 | 10.47 | 9.86 | +0.91 ❌ |
| `salience` | 32.42 | 20.92 | 17.33 | 16.42 | 15.80 | 10.71 | +4.01 ❌ |
| `hessian_diag` | 32.42 | 30.49 | 30.49 | 18.12 | 16.47 | 10.56 | +8.16 ❌ |

Concentration (share of params forced to the lowest level 3-bit) — high share ⇒ pathological allocation:

| signal | ~3.0b | ~3.5b | ~4.0b | ~5.0b | ~6.0b | ~7.0b |
|---|---|---|---|---|---|---|
| `entropy` | 100% | 61% | 28% | 21% | 21% | 0% |
| `hessian_diag` | 100% | 79% | 79% | 31% | 26% | 12% |
| `magnitude` | 100% | 82% | 36% | 21% | 21% | 0% |
| `random` | 100% | 67% | 39% | 33% | 26% | 0% |
| `salience` | 100% | 56% | 30% | 24% | 20% | 4% |

## Verdict

| signal | mean gap vs random (per model) | overall |
|---|---|---|
| `entropy` | -0.09, -0.07, -0.19 | -0.12 |
| `hessian_diag` | +0.65, +1.49, +8.16 | +3.43 |
| `magnitude` | +0.09, -0.04, +0.91 | +0.32 |
| `random` | +0.00, +0.00, +0.00 | +0.00 |
| `salience` | +0.47, +1.41, +4.01 | +1.96 |

- **Positive gap = worse than random.** `hessian_diag` and `salience` — the winners of the reconstruction correlation — are far worse than random: their extensive per-layer sums over-protect `lm_head`/large layers and starve the rest to the lowest level.
- **`entropy`, `magnitude` ≈ `random`**: at module granularity and 3–7 bits, no signal reliably beats uniform/random. Entropy 'works' only because it is near-uniform across modules, so allocating by it ≈ uniform allocation.
- **Conclusion:** module-level sensitivity-guided mixed precision does not beat uniform here, and the reconstruction correlation is a misleading proxy. The open levers are per-channel allocation and a compounding-aware objective / per-parameter normalized signals with a low-bit floor.
