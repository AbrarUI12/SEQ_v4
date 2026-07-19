# Per-channel protection — meta-llama/Llama-3.2-1B

Backend `hqq`, base 4-bit, canonical PPL. FP16 PPL = **9.7571**.
Rows = signal; columns = protection config; `random` is the control. **At matched effective bits, signal < random means per-channel importance is real.**

## PPL by config (effective bits in parentheses)

| signal | [budget=0.25] | [budget=0.5] | [budget=1.0] | [budget=2.0] |
|---|---|---|---|---|
| `tier_alloc` | 10.631 (4.07b) | 10.592 (4.13b) | 10.555 (4.27b) | 10.479 (4.53b) |

## PPL gap vs random (negative = signal beats random)

| signal | [budget=0.25] | [budget=0.5] | [budget=1.0] | [budget=2.0] |
|---|---|---|---|---|
| `tier_alloc` | — | — | — | — |

