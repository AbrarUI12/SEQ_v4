# Per-channel protection — meta-llama/Llama-3.2-1B

Backend `hqq`, base 4-bit, canonical PPL. FP16 PPL = **9.7571**.
Rows = signal; columns = protection config; `random` is the control. **At matched effective bits, signal < random means per-channel importance is real.**

## PPL by config (effective bits in parentheses)

| signal | k=0.02 | k=0.05 | k=0.1 | k=0.2 |
|---|---|---|---|---|
| `act_max` | 10.692 (4.24b) | 10.657 (4.60b) | 10.638 (5.20b) | 10.362 (6.40b) |
| `act_scale` | 10.935 (4.24b) | 10.456 (4.60b) | 10.460 (5.20b) | 10.449 (6.40b) |
| `residual_rms` | 10.400 (4.24b) | 10.424 (4.60b) | 10.411 (5.20b) | 10.403 (6.40b) |

## PPL gap vs random (negative = signal beats random)

| signal | k=0.02 | k=0.05 | k=0.1 | k=0.2 |
|---|---|---|---|---|
| `act_max` | — | — | — | — |
| `act_scale` | — | — | — | — |
| `residual_rms` | — | — | — | — |

