# Per-channel protection — meta-llama/Llama-3.2-3B

Backend `hqq`, base 4-bit, canonical PPL. FP16 PPL = **7.8167**.
Rows = signal; columns = protection config; `random` is the control. **At matched effective bits, signal < random means per-channel importance is real.**

## PPL by config (effective bits in parentheses)

| signal | k=0.02 | k=0.05 | k=0.1 | k=0.2 |
|---|---|---|---|---|
| `act_max` | 8.108 (4.24b) | 8.076 (4.60b) | 8.045 (5.20b) | 8.007 (6.40b) |
| `act_scale` | 8.146 (4.24b) | 8.114 (4.60b) | 8.090 (5.20b) | 8.046 (6.40b) |
| `residual_rms` | 8.141 (4.24b) | 8.111 (4.60b) | 8.087 (5.20b) | 8.040 (6.40b) |

## PPL gap vs random (negative = signal beats random)

| signal | k=0.02 | k=0.05 | k=0.1 | k=0.2 |
|---|---|---|---|---|
| `act_max` | — | — | — | — |
| `act_scale` | — | — | — | — |
| `residual_rms` | — | — | — | — |

