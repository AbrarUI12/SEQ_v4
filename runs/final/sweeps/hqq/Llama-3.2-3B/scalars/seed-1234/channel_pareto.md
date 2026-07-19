# Per-channel protection — meta-llama/Llama-3.2-3B

Backend `hqq`, base 4-bit, canonical PPL. FP16 PPL = **7.8166**.
Rows = signal; columns = protection config; `random` is the control. **At matched effective bits, signal < random means per-channel importance is real.**

## PPL by config (effective bits in parentheses)

| signal | k=0.02 | k=0.05 | k=0.1 | k=0.2 |
|---|---|---|---|---|
| `act_max` | 8.160 (4.24b) | 8.125 (4.60b) | 8.091 (5.20b) | 8.047 (6.40b) |
| `act_scale` | 8.205 (4.24b) | 8.171 (4.60b) | 8.146 (5.20b) | 8.096 (6.40b) |
| `residual_rms` | 8.200 (4.24b) | 8.167 (4.60b) | 8.141 (5.20b) | 8.090 (6.40b) |

## PPL gap vs random (negative = signal beats random)

| signal | k=0.02 | k=0.05 | k=0.1 | k=0.2 |
|---|---|---|---|---|
| `act_max` | — | — | — | — |
| `act_scale` | — | — | — | — |
| `residual_rms` | — | — | — | — |

