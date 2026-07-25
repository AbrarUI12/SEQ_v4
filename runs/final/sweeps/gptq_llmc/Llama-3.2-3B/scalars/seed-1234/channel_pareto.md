# Per-channel protection — meta-llama/Llama-3.2-3B

Backend `hqq`, base 4-bit, canonical PPL. FP16 PPL = **7.8167**.
Rows = signal; columns = protection config; `random` is the control. **At matched effective bits, signal < random means per-channel importance is real.**

## PPL by config (effective bits in parentheses)

| signal | k=0.02 | k=0.05 | k=0.1 | k=0.2 |
|---|---|---|---|---|
| `act_max` | 8.181 (4.24b) | 8.173 (4.60b) | 8.136 (5.20b) | 8.102 (6.40b) |
| `act_scale` | 8.586 (4.24b) | 8.204 (4.60b) | 8.187 (5.20b) | 8.154 (6.40b) |
| `residual_rms` | 8.146 (4.24b) | 8.153 (4.60b) | 8.160 (5.20b) | 8.139 (6.40b) |

## PPL gap vs random (negative = signal beats random)

| signal | k=0.02 | k=0.05 | k=0.1 | k=0.2 |
|---|---|---|---|---|
| `act_max` | — | — | — | — |
| `act_scale` | — | — | — | — |
| `residual_rms` | — | — | — | — |

