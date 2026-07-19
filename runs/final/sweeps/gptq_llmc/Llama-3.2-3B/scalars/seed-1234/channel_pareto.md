# Per-channel protection — meta-llama/Llama-3.2-3B

Backend `hqq`, base 4-bit, canonical PPL. FP16 PPL = **7.8166**.
Rows = signal; columns = protection config; `random` is the control. **At matched effective bits, signal < random means per-channel importance is real.**

## PPL by config (effective bits in parentheses)

| signal | k=0.02 | k=0.05 | k=0.1 | k=0.2 |
|---|---|---|---|---|
| `act_max` | 8.168 (4.24b) | 8.168 (4.60b) | 8.134 (5.20b) | 8.106 (6.40b) |
| `act_scale` | 8.577 (4.24b) | 8.210 (4.60b) | 8.189 (5.20b) | 8.152 (6.40b) |
| `residual_rms` | 8.138 (4.24b) | 8.147 (4.60b) | 8.148 (5.20b) | 8.129 (6.40b) |

## PPL gap vs random (negative = signal beats random)

| signal | k=0.02 | k=0.05 | k=0.1 | k=0.2 |
|---|---|---|---|---|
| `act_max` | — | — | — | — |
| `act_scale` | — | — | — | — |
| `residual_rms` | — | — | — | — |

