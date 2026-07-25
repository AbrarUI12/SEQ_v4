# Per-channel protection — meta-llama/Llama-3.2-1B

Backend `hqq`, base 4-bit, canonical PPL. FP16 PPL = **9.7572**.
Rows = signal; columns = protection config; `random` is the control. **At matched effective bits, signal < random means per-channel importance is real.**

## PPL by config (effective bits in parentheses)

| signal | k=0.02 | k=0.05 | k=0.1 | k=0.2 |
|---|---|---|---|---|
| `act_max` | 10.711 (4.24b) | 10.687 (4.60b) | 10.646 (5.20b) | 10.383 (6.40b) |
| `act_scale` | 10.930 (4.24b) | 10.476 (4.60b) | 10.484 (5.20b) | 10.479 (6.40b) |
| `residual_rms` | 10.434 (4.24b) | 10.436 (4.60b) | 10.441 (5.20b) | 10.452 (6.40b) |

## PPL gap vs random (negative = signal beats random)

| signal | k=0.02 | k=0.05 | k=0.1 | k=0.2 |
|---|---|---|---|---|
| `act_max` | — | — | — | — |
| `act_scale` | — | — | — | — |
| `residual_rms` | — | — | — | — |

