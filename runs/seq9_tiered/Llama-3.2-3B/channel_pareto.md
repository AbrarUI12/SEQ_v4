# Per-channel protection — meta-llama/Llama-3.2-3B

Backend `hqq`, base 3-bit, canonical PPL. FP16 PPL = **7.8165**.
Rows = signal; columns = protection config; `random` is the control. **At matched effective bits, signal < random means per-channel importance is real.**

## PPL by config (effective bits in parentheses)

| signal | [16:0.1] | [16:0.05,8:0.13] | [16:0.02,8:0.22] | [8:0.26] |
|---|---|---|---|---|
| `act_max` | 9.301 (4.30b) | 9.100 (4.30b) | 8.962 (4.36b) | 9.038 (4.30b) |
| `random` | 12.941 (4.30b) | 12.471 (4.30b) | 12.157 (4.36b) | 12.034 (4.30b) |

## PPL gap vs random (negative = signal beats random)

| signal | [16:0.1] | [16:0.05,8:0.13] | [16:0.02,8:0.22] | [8:0.26] |
|---|---|---|---|---|
| `act_max` | -3.641 | -3.371 | -3.195 | -2.996 |
