# Per-channel protection — meta-llama/Llama-3.2-1B

Backend `hqq`, base 3-bit, canonical PPL. FP16 PPL = **9.7573**.
Rows = signal; columns = protection config; `random` is the control. **At matched effective bits, signal < random means per-channel importance is real.**

## PPL by config (effective bits in parentheses)

| signal | [16:0.1] | [16:0.05,8:0.13] | [16:0.02,8:0.22] | [8:0.26] |
|---|---|---|---|---|
| `act_max` | 13.276 (4.30b) | 12.654 (4.30b) | 12.301 (4.36b) | 12.394 (4.30b) |
| `random` | 28.014 (4.30b) | 24.723 (4.30b) | 23.339 (4.36b) | 22.716 (4.30b) |

## PPL gap vs random (negative = signal beats random)

| signal | [16:0.1] | [16:0.05,8:0.13] | [16:0.02,8:0.22] | [8:0.26] |
|---|---|---|---|---|
| `act_max` | -14.737 | -12.068 | -11.038 | -10.321 |
