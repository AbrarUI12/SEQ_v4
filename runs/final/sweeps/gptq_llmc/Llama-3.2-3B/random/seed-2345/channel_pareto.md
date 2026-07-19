# Per-channel protection — meta-llama/Llama-3.2-3B

Backend `hqq`, base 4-bit, canonical PPL. FP16 PPL = **7.8166**.
Rows = signal; columns = protection config; `random` is the control. **At matched effective bits, signal < random means per-channel importance is real.**

## PPL by config (effective bits in parentheses)

| signal | k=0.02 | k=0.05 | k=0.1 | k=0.2 |
|---|---|---|---|---|
| `random` | 8.171 (4.24b) | 8.153 (4.60b) | 8.156 (5.20b) | 8.143 (6.40b) |

## PPL gap vs random (negative = signal beats random)

| signal | k=0.02 | k=0.05 | k=0.1 | k=0.2 |
|---|---|---|---|---|

