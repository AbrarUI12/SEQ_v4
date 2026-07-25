# Per-channel protection — meta-llama/Llama-3.2-1B

Backend `hqq`, base 4-bit, canonical PPL. FP16 PPL = **9.7572**.
Rows = signal; columns = protection config; `random` is the control. **At matched effective bits, signal < random means per-channel importance is real.**

## PPL by config (effective bits in parentheses)

| signal | k=0.02 |
|---|---|
| `greedy` | 63.946 (4.24b) |

## PPL gap vs random (negative = signal beats random)

| signal | k=0.02 |
|---|---|
| `greedy` | — |

