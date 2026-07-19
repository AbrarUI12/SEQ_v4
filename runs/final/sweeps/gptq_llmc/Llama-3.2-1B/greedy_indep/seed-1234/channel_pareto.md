# Per-channel protection — meta-llama/Llama-3.2-1B

Backend `hqq`, base 4-bit, canonical PPL. FP16 PPL = **9.7571**.
Rows = signal; columns = protection config; `random` is the control. **At matched effective bits, signal < random means per-channel importance is real.**

## PPL by config (effective bits in parentheses)

| signal | k=0.02 | k=0.05 | k=0.1 | k=0.2 |
|---|---|---|---|---|
| `greedy_indep` | 15.639 (4.24b) | 14.884 (4.60b) | 14.864 (5.20b) | 15.607 (6.40b) |

## PPL gap vs random (negative = signal beats random)

| signal | k=0.02 | k=0.05 | k=0.1 | k=0.2 |
|---|---|---|---|---|
| `greedy_indep` | — | — | — | — |

