# Per-channel protection — Qwen/Qwen2.5-3B

Backend `hqq`, base 4-bit, canonical PPL. FP16 PPL = **8.0304**.
Rows = signal; columns = protection config; `random` is the control. **At matched effective bits, signal < random means per-channel importance is real.**

## PPL by config (effective bits in parentheses)

| signal | k=0.02 |
|---|---|
| `greedy` | 8.335 (4.24b) |

## PPL gap vs random (negative = signal beats random)

| signal | k=0.02 |
|---|---|
| `greedy` | — |

