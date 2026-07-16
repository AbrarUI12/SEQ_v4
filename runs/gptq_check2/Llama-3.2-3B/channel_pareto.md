# Per-channel protection — meta-llama/Llama-3.2-3B

Backend `hqq`, base 4-bit, protected columns at 16-bit, canonical PPL. FP16 PPL = **7.8165**.
Rows = signal used to pick protected channels; `random` is the control. **At each k (same effective bits), signal < random means per-channel importance is real.**

## PPL by protection fraction k (effective bits in parentheses)

| signal | k=0.0 |
|---|---|
| `act_max` | 3149.323 (4.00b) |

## PPL gap vs random at each k (negative = signal beats random)

| signal | k=0.0 |
|---|---|
| `act_max` | — |

