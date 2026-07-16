# Per-channel protection — meta-llama/Llama-3.1-8B

Backend `hqq`, base 4-bit, protected columns at 16-bit, canonical PPL. FP16 PPL = **6.2384**.
Rows = signal used to pick protected channels; `random` is the control. **At each k (same effective bits), signal < random means per-channel importance is real.**

## PPL by protection fraction k (effective bits in parentheses)

| signal | k=0.0 |
|---|---|
| `act_max` | 8086.029 (4.00b) |

## PPL gap vs random at each k (negative = signal beats random)

| signal | k=0.0 |
|---|---|
| `act_max` | — |

