# Per-channel protection — meta-llama/Llama-3.2-3B

Backend `hqq`, base 4-bit, protected columns at 16-bit, canonical PPL. FP16 PPL = **7.8165**.
Rows = signal used to pick protected channels; `random` is the control. **At each k (same effective bits), signal < random means per-channel importance is real.**

## PPL by protection fraction k (effective bits in parentheses)

| signal | k=0.0 | k=0.02 | k=0.05 | k=0.1 | k=0.2 |
|---|---|---|---|---|---|
| `act_max` | 2379.739 (4.00b) | 8.349 (4.24b) | 8.285 (4.60b) | 8.235 (5.20b) | 8.173 (6.40b) |
| `act_scale` | 2379.739 (4.00b) | 1539.926 (4.24b) | 1403.941 (4.60b) | 1269.079 (5.20b) | 1238.301 (6.40b) |
| `random` | 2379.739 (4.00b) | 2311.068 (4.24b) | 2284.712 (4.60b) | 2265.552 (5.20b) | 2316.197 (6.40b) |

## PPL gap vs random at each k (negative = signal beats random)

| signal | k=0.0 | k=0.02 | k=0.05 | k=0.1 | k=0.2 |
|---|---|---|---|---|---|
| `act_max` | +0.000 | -2302.720 | -2276.427 | -2257.316 | -2308.023 |
| `act_scale` | +0.000 | -771.142 | -880.771 | -996.473 | -1077.896 |

