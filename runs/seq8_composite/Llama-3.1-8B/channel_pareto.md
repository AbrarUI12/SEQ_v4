# Per-channel protection — meta-llama/Llama-3.1-8B

Backend `hqq`, base 3-bit, protected columns at 16-bit, canonical PPL. FP16 PPL = **6.2384**.
Rows = signal used to pick protected channels; `random` is the control. **At each k (same effective bits), signal < random means per-channel importance is real.**

## PPL by protection fraction k (effective bits in parentheses)

| signal | k=0.0 | k=0.02 | k=0.05 | k=0.1 | k=0.2 |
|---|---|---|---|---|---|
| `act_entropy` | 9.726 (3.00b) | 9.664 (3.26b) | 9.557 (3.65b) | 9.368 (4.30b) | 9.055 (5.60b) |
| `act_max` | 9.726 (3.00b) | 7.843 (3.26b) | 7.631 (3.65b) | 7.506 (4.30b) | 7.316 (5.60b) |
| `act_max*act_kurt` | 9.726 (3.00b) | 8.170 (3.26b) | 8.063 (3.65b) | 7.872 (4.30b) | 7.574 (5.60b) |
| `act_max*act_rms` | 9.726 (3.00b) | 7.918 (3.26b) | 7.722 (3.65b) | 7.568 (4.30b) | 7.341 (5.60b) |
| `act_scale` | 9.726 (3.00b) | 8.056 (3.26b) | 7.889 (3.65b) | 7.725 (4.30b) | 7.495 (5.60b) |
| `neg_act_entropy` | 9.726 (3.00b) | 8.321 (3.26b) | 8.171 (3.65b) | 7.968 (4.30b) | 7.702 (5.60b) |
| `random` | 9.726 (3.00b) | 9.646 (3.26b) | 9.553 (3.65b) | 9.313 (4.30b) | 8.883 (5.60b) |

## PPL gap vs random at each k (negative = signal beats random)

| signal | k=0.0 | k=0.02 | k=0.05 | k=0.1 | k=0.2 |
|---|---|---|---|---|---|
| `act_entropy` | +0.000 | +0.019 | +0.003 | +0.055 | +0.172 |
| `act_max` | +0.000 | -1.802 | -1.922 | -1.807 | -1.567 |
| `act_max*act_kurt` | +0.000 | -1.475 | -1.490 | -1.441 | -1.309 |
| `act_max*act_rms` | +0.000 | -1.728 | -1.832 | -1.745 | -1.542 |
| `act_scale` | +0.000 | -1.590 | -1.664 | -1.588 | -1.388 |
| `neg_act_entropy` | +0.000 | -1.324 | -1.383 | -1.345 | -1.181 |

