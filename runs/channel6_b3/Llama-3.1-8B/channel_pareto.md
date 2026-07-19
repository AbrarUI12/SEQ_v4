# Per-channel protection — meta-llama/Llama-3.1-8B

Backend `hqq`, base 3-bit, protected columns at 16-bit, canonical PPL. FP16 PPL = **6.2384**.
Rows = signal used to pick protected channels; `random` is the control. **At each k (same effective bits), signal < random means per-channel importance is real.**

## PPL by protection fraction k (effective bits in parentheses)

| signal | k=0.0 | k=0.02 | k=0.05 | k=0.1 | k=0.2 | k=0.3 |
|---|---|---|---|---|---|---|
| `act_max` | 9.726 (3.00b) | 7.843 (3.26b) | 7.631 (3.65b) | 7.506 (4.30b) | 7.316 (5.60b) | 7.140 (6.90b) |
| `act_rms` | 9.726 (3.00b) | 8.048 (3.26b) | 7.875 (3.65b) | 7.707 (4.30b) | 7.475 (5.60b) | 7.280 (6.90b) |
| `act_scale` | 9.726 (3.00b) | 8.056 (3.26b) | 7.889 (3.65b) | 7.725 (4.30b) | 7.495 (5.60b) | 7.298 (6.90b) |
| `random` | 9.726 (3.00b) | 9.646 (3.26b) | 9.553 (3.65b) | 9.313 (4.30b) | 8.883 (5.60b) | 8.529 (6.90b) |

## PPL gap vs random at each k (negative = signal beats random)

| signal | k=0.0 | k=0.02 | k=0.05 | k=0.1 | k=0.2 | k=0.3 |
|---|---|---|---|---|---|---|
| `act_max` | +0.000 | -1.802 | -1.922 | -1.807 | -1.567 | -1.389 |
| `act_rms` | +0.000 | -1.598 | -1.679 | -1.606 | -1.408 | -1.249 |
| `act_scale` | +0.000 | -1.590 | -1.664 | -1.588 | -1.388 | -1.231 |
