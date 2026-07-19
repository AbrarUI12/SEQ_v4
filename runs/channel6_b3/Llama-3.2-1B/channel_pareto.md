# Per-channel protection — meta-llama/Llama-3.2-1B

Backend `hqq`, base 3-bit, protected columns at 16-bit, canonical PPL. FP16 PPL = **9.7573**.
Rows = signal used to pick protected channels; `random` is the control. **At each k (same effective bits), signal < random means per-channel importance is real.**

## PPL by protection fraction k (effective bits in parentheses)

| signal | k=0.0 | k=0.02 | k=0.05 | k=0.1 | k=0.2 | k=0.3 |
|---|---|---|---|---|---|---|
| `act_max` | 32.430 (3.00b) | 14.976 (3.26b) | 14.024 (3.65b) | 13.276 (4.30b) | 12.519 (5.60b) | 11.971 (6.90b) |
| `act_rms` | 32.430 (3.00b) | 16.039 (3.26b) | 15.525 (3.65b) | 14.781 (4.30b) | 13.763 (5.60b) | 12.926 (6.90b) |
| `act_scale` | 32.430 (3.00b) | 16.112 (3.26b) | 15.517 (3.65b) | 14.799 (4.30b) | 13.802 (5.60b) | 12.996 (6.90b) |
| `random` | 32.430 (3.00b) | 31.888 (3.26b) | 30.741 (3.65b) | 28.014 (4.30b) | 24.060 (5.60b) | 21.198 (6.90b) |

## PPL gap vs random at each k (negative = signal beats random)

| signal | k=0.0 | k=0.02 | k=0.05 | k=0.1 | k=0.2 | k=0.3 |
|---|---|---|---|---|---|---|
| `act_max` | +0.000 | -16.912 | -16.717 | -14.737 | -11.541 | -9.227 |
| `act_rms` | +0.000 | -15.849 | -15.216 | -13.233 | -10.297 | -8.272 |
| `act_scale` | +0.000 | -15.776 | -15.224 | -13.215 | -10.257 | -8.203 |
