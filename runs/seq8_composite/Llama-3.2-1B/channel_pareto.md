# Per-channel protection — meta-llama/Llama-3.2-1B

Backend `hqq`, base 3-bit, protected columns at 16-bit, canonical PPL. FP16 PPL = **9.7573**.
Rows = signal used to pick protected channels; `random` is the control. **At each k (same effective bits), signal < random means per-channel importance is real.**

## PPL by protection fraction k (effective bits in parentheses)

| signal | k=0.0 | k=0.02 | k=0.05 | k=0.1 | k=0.2 |
|---|---|---|---|---|---|
| `act_entropy` | 32.430 (3.00b) | 31.723 (3.26b) | 30.563 (3.65b) | 27.872 (4.30b) | 24.756 (5.60b) |
| `act_max` | 32.430 (3.00b) | 14.976 (3.26b) | 14.024 (3.65b) | 13.276 (4.30b) | 12.519 (5.60b) |
| `act_max*act_kurt` | 32.430 (3.00b) | 16.321 (3.26b) | 15.849 (3.65b) | 15.243 (4.30b) | 13.899 (5.60b) |
| `act_max*act_rms` | 32.430 (3.00b) | 15.250 (3.26b) | 14.446 (3.65b) | 13.785 (4.30b) | 12.928 (5.60b) |
| `act_scale` | 32.430 (3.00b) | 16.112 (3.26b) | 15.517 (3.65b) | 14.799 (4.30b) | 13.802 (5.60b) |
| `neg_act_entropy` | 32.430 (3.00b) | 16.891 (3.26b) | 16.031 (3.65b) | 15.499 (4.30b) | 14.379 (5.60b) |
| `random` | 32.430 (3.00b) | 31.888 (3.26b) | 30.741 (3.65b) | 28.014 (4.30b) | 24.060 (5.60b) |

## PPL gap vs random at each k (negative = signal beats random)

| signal | k=0.0 | k=0.02 | k=0.05 | k=0.1 | k=0.2 |
|---|---|---|---|---|---|
| `act_entropy` | +0.000 | -0.165 | -0.177 | -0.142 | +0.696 |
| `act_max` | +0.000 | -16.912 | -16.717 | -14.737 | -11.541 |
| `act_max*act_kurt` | +0.000 | -15.567 | -14.892 | -12.770 | -10.161 |
| `act_max*act_rms` | +0.000 | -16.638 | -16.294 | -14.229 | -11.132 |
| `act_scale` | +0.000 | -15.776 | -15.224 | -13.215 | -10.257 |
| `neg_act_entropy` | +0.000 | -14.997 | -14.710 | -12.515 | -9.681 |

