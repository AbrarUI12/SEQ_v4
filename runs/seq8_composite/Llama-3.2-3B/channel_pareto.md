# Per-channel protection — meta-llama/Llama-3.2-3B

Backend `hqq`, base 3-bit, protected columns at 16-bit, canonical PPL. FP16 PPL = **7.8165**.
Rows = signal used to pick protected channels; `random` is the control. **At each k (same effective bits), signal < random means per-channel importance is real.**

## PPL by protection fraction k (effective bits in parentheses)

| signal | k=0.0 | k=0.02 | k=0.05 | k=0.1 | k=0.2 |
|---|---|---|---|---|---|
| `act_entropy` | 13.778 (3.00b) | 13.691 (3.26b) | 13.504 (3.65b) | 13.157 (4.30b) | 12.545 (5.60b) |
| `act_max` | 13.778 (3.00b) | 9.843 (3.26b) | 9.547 (3.65b) | 9.301 (4.30b) | 9.044 (5.60b) |
| `act_max*act_kurt` | 13.778 (3.00b) | 10.181 (3.26b) | 10.036 (3.65b) | 9.814 (4.30b) | 9.441 (5.60b) |
| `act_max*act_rms` | 13.778 (3.00b) | 9.857 (3.26b) | 9.628 (3.65b) | 9.420 (4.30b) | 9.133 (5.60b) |
| `act_scale` | 13.778 (3.00b) | 10.045 (3.26b) | 9.858 (3.65b) | 9.657 (4.30b) | 9.343 (5.60b) |
| `neg_act_entropy` | 13.778 (3.00b) | 10.321 (3.26b) | 10.115 (3.65b) | 9.919 (4.30b) | 9.574 (5.60b) |
| `random` | 13.778 (3.00b) | 13.541 (3.26b) | 13.310 (3.65b) | 12.941 (4.30b) | 12.356 (5.60b) |

## PPL gap vs random at each k (negative = signal beats random)

| signal | k=0.0 | k=0.02 | k=0.05 | k=0.1 | k=0.2 |
|---|---|---|---|---|---|
| `act_entropy` | +0.000 | +0.150 | +0.194 | +0.216 | +0.189 |
| `act_max` | +0.000 | -3.698 | -3.763 | -3.641 | -3.312 |
| `act_max*act_kurt` | +0.000 | -3.360 | -3.274 | -3.127 | -2.915 |
| `act_max*act_rms` | +0.000 | -3.684 | -3.682 | -3.521 | -3.222 |
| `act_scale` | +0.000 | -3.495 | -3.452 | -3.284 | -3.013 |
| `neg_act_entropy` | +0.000 | -3.220 | -3.194 | -3.022 | -2.782 |

