# Per-channel protection — meta-llama/Llama-3.2-1B

Backend `hqq`, base 4-bit, protected columns at 16-bit, canonical PPL. FP16 PPL = **9.7573**.
Rows = signal used to pick protected channels; `random` is the control. **At each k (same effective bits), signal < random means per-channel importance is real.**

## PPL by protection fraction k (effective bits in parentheses)

| signal | k=0.0 | k=0.02 | k=0.05 | k=0.1 | k=0.2 |
|---|---|---|---|---|---|
| `act_max` | 6717.578 (4.00b) | 11.006 (4.24b) | 10.850 (4.60b) | 10.722 (5.20b) | 10.558 (6.40b) |
| `act_scale` | 6717.578 (4.00b) | 4946.108 (4.24b) | 5102.981 (4.60b) | 4773.911 (5.20b) | 4202.373 (6.40b) |
| `random` | 6717.578 (4.00b) | 6340.413 (4.24b) | 5887.381 (4.60b) | 4546.685 (5.20b) | 3440.982 (6.40b) |

## PPL gap vs random at each k (negative = signal beats random)

| signal | k=0.0 | k=0.02 | k=0.05 | k=0.1 | k=0.2 |
|---|---|---|---|---|---|
| `act_max` | +0.000 | -6329.407 | -5876.531 | -4535.963 | -3430.424 |
| `act_scale` | +0.000 | -1394.305 | -784.400 | +227.226 | +761.391 |

