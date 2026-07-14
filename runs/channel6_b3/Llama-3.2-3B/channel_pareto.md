# Per-channel protection — meta-llama/Llama-3.2-3B

Backend `hqq`, base 3-bit, protected columns at 16-bit, canonical PPL. FP16 PPL = **7.8165**.
Rows = signal used to pick protected channels; `random` is the control. **At each k (same effective bits), signal < random means per-channel importance is real.**

## PPL by protection fraction k (effective bits in parentheses)

| signal | k=0.0 | k=0.02 | k=0.05 | k=0.1 | k=0.2 | k=0.3 |
|---|---|---|---|---|---|---|
| `act_max` | 13.778 (3.00b) | 9.843 (3.26b) | 9.547 (3.65b) | 9.301 (4.30b) | 9.044 (5.60b) | 8.821 (6.90b) |
| `act_rms` | 13.778 (3.00b) | 10.020 (3.26b) | 9.848 (3.65b) | 9.652 (4.30b) | 9.313 (5.60b) | 9.062 (6.90b) |
| `act_scale` | 13.778 (3.00b) | 10.045 (3.26b) | 9.858 (3.65b) | 9.657 (4.30b) | 9.343 (5.60b) | 9.073 (6.90b) |
| `random` | 13.778 (3.00b) | 13.541 (3.26b) | 13.310 (3.65b) | 12.941 (4.30b) | 12.356 (5.60b) | 11.834 (6.90b) |

## PPL gap vs random at each k (negative = signal beats random)

| signal | k=0.0 | k=0.02 | k=0.05 | k=0.1 | k=0.2 | k=0.3 |
|---|---|---|---|---|---|---|
| `act_max` | +0.000 | -3.698 | -3.763 | -3.641 | -3.312 | -3.012 |
| `act_rms` | +0.000 | -3.521 | -3.462 | -3.289 | -3.043 | -2.772 |
| `act_scale` | +0.000 | -3.495 | -3.452 | -3.284 | -3.013 | -2.761 |
