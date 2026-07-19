# Per-channel protection — meta-llama/Llama-3.2-3B

Backend `hqq`, base 2-bit, protected columns at 16-bit, canonical PPL. FP16 PPL = **7.8165**.
Rows = signal used to pick protected channels; `random` is the control. **At each k (same effective bits), signal < random means per-channel importance is real.**

## PPL by protection fraction k (effective bits in parentheses)

| signal | k=0.0 | k=0.02 | k=0.05 | k=0.1 | k=0.2 | k=0.3 |
|---|---|---|---|---|---|---|
| `act_max` | 20201.533 (2.00b) | 1238.606 (2.28b) | 466.637 (2.70b) | 145.462 (3.40b) | 47.049 (4.80b) | 26.847 (6.20b) |
| `act_rms` | 20201.533 (2.00b) | 1391.654 (2.28b) | 935.358 (2.70b) | 499.343 (3.40b) | 187.150 (4.80b) | 74.789 (6.20b) |
| `act_scale` | 20201.533 (2.00b) | 1535.288 (2.28b) | 912.148 (2.70b) | 484.539 (3.40b) | 212.425 (4.80b) | 81.837 (6.20b) |
| `random` | 20201.533 (2.00b) | 26979.633 (2.28b) | 32716.043 (2.70b) | 25720.599 (3.40b) | 14941.535 (4.80b) | 12843.264 (6.20b) |

## PPL gap vs random at each k (negative = signal beats random)

| signal | k=0.0 | k=0.02 | k=0.05 | k=0.1 | k=0.2 | k=0.3 |
|---|---|---|---|---|---|---|
| `act_max` | +0.000 | -25741.026 | -32249.406 | -25575.137 | -14894.486 | -12816.416 |
| `act_rms` | +0.000 | -25587.978 | -31780.685 | -25221.256 | -14754.385 | -12768.474 |
| `act_scale` | +0.000 | -25444.345 | -31803.895 | -25236.060 | -14729.110 | -12761.426 |
