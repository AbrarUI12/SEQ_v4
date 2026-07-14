# Per-channel protection — meta-llama/Llama-3.2-1B

Backend `hqq`, base 2-bit, protected columns at 16-bit, canonical PPL. FP16 PPL = **9.7573**.
Rows = signal used to pick protected channels; `random` is the control. **At each k (same effective bits), signal < random means per-channel importance is real.**

## PPL by protection fraction k (effective bits in parentheses)

| signal | k=0.0 | k=0.02 | k=0.05 | k=0.1 | k=0.2 | k=0.3 |
|---|---|---|---|---|---|---|
| `act_max` | 96033.292 (2.00b) | 10779.176 (2.28b) | 5239.422 (2.70b) | 2837.364 (3.40b) | 834.535 (4.80b) | 229.228 (6.20b) |
| `act_rms` | 96033.292 (2.00b) | 9315.044 (2.28b) | 7499.939 (2.70b) | 4402.646 (3.40b) | 1928.106 (4.80b) | 1061.001 (6.20b) |
| `act_scale` | 96033.292 (2.00b) | 10656.055 (2.28b) | 7000.485 (2.70b) | 4444.923 (3.40b) | 1989.847 (4.80b) | 1097.398 (6.20b) |
| `random` | 96033.292 (2.00b) | 119036.528 (2.28b) | 84137.821 (2.70b) | 94304.957 (3.40b) | 73935.910 (4.80b) | 32615.850 (6.20b) |

## PPL gap vs random at each k (negative = signal beats random)

| signal | k=0.0 | k=0.02 | k=0.05 | k=0.1 | k=0.2 | k=0.3 |
|---|---|---|---|---|---|---|
| `act_max` | +0.000 | -108257.352 | -78898.399 | -91467.592 | -73101.375 | -32386.622 |
| `act_rms` | +0.000 | -109721.484 | -76637.882 | -89902.311 | -72007.805 | -31554.849 |
| `act_scale` | +0.000 | -108380.473 | -77137.336 | -89860.033 | -71946.063 | -31518.452 |
