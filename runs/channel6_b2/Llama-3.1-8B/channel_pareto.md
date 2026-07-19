# Per-channel protection — meta-llama/Llama-3.1-8B

Backend `hqq`, base 2-bit, protected columns at 16-bit, canonical PPL. FP16 PPL = **6.2384**.
Rows = signal used to pick protected channels; `random` is the control. **At each k (same effective bits), signal < random means per-channel importance is real.**

## PPL by protection fraction k (effective bits in parentheses)

| signal | k=0.0 | k=0.02 | k=0.05 | k=0.1 | k=0.2 | k=0.3 |
|---|---|---|---|---|---|---|
| `act_max` | 41757.412 (2.00b) | 1003.101 (2.28b) | 270.677 (2.70b) | 102.841 (3.40b) | 37.950 (4.80b) | 19.945 (6.20b) |
| `act_rms` | 41757.412 (2.00b) | 3489.396 (2.28b) | 1350.755 (2.70b) | 347.928 (3.40b) | 78.739 (4.80b) | 33.105 (6.20b) |
| `act_scale` | 41757.412 (2.00b) | 3675.634 (2.28b) | 1155.214 (2.70b) | 422.817 (3.40b) | 90.563 (4.80b) | 33.840 (6.20b) |
| `random` | 41757.412 (2.00b) | 42822.734 (2.28b) | 39157.740 (2.70b) | 34460.031 (3.40b) | 20160.485 (4.80b) | 11172.405 (6.20b) |

## PPL gap vs random at each k (negative = signal beats random)

| signal | k=0.0 | k=0.02 | k=0.05 | k=0.1 | k=0.2 | k=0.3 |
|---|---|---|---|---|---|---|
| `act_max` | +0.000 | -41819.633 | -38887.064 | -34357.190 | -20122.535 | -11152.461 |
| `act_rms` | +0.000 | -39333.338 | -37806.985 | -34112.103 | -20081.746 | -11139.300 |
| `act_scale` | +0.000 | -39147.100 | -38002.526 | -34037.214 | -20069.922 | -11138.565 |
