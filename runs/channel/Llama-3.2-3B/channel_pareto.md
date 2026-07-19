# Per-channel protection — meta-llama/Llama-3.2-3B

Backend `hqq`, base 4-bit, protected columns at 16-bit, canonical PPL. FP16 PPL = **7.8167**.
Rows = signal used to pick protected channels; `random` is the control. **At each k (same effective bits), signal < random means per-channel importance is real.**

## PPL by protection fraction k (effective bits in parentheses)

| signal | k=0.0 | k=0.02 | k=0.05 | k=0.1 | k=0.2 |
|---|---|---|---|---|---|
| `act_scale` | 8.387 (4.00b) | 8.205 (4.24b) | 8.171 (4.60b) | 8.146 (5.20b) | 8.096 (6.40b) |
| `hessian_diag` | 8.387 (4.00b) | 8.208 (4.24b) | 8.186 (4.60b) | 8.162 (5.20b) | 8.109 (6.40b) |
| `magnitude` | 8.387 (4.00b) | 8.369 (4.24b) | 8.357 (4.60b) | 8.342 (5.20b) | 8.299 (6.40b) |
| `random` | 8.387 (4.00b) | 8.375 (4.24b) | 8.358 (4.60b) | 8.329 (5.20b) | 8.293 (6.40b) |
| `salience` | 8.387 (4.00b) | 8.215 (4.24b) | 8.191 (4.60b) | 8.167 (5.20b) | 8.113 (6.40b) |

## PPL gap vs random at each k (negative = signal beats random)

| signal | k=0.0 | k=0.02 | k=0.05 | k=0.1 | k=0.2 |
|---|---|---|---|---|---|
| `act_scale` | +0.000 | -0.171 | -0.187 | -0.183 | -0.197 |
| `hessian_diag` | +0.000 | -0.168 | -0.172 | -0.167 | -0.183 |
| `magnitude` | +0.000 | -0.007 | -0.001 | +0.013 | +0.006 |
| `salience` | +0.000 | -0.160 | -0.167 | -0.162 | -0.180 |

