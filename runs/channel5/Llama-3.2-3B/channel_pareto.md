# Per-channel protection — meta-llama/Llama-3.2-3B

Backend `hqq`, base 4-bit, protected columns at 16-bit, canonical PPL. FP16 PPL = **7.8165**.
Rows = signal used to pick protected channels; `random` is the control. **At each k (same effective bits), signal < random means per-channel importance is real.**

## PPL by protection fraction k (effective bits in parentheses)

| signal | k=0.0 | k=0.01 | k=0.02 | k=0.05 | k=0.1 |
|---|---|---|---|---|---|
| `act_entropy` | 8.387 (4.00b) | 8.385 (4.12b) | 8.383 (4.24b) | 8.376 (4.60b) | 8.355 (5.20b) |
| `act_kurt` | 8.387 (4.00b) | 8.353 (4.12b) | 8.341 (4.24b) | 8.189 (4.60b) | 8.168 (5.20b) |
| `act_max` | 8.387 (4.00b) | 8.193 (4.12b) | 8.160 (4.24b) | 8.125 (4.60b) | 8.091 (5.20b) |
| `act_rms` | 8.387 (4.00b) | 8.214 (4.12b) | 8.200 (4.24b) | 8.169 (4.60b) | 8.141 (5.20b) |
| `act_scale` | 8.387 (4.00b) | 8.221 (4.12b) | 8.205 (4.24b) | 8.171 (4.60b) | 8.146 (5.20b) |
| `hessian_diag` | 8.387 (4.00b) | 8.218 (4.12b) | 8.208 (4.24b) | 8.186 (4.60b) | 8.162 (5.20b) |
| `random` | 8.387 (4.00b) | 8.381 (4.12b) | 8.375 (4.24b) | 8.358 (4.60b) | 8.329 (5.20b) |

## PPL gap vs random at each k (negative = signal beats random)

| signal | k=0.0 | k=0.01 | k=0.02 | k=0.05 | k=0.1 |
|---|---|---|---|---|---|
| `act_entropy` | +0.000 | +0.004 | +0.008 | +0.018 | +0.026 |
| `act_kurt` | +0.000 | -0.027 | -0.034 | -0.169 | -0.161 |
| `act_max` | +0.000 | -0.188 | -0.215 | -0.233 | -0.238 |
| `act_rms` | +0.000 | -0.167 | -0.176 | -0.189 | -0.188 |
| `act_scale` | +0.000 | -0.159 | -0.171 | -0.187 | -0.183 |
| `hessian_diag` | +0.000 | -0.162 | -0.168 | -0.172 | -0.167 |
