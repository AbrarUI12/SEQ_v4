# Per-channel protection — meta-llama/Llama-3.1-8B

Backend `hqq`, base 4-bit, protected columns at 16-bit, canonical PPL. FP16 PPL = **6.2384**.
Rows = signal used to pick protected channels; `random` is the control. **At each k (same effective bits), signal < random means per-channel importance is real.**

## PPL by protection fraction k (effective bits in parentheses)

| signal | k=0.0 | k=0.01 | k=0.02 | k=0.05 | k=0.1 |
|---|---|---|---|---|---|
| `act_entropy` | 6.783 (4.00b) | 6.780 (4.12b) | 6.776 (4.24b) | 6.765 (4.60b) | 6.747 (5.20b) |
| `act_kurt` | 6.783 (4.00b) | 6.749 (4.12b) | 6.637 (4.24b) | 6.625 (4.60b) | 6.598 (5.20b) |
| `act_max` | 6.783 (4.00b) | 6.583 (4.12b) | 6.552 (4.24b) | 6.520 (4.60b) | 6.499 (5.20b) |
| `act_rms` | 6.783 (4.00b) | 6.611 (4.12b) | 6.593 (4.24b) | 6.567 (4.60b) | 6.535 (5.20b) |
| `act_scale` | 6.783 (4.00b) | 6.614 (4.12b) | 6.597 (4.24b) | 6.569 (4.60b) | 6.540 (5.20b) |
| `hessian_diag` | 6.783 (4.00b) | 6.622 (4.12b) | 6.606 (4.24b) | 6.584 (4.60b) | 6.550 (5.20b) |
| `random` | 6.783 (4.00b) | 6.778 (4.12b) | 6.775 (4.24b) | 6.759 (4.60b) | 6.728 (5.20b) |

## PPL gap vs random at each k (negative = signal beats random)

| signal | k=0.0 | k=0.01 | k=0.02 | k=0.05 | k=0.1 |
|---|---|---|---|---|---|
| `act_entropy` | +0.000 | +0.001 | +0.000 | +0.006 | +0.019 |
| `act_kurt` | +0.000 | -0.029 | -0.138 | -0.134 | -0.130 |
| `act_max` | +0.000 | -0.195 | -0.224 | -0.239 | -0.230 |
| `act_rms` | +0.000 | -0.167 | -0.182 | -0.192 | -0.193 |
| `act_scale` | +0.000 | -0.165 | -0.178 | -0.190 | -0.188 |
| `hessian_diag` | +0.000 | -0.156 | -0.169 | -0.175 | -0.178 |
