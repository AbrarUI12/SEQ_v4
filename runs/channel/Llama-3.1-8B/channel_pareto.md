# Per-channel protection — meta-llama/Llama-3.1-8B

Backend `hqq`, base 4-bit, protected columns at 16-bit, canonical PPL. FP16 PPL = **6.2384**.
Rows = signal used to pick protected channels; `random` is the control. **At each k (same effective bits), signal < random means per-channel importance is real.**

## PPL by protection fraction k (effective bits in parentheses)

| signal | k=0.0 | k=0.02 | k=0.05 | k=0.1 | k=0.2 |
|---|---|---|---|---|---|
| `act_scale` | 6.783 (4.00b) | 6.597 (4.24b) | 6.569 (4.60b) | 6.539 (5.20b) | 6.499 (6.40b) |
| `hessian_diag` | 6.783 (4.00b) | 6.606 (4.24b) | 6.585 (4.60b) | 6.550 (5.20b) | 6.501 (6.40b) |
| `magnitude` | 6.783 (4.00b) | 6.749 (4.24b) | 6.734 (4.60b) | 6.713 (5.20b) | 6.680 (6.40b) |
| `random` | 6.783 (4.00b) | 6.775 (4.24b) | 6.759 (4.60b) | 6.728 (5.20b) | 6.674 (6.40b) |
| `salience` | 6.783 (4.00b) | 6.611 (4.24b) | 6.586 (4.60b) | 6.552 (5.20b) | 6.503 (6.40b) |

## PPL gap vs random at each k (negative = signal beats random)

| signal | k=0.0 | k=0.02 | k=0.05 | k=0.1 | k=0.2 |
|---|---|---|---|---|---|
| `act_scale` | +0.000 | -0.178 | -0.190 | -0.189 | -0.175 |
| `hessian_diag` | +0.000 | -0.169 | -0.175 | -0.178 | -0.173 |
| `magnitude` | +0.000 | -0.026 | -0.025 | -0.015 | +0.006 |
| `salience` | +0.000 | -0.164 | -0.173 | -0.176 | -0.171 |

