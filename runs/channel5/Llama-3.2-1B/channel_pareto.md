# Per-channel protection — meta-llama/Llama-3.2-1B

Backend `hqq`, base 4-bit, protected columns at 16-bit, canonical PPL. FP16 PPL = **9.7573**.
Rows = signal used to pick protected channels; `random` is the control. **At each k (same effective bits), signal < random means per-channel importance is real.**

## PPL by protection fraction k (effective bits in parentheses)

| signal | k=0.0 | k=0.01 | k=0.02 | k=0.05 | k=0.1 |
|---|---|---|---|---|---|
| `act_entropy` | 11.187 (4.00b) | 11.176 (4.12b) | 11.175 (4.24b) | 11.140 (4.60b) | 11.090 (5.20b) |
| `act_kurt` | 11.187 (4.00b) | 11.150 (4.12b) | 10.692 (4.24b) | 10.621 (4.60b) | 10.569 (5.20b) |
| `act_max` | 11.187 (4.00b) | 10.592 (4.12b) | 10.537 (4.24b) | 10.424 (4.60b) | 10.332 (5.20b) |
| `act_rms` | 11.187 (4.00b) | 10.644 (4.12b) | 10.611 (4.24b) | 10.568 (4.60b) | 10.502 (5.20b) |
| `act_scale` | 11.187 (4.00b) | 10.647 (4.12b) | 10.621 (4.24b) | 10.571 (4.60b) | 10.504 (5.20b) |
| `hessian_diag` | 11.187 (4.00b) | 10.661 (4.12b) | 10.635 (4.24b) | 10.577 (4.60b) | 10.529 (5.20b) |
| `random` | 11.187 (4.00b) | 11.178 (4.12b) | 11.160 (4.24b) | 11.132 (4.60b) | 11.089 (5.20b) |

## PPL gap vs random at each k (negative = signal beats random)

| signal | k=0.0 | k=0.01 | k=0.02 | k=0.05 | k=0.1 |
|---|---|---|---|---|---|
| `act_entropy` | +0.000 | -0.001 | +0.015 | +0.007 | +0.001 |
| `act_kurt` | +0.000 | -0.027 | -0.468 | -0.511 | -0.520 |
| `act_max` | +0.000 | -0.585 | -0.623 | -0.708 | -0.757 |
| `act_rms` | +0.000 | -0.534 | -0.549 | -0.564 | -0.587 |
| `act_scale` | +0.000 | -0.531 | -0.539 | -0.562 | -0.585 |
| `hessian_diag` | +0.000 | -0.517 | -0.525 | -0.555 | -0.560 |
