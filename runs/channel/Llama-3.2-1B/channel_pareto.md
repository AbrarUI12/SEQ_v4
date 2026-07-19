# Per-channel protection — meta-llama/Llama-3.2-1B

Backend `hqq`, base 4-bit, protected columns at 16-bit, canonical PPL. FP16 PPL = **9.7572**.
Rows = signal used to pick protected channels; `random` is the control. **At each k (same effective bits), signal < random means per-channel importance is real.**

## PPL by protection fraction k (effective bits in parentheses)

| signal | k=0.0 | k=0.02 | k=0.05 | k=0.1 | k=0.2 |
|---|---|---|---|---|---|
| `act_scale` | 11.187 (4.00b) | 10.621 (4.24b) | 10.570 (4.60b) | 10.504 (5.20b) | 10.399 (6.40b) |
| `hessian_diag` | 11.187 (4.00b) | 10.635 (4.24b) | 10.577 (4.60b) | 10.529 (5.20b) | 10.435 (6.40b) |
| `magnitude` | 11.187 (4.00b) | 11.140 (4.24b) | 11.096 (4.60b) | 11.055 (5.20b) | 10.984 (6.40b) |
| `random` | 11.187 (4.00b) | 11.160 (4.24b) | 11.132 (4.60b) | 11.089 (5.20b) | 10.993 (6.40b) |
| `salience` | 11.187 (4.00b) | 10.638 (4.24b) | 10.586 (4.60b) | 10.535 (5.20b) | 10.437 (6.40b) |

## PPL gap vs random at each k (negative = signal beats random)

| signal | k=0.0 | k=0.02 | k=0.05 | k=0.1 | k=0.2 |
|---|---|---|---|---|---|
| `act_scale` | +0.000 | -0.539 | -0.562 | -0.586 | -0.594 |
| `hessian_diag` | +0.000 | -0.525 | -0.555 | -0.560 | -0.558 |
| `magnitude` | +0.000 | -0.020 | -0.036 | -0.035 | -0.009 |
| `salience` | +0.000 | -0.522 | -0.546 | -0.554 | -0.556 |

