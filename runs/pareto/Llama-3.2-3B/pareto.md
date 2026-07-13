# Pareto: PPL vs effective bits — meta-llama/Llama-3.2-3B

Backend `hqq`, levels [3, 4, 8], canonical PPL. FP16 baseline PPL = **7.8167**.
Each signal is used in its native high→more-bits direction (how SEQ used entropy). Lower PPL at equal effective bits is better; `random` is the chance control.

## PPL at each target budget (effective bits in parentheses)

| signal | ~3.0 bits | ~3.5 bits | ~4.0 bits | ~5.0 bits | ~6.0 bits | ~7.0 bits |
|---|---|---|---|---|---|---|
| `entropy` | 13.779 (3.00) | 11.190 (3.50) | 8.575 (4.00) | 8.375 (5.01) | 8.156 (6.01) | 8.067 (7.01) |
| `hessian_diag` | 13.779 (3.00) | 13.352 (3.61) | 12.512 (4.00) | 10.944 (5.03) | 8.885 (6.01) | 8.034 (7.01) |
| `magnitude` | 13.779 (3.00) | 11.164 (3.50) | 8.642 (4.01) | 8.381 (5.01) | 8.308 (6.01) | 8.086 (7.01) |
| `random` | 13.779 (3.00) | 11.215 (3.50) | 8.852 (4.01) | 8.574 (5.01) | 8.236 (6.00) | 7.935 (7.02) |
| `salience` | 13.779 (3.00) | 11.917 (3.50) | 10.979 (4.01) | 10.538 (5.03) | 10.307 (6.03) | 9.526 (7.01) |

## ΔPPL vs FP16 (lower = closer to lossless)

| signal | ~3.0 bits | ~3.5 bits | ~4.0 bits | ~5.0 bits | ~6.0 bits | ~7.0 bits |
|---|---|---|---|---|---|---|
| `entropy` | +5.962 | +3.373 | +0.758 | +0.559 | +0.339 | +0.251 |
| `hessian_diag` | +5.962 | +5.535 | +4.695 | +3.127 | +1.069 | +0.218 |
| `magnitude` | +5.962 | +3.347 | +0.825 | +0.564 | +0.491 | +0.269 |
| `random` | +5.962 | +3.398 | +1.036 | +0.757 | +0.419 | +0.118 |
| `salience` | +5.962 | +4.100 | +3.163 | +2.721 | +2.491 | +1.709 |

