# Pareto: PPL vs effective bits — meta-llama/Llama-3.2-1B

Backend `hqq`, levels [3, 4, 8], canonical PPL. FP16 baseline PPL = **9.7572**.
Each signal is used in its native high→more-bits direction (how SEQ used entropy). Lower PPL at equal effective bits is better; `random` is the chance control.

## PPL at each target budget (effective bits in parentheses)

| signal | ~3.0 bits | ~3.5 bits | ~4.0 bits | ~5.0 bits | ~6.0 bits | ~7.0 bits |
|---|---|---|---|---|---|---|
| `entropy` | 32.424 (3.00) | 13.196 (3.50) | 11.505 (4.00) | 11.058 (5.04) | 10.376 (6.01) | 9.863 (7.15) |
| `hessian_diag` | 32.424 (3.00) | 30.487 (4.06) | 30.487 (4.06) | 18.121 (5.03) | 16.474 (6.03) | 10.563 (7.01) |
| `magnitude` | 32.424 (3.00) | 19.029 (3.50) | 12.078 (4.00) | 11.167 (5.04) | 10.474 (6.01) | 9.863 (7.15) |
| `random` | 32.424 (3.00) | 13.721 (3.51) | 12.019 (4.03) | 10.977 (5.05) | 10.544 (6.01) | 9.887 (7.03) |
| `salience` | 32.424 (3.00) | 20.925 (3.51) | 17.334 (4.01) | 16.416 (5.04) | 15.801 (6.00) | 10.714 (7.03) |

## ΔPPL vs FP16 (lower = closer to lossless)

| signal | ~3.0 bits | ~3.5 bits | ~4.0 bits | ~5.0 bits | ~6.0 bits | ~7.0 bits |
|---|---|---|---|---|---|---|
| `entropy` | +22.667 | +3.439 | +1.748 | +1.301 | +0.619 | +0.106 |
| `hessian_diag` | +22.667 | +20.729 | +20.729 | +8.364 | +6.717 | +0.805 |
| `magnitude` | +22.667 | +9.271 | +2.320 | +1.410 | +0.717 | +0.106 |
| `random` | +22.667 | +3.964 | +2.262 | +1.220 | +0.786 | +0.130 |
| `salience` | +22.667 | +11.167 | +7.577 | +6.659 | +6.044 | +0.957 |

