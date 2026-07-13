# Pareto: PPL vs effective bits — meta-llama/Llama-3.1-8B

Backend `hqq`, levels [3, 4, 8], canonical PPL. FP16 baseline PPL = **6.2384**.
Each signal is used in its native high→more-bits direction (how SEQ used entropy). Lower PPL at equal effective bits is better; `random` is the chance control.

## PPL at each target budget (effective bits in parentheses)

| signal | ~3.0 bits | ~3.5 bits | ~4.0 bits | ~5.0 bits | ~6.0 bits | ~7.0 bits |
|---|---|---|---|---|---|---|
| `entropy` | 9.726 (3.00) | 8.305 (3.50) | 6.950 (4.00) | 6.795 (5.02) | 6.721 (6.02) | 6.550 (7.02) |
| `hessian_diag` | 9.726 (3.00) | 9.112 (3.50) | 8.587 (4.01) | 7.901 (5.02) | 7.733 (6.02) | 6.427 (7.01) |
| `magnitude` | 9.726 (3.00) | 8.454 (3.50) | 7.923 (4.01) | 6.800 (5.02) | 6.712 (6.02) | 6.530 (7.02) |
| `random` | 9.726 (3.00) | 8.525 (3.50) | 7.227 (4.01) | 7.004 (5.02) | 6.717 (6.02) | 6.391 (7.00) |
| `salience` | 9.726 (3.00) | 8.586 (3.50) | 8.006 (4.01) | 7.611 (5.02) | 7.470 (6.00) | 7.006 (7.02) |

## ΔPPL vs FP16 (lower = closer to lossless)

| signal | ~3.0 bits | ~3.5 bits | ~4.0 bits | ~5.0 bits | ~6.0 bits | ~7.0 bits |
|---|---|---|---|---|---|---|
| `entropy` | +3.487 | +2.067 | +0.711 | +0.557 | +0.482 | +0.312 |
| `hessian_diag` | +3.487 | +2.874 | +2.349 | +1.663 | +1.494 | +0.189 |
| `magnitude` | +3.487 | +2.215 | +1.685 | +0.562 | +0.474 | +0.292 |
| `random` | +3.487 | +2.286 | +0.989 | +0.766 | +0.478 | +0.152 |
| `salience` | +3.487 | +2.347 | +1.768 | +1.373 | +1.231 | +0.767 |

