# Pareto: PPL vs effective bits — meta-llama/Llama-3.1-8B

Backend `hqq`, levels [3, 4, 8], canonical PPL. FP16 baseline PPL = **nan**.
Each signal is used in its native high→more-bits direction (how SEQ used entropy). Lower PPL at equal effective bits is better; `random` is the chance control.

## PPL at each target budget (effective bits in parentheses)

| signal | ~3.0 bits | ~3.5 bits | ~4.0 bits | ~5.0 bits | ~6.0 bits | ~7.0 bits |
|---|---|---|---|---|---|---|
| `entropy` | — | — | — | — | — | — |
| `hessian_diag` | — | — | — | — | — | — |
| `magnitude` | — | — | — | — | — | — |
| `random` | — | — | — | — | — | — |
| `salience` | — | — | — | — | — | — |

## ΔPPL vs FP16 (lower = closer to lossless)

| signal | ~3.0 bits | ~3.5 bits | ~4.0 bits | ~5.0 bits | ~6.0 bits | ~7.0 bits |
|---|---|---|---|---|---|---|
| `entropy` | — | — | — | — | — | — |
| `hessian_diag` | — | — | — | — | — | — |
| `magnitude` | — | — | — | — | — | — |
| `random` | — | — | — | — | — | — |
| `salience` | — | — | — | — | — | — |

