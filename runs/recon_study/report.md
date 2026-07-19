# Signal-quality study — meta-llama/Llama-3.2-1B

- modules: 113

## RQ1/RQ2 — signal vs. measured sensitivity

| rank | signal | Spearman ρ | Kendall τ | Pearson | n |
|---|---|---|---|---|---|
| 1 | `hessian_diag` | 0.997 | 0.966 | 1.000 | 113 |
| 2 | `salience` | 0.992 | 0.941 | 0.987 | 113 |
| 3 | `salience_pp` | 0.990 | 0.936 | 0.986 | 113 |
| 4 | `hessian_diag_pp` | 0.697 | 0.547 | 0.878 | 113 |
| 5 | `act_scale` | 0.632 | 0.488 | 0.698 | 113 |
| 6 | `magnitude` | 0.599 | 0.366 | 0.018 | 113 |
| 7 | `kurtosis` | 0.488 | 0.297 | -0.037 | 113 |
| 8 | `outlier_frac` | 0.259 | 0.162 | -0.073 | 113 |
| 9 | `entropy` | -0.177 | -0.107 | 0.088 | 113 |

## Downstream allocation

- best signal: `hessian_diag`
- target effective bits: 6.0
- achieved effective bits: 6.030547305897327
- params by bits: {8: 707264512, 3: 319815680, 4: 208666624}
