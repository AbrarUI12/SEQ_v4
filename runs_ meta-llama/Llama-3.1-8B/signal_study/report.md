# Signal-quality study — meta-llama/Llama-3.1-8B

- modules: 225
- baseline FP16 ppl: 6.921339317803182
- degrade bits: 3

## RQ1/RQ2 — signal vs. measured sensitivity

| rank | signal | Spearman ρ | Kendall τ | Pearson | n |
|---|---|---|---|---|---|
| 1 | `entropy` | 0.188 | 0.118 | 0.041 | 225 |
| 2 | `salience` | 0.087 | 0.057 | 0.239 | 225 |
| 3 | `hessian_diag` | 0.038 | 0.020 | 0.262 | 225 |
| 4 | `act_scale` | -0.056 | -0.051 | 0.082 | 225 |
| 5 | `magnitude` | -0.102 | -0.072 | -0.047 | 225 |
| 6 | `kurtosis` | -0.138 | -0.088 | -0.043 | 225 |
| 7 | `outlier_frac` | -0.180 | -0.107 | -0.075 | 225 |

## Downstream allocation

- best signal: `entropy`
- target effective bits: 6.0
- achieved effective bits: 6.022495458991197
- params by bits: {8: 3925868544, 4: 3053453312, 3: 525336576}
