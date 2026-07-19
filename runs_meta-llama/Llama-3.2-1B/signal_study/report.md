# Signal-quality study — meta-llama/Llama-3.2-1B

- modules: 113
- baseline FP16 ppl: 10.625045066517236
- degrade bits: 3

## RQ1/RQ2 — signal vs. measured sensitivity

| rank | signal | Spearman ρ | Kendall τ | Pearson | n |
|---|---|---|---|---|---|
| 1 | `entropy` | 0.164 | 0.116 | 0.137 | 113 |
| 2 | `hessian_diag` | 0.002 | 0.008 | 0.213 | 113 |
| 3 | `salience` | -0.003 | 0.002 | 0.192 | 113 |
| 4 | `act_scale` | -0.007 | -0.008 | 0.041 | 113 |
| 5 | `magnitude` | -0.145 | -0.099 | -0.078 | 113 |
| 6 | `outlier_frac` | -0.152 | -0.106 | -0.148 | 113 |
| 7 | `kurtosis` | -0.165 | -0.109 | -0.087 | 113 |

## Downstream allocation

- best signal: `entropy`
- target effective bits: 6.0
- achieved effective bits: 6.014000848536275
- params by bits: {8: 687865856, 4: 285212672, 3: 262668288}
