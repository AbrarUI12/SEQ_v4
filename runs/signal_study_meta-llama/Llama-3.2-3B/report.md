# Signal-quality study — meta-llama/Llama-3.2-3B

- modules: 197
- baseline FP16 ppl: 8.149860613361158
- degrade bits: 3

## RQ1/RQ2 — signal vs. measured sensitivity

| rank | signal | Spearman ρ | Kendall τ | Pearson | n |
|---|---|---|---|---|---|
| 1 | `entropy` | 0.121 | 0.082 | 0.046 | 197 |
| 2 | `salience` | -0.029 | -0.024 | 0.128 | 197 |
| 3 | `hessian_diag` | -0.060 | -0.046 | 0.158 | 197 |
| 4 | `outlier_frac` | -0.123 | -0.082 | -0.071 | 197 |
| 5 | `act_scale` | -0.146 | -0.097 | -0.028 | 197 |
| 6 | `kurtosis` | -0.164 | -0.108 | -0.045 | 197 |
| 7 | `magnitude` | -0.191 | -0.130 | -0.055 | 197 |

## Downstream allocation

- best signal: `entropy`
- target effective bits: 6.0
- achieved effective bits: 6.008078335373317
- params by bits: {8: 1711276032, 4: 1107296256, 3: 394002432}
