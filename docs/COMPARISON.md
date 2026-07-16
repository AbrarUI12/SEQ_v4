# SEQ vs baselines — actual-bits comparison

Points from SEQ sweeps + external baselines, sorted by actual bits. ★ = on the Pareto frontier (no method has both fewer bits and lower PPL). SEQ bits include the FP16 residual + index table; base group scales are common to all methods and excluded from this axis.

## meta-llama/Llama-3.2-1B  (FP16 PPL 9.7573)

| method | actual bits | nominal bits | PPL | Δ vs FP16 | frontier |
|---|---|---|---|---|---|
| GPTQ-4 g128 | 4.00 | 4.00 | 10.393 | +0.636 | ★ |
| HQQ-4 uniform | 4.00 | 4.00 | 11.190 | +1.433 |  |
| AWQ-4 g128 | 4.00 | 4.00 | 11.282 | +1.525 |  |
| SEQ:act_max(hqq-3b [8:0.26]) | 4.40 | 4.30 | 12.394 | +2.637 |  |
| SEQ:act_max(hqq-3b [16:0.1]) | 4.40 | 4.30 | 13.276 | +3.519 |  |
| SEQ:random(hqq-3b [8:0.26]) | 4.40 | 4.30 | 22.715 | +12.958 |  |
| SEQ:random(hqq-3b [16:0.1]) | 4.40 | 4.30 | 28.014 | +18.256 |  |
| SEQ:act_max(hqq-3b [16:0.05,8:0.13]) | 4.41 | 4.30 | 12.654 | +2.897 |  |
| SEQ:random(hqq-3b [16:0.05,8:0.13]) | 4.41 | 4.30 | 24.723 | +14.965 |  |
| SEQ:act_max(hqq-3b [16:0.02,8:0.22]) | 4.46 | 4.36 | 12.301 | +2.544 |  |
| SEQ:random(hqq-3b [16:0.02,8:0.22]) | 4.46 | 4.36 | 23.339 | +13.581 |  |
| FP16 | 16.00 | 16.00 | 9.757 | +0.000 | ★ |

> **No SEQ point is on the frontier** — a baseline dominates it.

## meta-llama/Llama-3.2-3B

| method | actual bits | nominal bits | PPL | Δ vs FP16 | frontier |
|---|---|---|---|---|---|
| SEQ:act_max(hqq-3b [8:0.26]) | 4.40 | 4.30 | 9.038 | — | ★ |
| SEQ:random(hqq-3b [8:0.26]) | 4.40 | 4.30 | 12.034 | — |  |
| SEQ:act_max(hqq-3b [16:0.05,8:0.13]) | 4.40 | 4.30 | 9.100 | — |  |
| SEQ:random(hqq-3b [16:0.05,8:0.13]) | 4.40 | 4.30 | 12.471 | — |  |
| SEQ:act_max(hqq-3b [16:0.1]) | 4.40 | 4.30 | 9.301 | — |  |
| SEQ:random(hqq-3b [16:0.1]) | 4.40 | 4.30 | 12.941 | — |  |
| SEQ:act_max(hqq-3b [16:0.02,8:0.22]) | 4.46 | 4.36 | 8.962 | — | ★ |
| SEQ:random(hqq-3b [16:0.02,8:0.22]) | 4.46 | 4.36 | 12.157 | — |  |

> SEQ is on the Pareto frontier (2 point(s)).
