# SEQ vs baselines — actual-bits comparison

Points from SEQ sweeps + external baselines, sorted by actual bits. ★ = on the Pareto frontier (no method has both fewer bits and lower PPL). SEQ bits include the FP16 residual + index table; base group scales are common to all methods and excluded from this axis.

## meta-llama/Llama-3.1-8B

| method | actual bits | nominal bits | PPL | Δ vs FP16 | frontier |
|---|---|---|---|---|---|
| SEQ:act_scale(hqq-4b k=0.0) | 4.00 | 4.00 | 6.783 | — | ★ |
| SEQ:act_max(hqq-4b k=0.0) | 4.00 | 4.00 | 6.783 | — | ★ |
| SEQ:random(hqq-4b k=0.0) | 4.00 | 4.00 | 6.783 | — | ★ |
| SEQ:act_max(hqq-4b k=0.01) | 4.22 | 4.12 | 6.583 | — | ★ |
| SEQ:act_scale(hqq-4b k=0.01) | 4.22 | 4.12 | 6.614 | — |  |
| SEQ:random(hqq-4b k=0.01) | 4.22 | 4.12 | 6.778 | — |  |
| SEQ:act_max(hqq-4b k=0.02) | 4.34 | 4.24 | 6.552 | — | ★ |
| SEQ:act_scale(hqq-4b k=0.02) | 4.34 | 4.24 | 6.597 | — |  |
| SEQ:random(hqq-4b k=0.02) | 4.34 | 4.24 | 6.775 | — |  |
| SEQ:act_max(hqq-4b k=0.05) | 4.70 | 4.60 | 6.520 | — | ★ |
| SEQ:act_scale(hqq-4b k=0.05) | 4.70 | 4.60 | 6.569 | — |  |
| SEQ:random(hqq-4b k=0.05) | 4.70 | 4.60 | 6.759 | — |  |
| SEQ:act_max(hqq-4b k=0.1) | 5.30 | 5.20 | 6.498 | — | ★ |
| SEQ:act_scale(hqq-4b k=0.1) | 5.30 | 5.20 | 6.540 | — |  |
| SEQ:random(hqq-4b k=0.1) | 5.30 | 5.20 | 6.728 | — |  |

> SEQ is on the Pareto frontier (7 point(s)).
## meta-llama/Llama-3.2-1B  (FP16 PPL 9.7573)

| method | actual bits | nominal bits | PPL | Δ vs FP16 | frontier |
|---|---|---|---|---|---|
| GPTQ-4 g128 | 4.00 | 4.00 | 10.393 | +0.636 | ★ |
| SEQ:act_scale(hqq-4b k=0.0) | 4.00 | 4.00 | 11.187 | +1.430 |  |
| SEQ:act_max(hqq-4b k=0.0) | 4.00 | 4.00 | 11.187 | +1.430 |  |
| SEQ:random(hqq-4b k=0.0) | 4.00 | 4.00 | 11.187 | +1.430 |  |
| HQQ-4 uniform | 4.00 | 4.00 | 11.190 | +1.433 |  |
| AWQ-4 g128 | 4.00 | 4.00 | 11.282 | +1.525 |  |
| SEQ:act_max(hqq-4b k=0.01) | 4.22 | 4.12 | 10.592 | +0.835 |  |
| SEQ:act_scale(hqq-4b k=0.01) | 4.22 | 4.12 | 10.647 | +0.889 |  |
| SEQ:random(hqq-4b k=0.01) | 4.22 | 4.12 | 11.178 | +1.420 |  |
| SEQ:act_max(hqq-4b k=0.02) | 4.34 | 4.24 | 10.537 | +0.780 |  |
| SEQ:act_scale(hqq-4b k=0.02) | 4.34 | 4.24 | 10.621 | +0.863 |  |
| SEQ:random(hqq-4b k=0.02) | 4.34 | 4.24 | 11.160 | +1.403 |  |
| SEQ:act_max(hqq-3b [8:0.26]) | 4.40 | 4.30 | 12.394 | +2.637 |  |
| SEQ:act_max(hqq-3b [16:0.1]) | 4.40 | 4.30 | 13.276 | +3.519 |  |
| SEQ:random(hqq-3b [8:0.26]) | 4.40 | 4.30 | 22.715 | +12.958 |  |
| SEQ:random(hqq-3b [16:0.1]) | 4.40 | 4.30 | 28.014 | +18.256 |  |
| SEQ:act_max(hqq-3b [16:0.05,8:0.13]) | 4.41 | 4.30 | 12.654 | +2.897 |  |
| SEQ:random(hqq-3b [16:0.05,8:0.13]) | 4.41 | 4.30 | 24.723 | +14.965 |  |
| SEQ:act_max(hqq-3b [16:0.02,8:0.22]) | 4.46 | 4.36 | 12.301 | +2.544 |  |
| SEQ:random(hqq-3b [16:0.02,8:0.22]) | 4.46 | 4.36 | 23.339 | +13.581 |  |
| SEQ:act_max(hqq-4b k=0.05) | 4.70 | 4.60 | 10.424 | +0.667 |  |
| SEQ:act_scale(hqq-4b k=0.05) | 4.70 | 4.60 | 10.571 | +0.813 |  |
| SEQ:random(hqq-4b k=0.05) | 4.70 | 4.60 | 11.132 | +1.375 |  |
| SEQ:act_max(hqq-4b k=0.1) | 5.30 | 5.20 | 10.332 | +0.574 | ★ |
| SEQ:act_scale(hqq-4b k=0.1) | 5.30 | 5.20 | 10.504 | +0.746 |  |
| SEQ:random(hqq-4b k=0.1) | 5.30 | 5.20 | 11.089 | +1.332 |  |
| FP16 | 16.00 | 16.00 | 9.757 | +0.000 | ★ |

> SEQ is on the Pareto frontier (1 point(s)).

## meta-llama/Llama-3.2-3B

| method | actual bits | nominal bits | PPL | Δ vs FP16 | frontier |
|---|---|---|---|---|---|
| SEQ:act_scale(hqq-4b k=0.0) | 4.00 | 4.00 | 8.387 | — | ★ |
| SEQ:act_max(hqq-4b k=0.0) | 4.00 | 4.00 | 8.387 | — | ★ |
| SEQ:random(hqq-4b k=0.0) | 4.00 | 4.00 | 8.387 | — | ★ |
| SEQ:act_max(hqq-4b k=0.01) | 4.22 | 4.12 | 8.193 | — | ★ |
| SEQ:act_scale(hqq-4b k=0.01) | 4.22 | 4.12 | 8.221 | — |  |
| SEQ:random(hqq-4b k=0.01) | 4.22 | 4.12 | 8.380 | — |  |
| SEQ:act_max(hqq-4b k=0.02) | 4.34 | 4.24 | 8.160 | — | ★ |
| SEQ:act_scale(hqq-4b k=0.02) | 4.34 | 4.24 | 8.205 | — |  |
| SEQ:random(hqq-4b k=0.02) | 4.34 | 4.24 | 8.375 | — |  |
| SEQ:act_max(hqq-3b [8:0.26]) | 4.40 | 4.30 | 9.038 | — |  |
| SEQ:random(hqq-3b [8:0.26]) | 4.40 | 4.30 | 12.034 | — |  |
| SEQ:act_max(hqq-3b [16:0.05,8:0.13]) | 4.40 | 4.30 | 9.100 | — |  |
| SEQ:random(hqq-3b [16:0.05,8:0.13]) | 4.40 | 4.30 | 12.471 | — |  |
| SEQ:act_max(hqq-3b [16:0.1]) | 4.40 | 4.30 | 9.301 | — |  |
| SEQ:random(hqq-3b [16:0.1]) | 4.40 | 4.30 | 12.941 | — |  |
| SEQ:act_max(hqq-3b [16:0.02,8:0.22]) | 4.46 | 4.36 | 8.962 | — |  |
| SEQ:random(hqq-3b [16:0.02,8:0.22]) | 4.46 | 4.36 | 12.157 | — |  |
| SEQ:act_max(hqq-4b k=0.05) | 4.70 | 4.60 | 8.125 | — | ★ |
| SEQ:act_scale(hqq-4b k=0.05) | 4.70 | 4.60 | 8.171 | — |  |
| SEQ:random(hqq-4b k=0.05) | 4.70 | 4.60 | 8.358 | — |  |
| SEQ:act_max(hqq-4b k=0.1) | 5.30 | 5.20 | 8.091 | — | ★ |
| SEQ:act_scale(hqq-4b k=0.1) | 5.30 | 5.20 | 8.146 | — |  |
| SEQ:random(hqq-4b k=0.1) | 5.30 | 5.20 | 8.329 | — |  |

> SEQ is on the Pareto frontier (7 point(s)).
