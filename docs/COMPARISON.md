# SEQ vs baselines — actual-bits comparison

Points from SEQ sweeps + external baselines, sorted by actual bits. ★ = on the Pareto frontier (no method has both fewer bits and lower PPL). SEQ bits include the FP16 residual + index table; base group scales are common to all methods and excluded from this axis.

## meta-llama/Llama-3.2-1B  (FP16 PPL 9.7572)

| method | actual bits | nominal bits | PPL | Δ vs FP16 | frontier |
|---|---|---|---|---|---|
| GPTQ-4 g128 | 4.00 | 4.00 | 10.363 | +0.606 | ★ |
| SEQ:greedy(hqq-4b k=0.0) | 4.00 | 4.00 | 11.187 | +1.430 |  |
| SEQ:act_max(hqq-4b k=0.0) | 4.00 | 4.00 | 11.187 | +1.430 |  |
| SEQ:residual_rms(hqq-4b k=0.0) | 4.00 | 4.00 | 11.187 | +1.430 |  |
| SEQ:residual_max(hqq-4b k=0.0) | 4.00 | 4.00 | 11.187 | +1.430 |  |
| SEQ:random(hqq-4b k=0.0) | 4.00 | 4.00 | 11.187 | +1.430 |  |
| AWQ-4 g128 | 4.00 | 4.00 | 11.278 | +1.520 |  |
| RTN-4 g128 | 4.00 | 4.00 | 11.710 | +1.953 |  |
| SEQ:act_max(hqq-4b k=0.02) | 4.34 | 4.24 | 10.516 | +0.759 |  |
| SEQ:residual_max(hqq-4b k=0.02) | 4.34 | 4.24 | 10.523 | +0.766 |  |
| SEQ:residual_rms(hqq-4b k=0.02) | 4.34 | 4.24 | 10.615 | +0.858 |  |
| SEQ:greedy(hqq-4b k=0.02) | 4.34 | 4.24 | 11.113 | +1.356 |  |
| SEQ:random(hqq-4b k=0.02) | 4.34 | 4.24 | 11.160 | +1.403 |  |
| SEQ:act_max(hqq-4b k=0.0) | 4.50 | 4.00 | 11.187 | +1.430 |  |
| HQQ-4 uniform | 4.50 | 4.00 | 11.187 | +1.430 |  |
| SEQ:residual_max(hqq-4b k=0.05) | 4.70 | 4.60 | 10.422 | +0.665 |  |
| SEQ:act_max(hqq-4b k=0.05) | 4.70 | 4.60 | 10.422 | +0.665 |  |
| SEQ:residual_rms(hqq-4b k=0.05) | 4.70 | 4.60 | 10.568 | +0.810 |  |
| SEQ:greedy(hqq-4b k=0.05) | 4.70 | 4.60 | 10.602 | +0.845 |  |
| SEQ:random(hqq-4b k=0.05) | 4.70 | 4.60 | 11.132 | +1.375 |  |
| SEQ:act_max(hqq-5b k=0.0) | 5.00 | 5.00 | 10.064 | +0.306 | ★ |
| HQQ-5 uniform | 5.00 | 5.00 | 10.064 | +0.306 | ★ |
| SEQ:act_max(hqq-4b k=0.1) | 5.30 | 5.20 | 10.325 | +0.568 |  |
| SEQ:residual_max(hqq-4b k=0.1) | 5.30 | 5.20 | 10.337 | +0.580 |  |
| SEQ:greedy(hqq-4b k=0.1) | 5.30 | 5.20 | 10.441 | +0.683 |  |
| SEQ:residual_rms(hqq-4b k=0.1) | 5.30 | 5.20 | 10.500 | +0.743 |  |
| SEQ:random(hqq-4b k=0.1) | 5.30 | 5.20 | 11.089 | +1.332 |  |
| SEQ:act_max(hqq-6b k=0.0) | 6.00 | 6.00 | 9.829 | +0.072 | ★ |
| HQQ-6 uniform | 6.00 | 6.00 | 9.829 | +0.072 | ★ |
| SEQ:greedy(hqq-4b k=0.2) | 6.50 | 6.40 | 10.205 | +0.448 |  |
| SEQ:act_max(hqq-4b k=0.2) | 6.50 | 6.40 | 10.238 | +0.481 |  |
| SEQ:residual_max(hqq-4b k=0.2) | 6.50 | 6.40 | 10.243 | +0.486 |  |
| SEQ:residual_rms(hqq-4b k=0.2) | 6.50 | 6.40 | 10.395 | +0.637 |  |
| SEQ:random(hqq-4b k=0.2) | 6.50 | 6.40 | 10.993 | +1.236 |  |
| SEQ:act_scale(gptq_llmc-4b k=0.0) | 7.90 | 4.00 | 10.557 | +0.800 |  |
| SEQ:act_max(gptq_llmc-4b k=0.0) | 7.90 | 4.00 | 10.557 | +0.800 |  |
| SEQ:residual_rms(gptq_llmc-4b k=0.0) | 7.90 | 4.00 | 10.557 | +0.800 |  |
| SEQ:residual_max(gptq_llmc-4b k=0.0) | 7.90 | 4.00 | 10.557 | +0.800 |  |
| SEQ:random(gptq_llmc-4b k=0.0) | 7.90 | 4.00 | 10.557 | +0.800 |  |
| SEQ:greedy(gptq_llmc-4b k=0.0) | 7.90 | 4.00 | 10.557 | +0.800 |  |
| SEQ:act_max(hqq-8b k=0.0) | 8.00 | 8.00 | 9.762 | +0.005 | ★ |
| HQQ-8 uniform | 8.00 | 8.00 | 9.762 | +0.005 | ★ |
| SEQ:greedy(gptq_llmc-4b k=0.02) | 8.15 | 4.19 | 10.770 | +1.013 |  |
| SEQ:tier_alloc(hqq-4b [budget=0.25]) | 8.16 | 4.07 | 10.631 | +0.873 |  |
| SEQ:random(gptq_llmc-4b k=0.02) | 8.22 | 4.24 | 10.412 | +0.654 |  |
| SEQ:residual_max(gptq_llmc-4b k=0.02) | 8.22 | 4.24 | 10.423 | +0.666 |  |
| SEQ:residual_rms(gptq_llmc-4b k=0.02) | 8.22 | 4.24 | 10.434 | +0.677 |  |
| SEQ:act_max(gptq_llmc-4b k=0.02) | 8.22 | 4.24 | 10.711 | +0.954 |  |
| SEQ:act_scale(gptq_llmc-4b k=0.02) | 8.22 | 4.24 | 10.930 | +1.172 |  |
| SEQ:tier_alloc(hqq-4b [budget=0.5]) | 8.41 | 4.13 | 10.592 | +0.835 |  |
| SEQ:greedy(gptq_llmc-4b k=0.05) | 8.53 | 4.47 | 10.748 | +0.991 |  |
| SEQ:random(gptq_llmc-4b k=0.05) | 8.71 | 4.60 | 10.413 | +0.656 |  |
| SEQ:residual_max(gptq_llmc-4b k=0.05) | 8.71 | 4.60 | 10.436 | +0.678 |  |
| SEQ:residual_rms(gptq_llmc-4b k=0.05) | 8.71 | 4.60 | 10.436 | +0.679 |  |
| SEQ:act_scale(gptq_llmc-4b k=0.05) | 8.71 | 4.60 | 10.476 | +0.719 |  |
| SEQ:act_max(gptq_llmc-4b k=0.05) | 8.71 | 4.60 | 10.687 | +0.930 |  |
| SEQ:tier_alloc(hqq-4b [budget=1.0]) | 8.93 | 4.27 | 10.555 | +0.798 |  |
| SEQ:greedy(gptq_llmc-4b k=0.1) | 9.16 | 4.95 | 77.987 | +68.230 |  |
| SEQ:residual_max(gptq_llmc-4b k=0.1) | 9.50 | 5.20 | 10.413 | +0.656 |  |
| SEQ:random(gptq_llmc-4b k=0.1) | 9.50 | 5.20 | 10.432 | +0.675 |  |
| SEQ:residual_rms(gptq_llmc-4b k=0.1) | 9.50 | 5.20 | 10.441 | +0.684 |  |
| SEQ:act_scale(gptq_llmc-4b k=0.1) | 9.50 | 5.20 | 10.484 | +0.727 |  |
| SEQ:act_max(gptq_llmc-4b k=0.1) | 9.50 | 5.20 | 10.646 | +0.888 |  |
| SEQ:tier_alloc(hqq-4b [budget=2.0]) | 9.96 | 4.54 | 10.479 | +0.721 |  |
| FP16 | 16.00 | 16.00 | 9.757 | +0.000 | ★ |

> SEQ is on the Pareto frontier (3 point(s)). At ~8.0 bits, best SEQ (9.762) **loses to** HQQ-8 uniform (9.762).

## meta-llama/Llama-3.2-3B  (FP16 PPL 7.8167)

| method | actual bits | nominal bits | PPL | Δ vs FP16 | frontier |
|---|---|---|---|---|---|
| GPTQ-4 g128 | 4.00 | 4.00 | 8.304 | +0.487 | ★ |
| SEQ:act_max(hqq-4b k=0.0) | 4.00 | 4.00 | 8.387 | +0.570 |  |
| HQQ-4 uniform | 4.00 | 4.00 | 8.387 | +0.570 |  |
| AWQ-4 g128 | 4.00 | 4.00 | 8.405 | +0.588 |  |
| RTN-4 g128 | 4.00 | 4.00 | 8.498 | +0.682 |  |
| SEQ:act_scale(gptq_llmc-4b k=0.0) | 6.46 | 4.00 | 8.171 | +0.355 | ★ |
| SEQ:residual_max(gptq_llmc-4b k=0.0) | 6.46 | 4.00 | 8.171 | +0.355 | ★ |
| SEQ:random(gptq_llmc-4b k=0.0) | 6.46 | 4.00 | 8.171 | +0.355 | ★ |
| SEQ:greedy(hqq-4b k=0.0) | 6.46 | 4.00 | 8.387 | +0.570 |  |
| SEQ:act_max(hqq-4b k=0.0) | 6.46 | 4.00 | 8.387 | +0.570 |  |
| SEQ:residual_rms(hqq-4b k=0.0) | 6.46 | 4.00 | 8.387 | +0.570 |  |
| SEQ:residual_max(hqq-4b k=0.0) | 6.46 | 4.00 | 8.387 | +0.570 |  |
| SEQ:random(hqq-4b k=0.0) | 6.46 | 4.00 | 8.387 | +0.570 |  |
| SEQ:residual_max(gptq_llmc-4b k=0.02) | 6.79 | 4.24 | 8.100 | +0.283 | ★ |
| SEQ:act_max(hqq-4b k=0.02) | 6.79 | 4.24 | 8.160 | +0.344 |  |
| SEQ:residual_max(hqq-4b k=0.02) | 6.79 | 4.24 | 8.161 | +0.344 |  |
| SEQ:random(gptq_llmc-4b k=0.02) | 6.79 | 4.24 | 8.181 | +0.364 |  |
| SEQ:residual_rms(hqq-4b k=0.02) | 6.79 | 4.24 | 8.200 | +0.383 |  |
| SEQ:greedy(hqq-4b k=0.02) | 6.79 | 4.24 | 8.369 | +0.553 |  |
| SEQ:random(hqq-4b k=0.02) | 6.79 | 4.24 | 8.375 | +0.559 |  |
| SEQ:act_scale(gptq_llmc-4b k=0.02) | 6.79 | 4.24 | 8.586 | +0.769 |  |
| SEQ:residual_max(gptq_llmc-4b k=0.05) | 7.26 | 4.60 | 8.100 | +0.283 |  |
| SEQ:act_max(hqq-4b k=0.05) | 7.26 | 4.60 | 8.125 | +0.309 |  |
| SEQ:residual_max(hqq-4b k=0.05) | 7.26 | 4.60 | 8.126 | +0.310 |  |
| SEQ:residual_rms(hqq-4b k=0.05) | 7.26 | 4.60 | 8.168 | +0.351 |  |
| SEQ:greedy(hqq-4b k=0.05) | 7.26 | 4.60 | 8.201 | +0.384 |  |
| SEQ:random(gptq_llmc-4b k=0.05) | 7.26 | 4.60 | 8.220 | +0.404 |  |
| SEQ:random(hqq-4b k=0.05) | 7.26 | 4.60 | 8.358 | +0.541 |  |
| SEQ:act_max(hqq-5b k=0.0) | 7.46 | 5.00 | 7.957 | +0.140 | ★ |
| HQQ-5 uniform | 7.46 | 5.00 | 7.957 | +0.140 | ★ |
| SEQ:act_max(hqq-4b k=0.1) | 8.07 | 5.20 | 8.091 | +0.274 |  |
| SEQ:residual_max(hqq-4b k=0.1) | 8.07 | 5.20 | 8.095 | +0.278 |  |
| SEQ:residual_max(gptq_llmc-4b k=0.1) | 8.07 | 5.20 | 8.101 | +0.284 |  |
| SEQ:residual_rms(hqq-4b k=0.1) | 8.07 | 5.20 | 8.141 | +0.325 |  |
| SEQ:greedy(hqq-4b k=0.1) | 8.07 | 5.20 | 8.148 | +0.331 |  |
| SEQ:random(hqq-4b k=0.1) | 8.07 | 5.20 | 8.329 | +0.512 |  |
| SEQ:random(gptq_llmc-4b k=0.1) | 8.07 | 5.20 | 9.048 | +1.231 |  |
| SEQ:act_max(hqq-6b k=0.0) | 8.46 | 6.00 | 7.845 | +0.028 | ★ |
| HQQ-6 uniform | 8.46 | 6.00 | 7.845 | +0.028 | ★ |
| SEQ:greedy(hqq-4b k=0.2) | 9.67 | 6.40 | 8.028 | +0.211 |  |
| SEQ:act_max(hqq-4b k=0.2) | 9.67 | 6.40 | 8.047 | +0.230 |  |
| SEQ:residual_max(hqq-4b k=0.2) | 9.67 | 6.40 | 8.048 | +0.231 |  |
| SEQ:residual_rms(hqq-4b k=0.2) | 9.67 | 6.40 | 8.090 | +0.273 |  |
| SEQ:random(hqq-4b k=0.2) | 9.67 | 6.40 | 8.293 | +0.476 |  |
| SEQ:act_max(hqq-8b k=0.0) | 10.46 | 8.00 | 7.820 | +0.003 | ★ |
| HQQ-8 uniform | 10.46 | 8.00 | 7.820 | +0.003 | ★ |
| FP16 | 16.00 | 16.00 | 7.817 | +0.000 | ★ |

> SEQ is on the Pareto frontier (7 point(s)). At ~10.5 bits, best SEQ (7.820) **loses to** HQQ-8 uniform (7.820).
