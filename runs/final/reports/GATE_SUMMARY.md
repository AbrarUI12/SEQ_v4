# Early selector gate

Status: **PASS**
Framing: **audit**

| model | base | fraction | bits | greedy | greedy_indep | residual_max | random mean [95% CI] |
|---|---|---:|---:|---:|---:|---:|---:|
| meta-llama/Llama-3.2-1B | hqq | 0.02 | 4.820398 | 10.4954 | 10.5057 | 10.5334 | 11.1645 [11.1552, 11.1738] |
| meta-llama/Llama-3.2-1B | hqq | 0.05 | 5.304053 | 10.3814 | 10.4054 | 10.4283 | 11.1331 [11.1228, 11.1434] |
| meta-llama/Llama-3.2-1B | hqq | 0.10 | 6.101990 | 10.2989 | 10.3228 | 10.3369 | 11.0798 [11.0351, 11.1244] |
| meta-llama/Llama-3.2-1B | hqq | 0.20 | 7.703555 | 10.2067 | 10.2298 | 10.2320 | 10.9746 [10.9317, 11.0175] |
| meta-llama/Llama-3.2-1B | gptq_llmc | 0.02 | 4.820398 | 104.1635 | 15.6392 | 10.3907 | 10.6803 [10.4088, 10.9518] |
| meta-llama/Llama-3.2-1B | gptq_llmc | 0.05 | 5.304053 | 98.4098 | 14.8839 | 10.4004 | 10.6892 [10.2818, 11.0966] |
| meta-llama/Llama-3.2-1B | gptq_llmc | 0.10 | 6.101990 | 103.5316 | 14.8640 | 10.3909 | 10.6025 [10.1301, 11.0749] |
| meta-llama/Llama-3.2-1B | gptq_llmc | 0.20 | 7.703555 | 106.8236 | 15.6073 | 10.3495 | 10.6495 [10.2509, 11.0481] |
| meta-llama/Llama-3.2-3B | hqq | 0.02 | 4.822421 | 8.1493 | 8.1510 | 8.1606 | 8.3760 [8.3704, 8.3816] |
| meta-llama/Llama-3.2-3B | hqq | 0.05 | 5.301985 | 8.1120 | 8.1108 | 8.1263 | 8.3609 [8.3510, 8.3708] |
| meta-llama/Llama-3.2-3B | hqq | 0.10 | 6.103969 | 8.0732 | 8.0764 | 8.0946 | 8.3323 [8.3227, 8.3420] |
| meta-llama/Llama-3.2-3B | hqq | 0.20 | 7.703443 | 8.0278 | 8.0371 | 8.0483 | 8.2434 [8.0246, 8.4622] |
| meta-llama/Llama-3.2-3B | gptq_llmc | 0.02 | 4.822421 | 55.3448 | 44.8781 | 8.0985 | 8.1607 [8.1284, 8.1930] |
| meta-llama/Llama-3.2-3B | gptq_llmc | 0.05 | 5.301985 | 47.1931 | 45.5972 | 8.0980 | 8.1928 [8.0993, 8.2863] |
| meta-llama/Llama-3.2-3B | gptq_llmc | 0.10 | 6.103969 | 46.0269 | 46.7776 | 8.0936 | 8.4338 [7.3371, 9.5304] |
| meta-llama/Llama-3.2-3B | gptq_llmc | 0.20 | 7.703443 | 43.2086 | 45.4103 | 8.0697 | 8.3419 [7.9071, 8.7767] |

## Strata

- **PASS** meta-llama/Llama-3.2-1B / hqq: {"greedy_vs_greedy_indep": 4, "greedy_vs_random_ci": 4, "greedy_vs_residual_max": 4}
- **FAIL** meta-llama/Llama-3.2-1B / gptq_llmc: {"greedy_vs_greedy_indep": 0, "greedy_vs_random_ci": 0, "greedy_vs_residual_max": 0}
- **PASS** meta-llama/Llama-3.2-3B / hqq: {"greedy_vs_greedy_indep": 3, "greedy_vs_random_ci": 3, "greedy_vs_residual_max": 4}
- **FAIL** meta-llama/Llama-3.2-3B / gptq_llmc: {"greedy_vs_greedy_indep": 2, "greedy_vs_random_ci": 0, "greedy_vs_residual_max": 0}
