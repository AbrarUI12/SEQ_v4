# Experiment log

## Completed on 2026-07-17 (WSL, RTX 5090)

All runs used canonical WikiText-2 test PPL at sequence length 2048. Selector calibration used the first 8 configured prompts; this is a development run and must not be presented as the final 128-sample robustness protocol.

| run | output | validation |
|---|---|---|
| Llama-3.2-1B HQQ-4 k=0 | `runs/final_hqq_uniform/Llama-3.2-1B/b4` | finite PPL, 113 layers, zero errors |
| Llama-3.2-1B HQQ-5/6/8 k=0 | `runs/final_hqq_uniform/Llama-3.2-1B/b{5,6,8}` | finite PPL, zero errors |
| Llama-3.2-1B HQQ-4 scalar selectors | `runs/final_hqq_residual/Llama-3.2-1B` | 20 configurations, finite PPL, zero errors |
| Llama-3.2-1B HQQ-4 greedy | `runs/final_hqq_greedy/Llama-3.2-1B` | 5 configurations, finite PPL, zero errors |
| Llama-3.2-3B HQQ-4/5/6/8 k=0 | `runs/final_hqq_uniform/Llama-3.2-3B/b{4,5,6,8}` | finite PPL, zero errors |
| Llama-3.2-3B HQQ-4 scalar selectors | `runs/final_hqq_residual/Llama-3.2-3B` | 20 configurations, finite PPL, zero errors |
| Llama-3.2-3B HQQ-4 greedy | `runs/final_hqq_greedy/Llama-3.2-3B` | 5 configurations, finite PPL, zero errors |
| Llama-3.2-1B HQQ value-tier | `runs/final_hqq_value/Llama-3.2-1B` | 4 budgets, finite PPL, zero errors |
| LLMC RTN/GPTQ/AWQ 1B | `runs/final_llmc/Llama-3.2-1B` | all completed; fake-quant artifacts saved |
| LLMC RTN/GPTQ/AWQ 3B | `runs/final_llmc/Llama-3.2-3B` | all completed; fake-quant artifacts saved |
| GPTQ-base 1B scalar selectors | `runs/final_gptqbase/Llama-3.2-1B` | 16 configurations, finite PPL, zero errors |
| GPTQ-base 1B greedy | `runs/final_gptqgreedy/Llama-3.2-1B` | 4 configurations, finite PPL, zero errors |
| GPTQ-base 3B residual-max/random | `runs/final_gptqbase/Llama-3.2-3B` | 8 configurations, finite PPL, zero errors |

## Pending

- full lm-eval downstream tasks (a bounded 10-example-per-task smoke run is complete; metrics are diagnostic).
- compact serialized deployment-size measurements; current LLMC and SEQ exports are dense fake-quant evaluation artifacts.
