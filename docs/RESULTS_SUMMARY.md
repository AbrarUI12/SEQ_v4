# Results summary

## 2026-07-18 accounting and greedy-rerun update

The primary comparison axis is now weight-only average bits per quantized
linear weight, including quantizer metadata; FP16 embeddings, `lm_head`, and
other unquantized parameters remain only in the separate full-model reference
column. The CPU-regenerated comparison contains 74 points for the 1B/3B runs,
with old greedy rows excluded because they predate the selection-order fix.
GPTQ-4 metadata-aware weight bits are 4.293 (1B) and 4.284 (3B), rather than
the old inflated full-model values.

The corrected greedy runner is ready at
`scripts/run_fixed_greedy_sweeps.sh`, but its preflight stops because no
Llama-3.1-8B GPTQ fake-quant checkpoint exists in the repository. Therefore no
corrected GPU greedy sweep is represented as complete; see
`docs/GREEDY_REGEN_BLOCKED.md` and `runs/final_greedy_fixed/run.log`.

Only finite, saved WSL runs are listed. Selector runs used 8 calibration prompts, so they are validated development evidence, not final robustness evidence.

## Completed evidence

For Llama-3.2-1B, FP16 PPL is 9.7572 and uniform HQQ PPL is 11.1870/10.0636/9.8294/9.7618 at nominal 4/5/6/8 bits. At HQQ-4, `act_max` gives 10.3247 at 10% protection (5.201 nominal bits), compared with residual-rms 10.5000, residual-max 10.3373, greedy 10.4405, and random 11.0891. At 20%, greedy reaches 10.2054, `act_max` 10.2381, and random 10.9931.

For Llama-3.2-3B, FP16 PPL is 7.8167 and uniform HQQ is 8.3870/7.9565/7.8448/7.8197 at nominal 4/5/6/8 bits. The complete HQQ-4 residual and greedy sweeps are saved and finite.

Trusted LLMC fake-quant baselines are present for both models. On 1B, RTN-4/GPTQ-4/AWQ-4 PPL are 11.7099/10.3627/11.2776; on 3B they are 8.4982/8.3041/8.4050. The LLMC GPTQ checkpoint reloads at 10.5572 under the SEQ evaluator (0.1946 PPL from LLMC's 10.3627, within the documented 0.25 gate).

On the LLMC GPTQ base, 1B residual-max reaches 10.4128 at 5.20 nominal bits (9.50 estimated full-model bits), versus loaded-base 10.5572; 3B residual-max reaches 8.0998 at 6.79 estimated bits, versus 8.1715 loaded-base PPL. Random controls are worse on 3B. No downstream-task evaluation has been run.

## Current best result

The best HQQ-SEQ PPL below nominal 7 bits is greedy 20% at 10.2054 PPL and 6.402 nominal bits (1B). The best trusted-GPTQ point below 7 estimated full-model bits is 3B residual-max at 8.0998 PPL and 6.785 bits. Neither matches FP16.

## Negative results

Residual-aware scalar scores did not beat `act_max` in the HQQ 1B sweep. Greedy was worse at 2/5/10% protection but slightly better at 20%. On the GPTQ base, act_max is worse than residual-max and greedy is catastrophically unstable at 10% on 1B (PPL 77.99); this is retained as a negative result.

## Pending

`lm_eval==0.4.12` is installed. A 10-example-per-task smoke run has been completed for the saved SEQ checkpoint (HellaSwag 0.30, ARC-Easy 0.60, ARC-Challenge 0.30, PIQA 0.90, Winogrande 0.70, LAMBADA 0.50; diagnostic only). Reloaded WikiText PPL is 10.41271 versus 10.41285 before save (absolute difference 1.4e-4, pass at 1e-3). Full task-suite runs remain pending. The fake-quant LLMC checkpoints serialize BF16-style weights (~16 bits/parameter); compact deployment-size claims must not use those files as 4-bit checkpoints. Scenario B remains gated on full downstream results, exact matched-storage baselines, and compact deployment serialization.
