# Final SEQ experiment plan

Status date: 2026-07-17.

## Question and hypotheses

The decisive question is whether sparse per-input-channel correction improves a trusted LLMC GPTQ-W4g128 base at matched actual storage. The primary hypothesis is that residual-aware or interaction-aware selection beats random and `act_scale`; the null is that base-quantizer quality dominates and protection adds no useful GPTQ Pareto point.

## Protocol

- Models: Llama-3.2-1B, then Llama-3.2-3B after 1B validation. Llama-3.1-8B is optional.
- Calibration: fixed prompts, 2048 tokens, deterministic seed. Final robustness uses 32/64/128 samples and three seeds.
- Evaluation: canonical WikiText-2 test perplexity, 2048-token non-overlapping chunks; downstream tasks use zero-shot HellaSwag, ARC easy/challenge, PIQA, WinoGrande, and LAMBADA.
- Bases: uniform HQQ at 4/5/6/8 bits and LLMC RTN/GPTQ/AWQ W4g128 fake-quant checkpoints.
- Selectors: random, `act_scale`, `act_max`, `residual_rms`, `residual_max`, greedy residual reduction, and value-tier allocation.
- Budgets: channel fractions 0/2/5/10/20% for HQQ and 0/2/5/10% for GPTQ; report nominal and actual bits.

## Decision and stopping rules

Scenario B requires GPTQ k=0 reproduction, improvement over GPTQ at matched actual storage, wins over random and `act_scale`, consistency on 1B/3B, and no material downstream regression. Otherwise report Scenario A. Stop a configuration on quantization failure, module mismatch, non-finite PPL, invalid baseline, or invalid storage accounting. Never fill missing cells with estimates.

## Compute

The WSL GPU is an RTX 5090 (31.84 GiB). The validated 1B HQQ residual sweep took about 8.7 minutes; the greedy sweep took about 4.3 minutes. LLMC phases require a separate LightCompress checkout/venv that is not currently present.
