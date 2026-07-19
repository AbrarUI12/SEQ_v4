# Paper claims ledger

| claim | support | limitations | allowed wording | disallowed wording |
|---|---|---|---|---|
| Module entropy is unreliable | runs 1-6 | tested models/objectives only | Entropy was weak or misleading in our tested settings. | Entropy never works. |
| Activation-outlier correction helps HQQ | runs 4-6 and final HQQ sweeps | HQQ; selector calibration uses the configured prompt set | `act_max` improved HQQ-4 and beat random on tested models/configurations. | SEQ is state of the art. |
| Residual scalar selection beats `act_max` | final HQQ 1B/3B sweeps | result is negative on HQQ | Residual scores did not beat `act_max` in the tested HQQ sweeps. | Any superiority claim for residual scores on HQQ. |
| Greedy selection improves HQQ | final 1B/3B greedy sweeps | strongest only at selected fractions; no universal win | Greedy was competitive at selected HQQ operating points. | Greedy is consistently best. |
| SEQ improves trusted GPTQ | final GPTQ-base 1B/3B sweeps; GPTQ `act_scale` control | no full downstream tasks; 1B greedy instability; compact matched-storage deployment not measured | Residual-max protection improved selected loaded-GPTQ PPL points; at k=0.10 it beats the 1B random control and the 3B random/act-scale controls. | SEQ is consistently better than GPTQ at all matched storage and downstream tasks. |
| Actual storage is competitive | `seq_core/storage_accounting.py`, final sweep rows | LLMC fake-quant files serialize BF16 weights plus metadata; compact deployment checkpoint not measured | The pipeline reports full theoretical runtime storage and separately records serialized fake-quant size. | The fake-quant artifact is a compact 4-bit checkpoint. |

Current publication classification: Scenario A backbone with a conditional Scenario B result. GPTQ-base protection improves selected PPL points on 1B/3B, but Scenario B is not admissible until the full downstream suite and compact matched-storage baselines are complete.
