# SEQ-v5

Method name: `seq_v5`

Starts from SEQ-v1 activation masking and replaces the hard AND/XOR tier policy with risk-score assignment:

- `risk_score = max(weight_rank, act_rank)`
- FP16 when `risk_score >= 0.90`
- INT8 when `risk_score >= 0.65`
- INT4 otherwise

Only the original SEQ protection floors are enabled by default.
