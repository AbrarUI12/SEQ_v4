# SEQ-v3

Method name: `seq_v3`

Same as SEQ-v2, plus all attention projections are upgraded to at least INT8 when the base policy assigns INT4.

Override reason: `protect_all_attention_proj_int8`
