# SEQ-v4

Method name: `seq_v4`

Same as SEQ-v3, plus MLP `gate_proj` and `down_proj` modules are upgraded to at least INT8 when the base policy assigns INT4.

Override reason: `protect_gate_down_int8`
