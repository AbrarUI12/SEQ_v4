# Implementation log

## 2026-07-17

| file | old behavior | new behavior | validation | limitation |
|---|---|---|---|---|
| `scripts/audit_final_environment.py` | no consolidated audit | reports Python/packages/CUDA/GPU/paths/cache/Git/LLMC | executed in WSL | model access is inferred from cache, not downloaded |
| `seq_core/storage_accounting.py` | several nominal formulas | one byte-level accounting API | 3 unit tests | serialized checkpoint size remains unavailable until save support exists |
| `seq_core/channel_sweep.py` | nominal `effective_bits` only | emits storage breakdown and `actual_effective_bits` | compile/tests; HQQ k=0 run | corrections are runtime fake-quant FP16 tensors |
| `analysis/validate_*.py` | manual inspection | recursive JSON validation | compile/smoke | expected-count policy still needs final matrix manifest |
| `analysis/build_comparison.py` | Markdown only | UTF-8 Markdown plus CSV/JSON | regenerated comparison | old runs lack authoritative actual bits |
| shell runners | ad-hoc commands | baseline, HQQ, downstream, and master wrappers | `bash -n` pending | LLMC/lm-eval unavailable |
| script-style tests | `sys.exit()` broke pytest collection | exit only when run directly | 8 pytest tests pass | legacy checks still execute at import time |

Environment audit: core dependencies passed in WSL; CUDA/HQQ/datasets are available. `bitsandbytes` and `lm_eval` remain absent. LightCompress was cloned to `/mnt/d/LightCompress` and its Python 3.11 environment was installed; RTN/GPTQ/AWQ baselines completed for 1B and 3B.
