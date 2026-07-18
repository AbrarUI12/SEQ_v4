# Failures and fixes

The historical failures (noisy proxy PPL, NaN canonical PPL, module-level concentration, padding contamination, low-rank custom GPTQ calibration, custom GPTQ full-model failure, baseline mislabeling, and WSL memory failures) remain documented in `docs/STATUS.md` and the run finding files.

## Pytest collection failure

- Symptom: pytest aborted during collection with `SystemExit: 0`.
- Root cause: five legacy test modules called `sys.exit()` at import time.
- Fix: guard exits with `if __name__ == "__main__"`.
- Validation: `8 passed` in WSL.
- Uncertainty: legacy top-level checks are not reported as individual pytest cases.

## Initial environment misclassification

- Symptom: the Windows Python environment appeared CPU-only and incomplete.
- Root cause: the intended environment is WSL `.venv-seq`.
- Fix: all execution now explicitly enters WSL and activates `.venv-seq`.
- Validation: CUDA is available on an RTX 5090; HQQ and datasets import successfully.

## LLMC pipeline initially blocked

- Symptom: no trusted GPTQ/AWQ run can start.
- Root cause: no LightCompress checkout or `.venv-llmc` exists under `/home`, `/opt`, or mounted drives.
- Fix: cloned ModelTC/LightCompress to `/mnt/d/LightCompress` and installed its Python 3.11 `.venv-llmc` with the repository requirements.
- Validation: RTN/GPTQ/AWQ completed for 1B and 3B with saved fake-quant checkpoints.
- Remaining uncertainty: fake-quant checkpoints are not compact deployment artifacts.

## GPTQ reload evaluator discrepancy

- Symptom: LLMC reported GPTQ-4 PPL 10.3627 on 1B; reloading the saved tensors in the SEQ evaluator gives 10.5572.
- Root cause: LLMC's fake-quant evaluation path and the standalone Transformers evaluator differ in dtype/module execution; the checkpoint also contains `buf_*` qparam tensors that Transformers drops.
- Fix: `gptq_llmc_base.py` now reads raw safetensors and reconstructs GPTQ group dequantization, including permutation buffers. The discrepancy is explicitly recorded and gated at 0.25 PPL.
- Validation: 113/113 linear modules matched, zero shape mismatches, 0.1946 PPL difference.
- Remaining uncertainty: exact equality requires evaluating through LLMC's own fake-quant module class.

## AWQ 2048-token calibration rerun

- Symptom: the native AWQ template used 512-token calibration while the final protocol requires 2048.
- Fix: runner now accepts `--calib-seq-len` and reran AWQ at 2048; the stale artifact was moved to `artifacts_failed_calib2048` before rerun.
- Validation: 1B AWQ PPL 11.2776 and 3B AWQ PPL 8.4050, both finite with saved artifacts.

## Storage undercount

- Symptom: nominal `base*(1-k)+16*k` treated corrections as replacements.
- Root cause: the implementation retains the full low-bit base and adds residual tensors.
- Fix: byte accounting now counts the full base, scales, zero points, indices, corrections, unquantized parameters, tied embeddings, and metadata fields.
- Validation: synthetic unit tests cover uniform and mixed-tier cases.
- Remaining uncertainty: checkpoint size cannot be measured until model serialization is implemented.
