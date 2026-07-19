# Reproducibility

## Environment

Run the final workflow inside WSL from the repository checkout. The checkout
may contain spaces; the external LightCompress checkout and venv must not.

```bash
cd "/path/to/SEQ-clean-v4"
bash scripts/bootstrap_final_environment.sh --llmc-repo /mnt/e/LightCompress
source .venv-seq/bin/activate
export HF_TOKEN="$(cat /path/to/private/token)"  # never written to run reports
```

The bootstrap pins LightCompress to
`86f564ddb1d6548b228c67a10509a4ed7264345c`, creates its Python 3.11 venv at
`/mnt/e/LightCompress/.venv-llmc`, and installs the missing SEQ test/HQQ
dependencies. The final preflight records both dependency freezes, Git state,
GPU details, matrix digest, and resolved Hugging Face model revisions.

## Inspect and run

Dry-run the complete gate-first matrix from any working directory:

```bash
bash "/path/to/SEQ-clean-v4/scripts/run_final_seq_pipeline.sh" \
  --llmc-repo /mnt/e/LightCompress \
  --llmc-venv /mnt/e/LightCompress/.venv-llmc \
  --output-root "/path/to/SEQ-clean-v4/runs/final" \
  --dry-run
```

Run or resume without publication first:

```bash
bash scripts/run_final_seq_pipeline.sh \
  --llmc-repo /mnt/e/LightCompress \
  --llmc-venv /mnt/e/LightCompress/.venv-llmc \
  --output-root runs/final --resume
```

The pipeline generates the GPTQ bases, runs the early gate, writes
`runs/final/reports/GATE_SUMMARY.md`, and then completes the remaining matrix.
Inspect staged outputs under `runs/final/staging`; publish only validated results:

```bash
bash scripts/run_final_seq_pipeline.sh \
  --llmc-repo /mnt/e/LightCompress \
  --llmc-venv /mnt/e/LightCompress/.venv-llmc \
  --output-root runs/final --resume --publish
```

Publication requires PASS reports for the gate and full manifest-derived matrix.
The nominal 152-cell expansion is diagnostic only; missing cells and random seed
replicates are validated by their explicit identities.

## Protocol

The authoritative matrix is `configs/final_comparison_matrix.json`. Canonical
PPL uses the full WikiText-2 token stream in non-overlapping 2048-token chunks.
Deterministic selectors use seed 1234; random uses seeds 1234, 2345, and 3456.
The comparison axis is actual weight-only bits reconstructed from each saved
storage breakdown.
