# Reproducibility

## Environment

Use WSL at `/mnt/d/Abrar/SEQ/seq_v4`:

```bash
cd /mnt/d/Abrar/SEQ/seq_v4
source .venv-seq/bin/activate
python scripts/audit_final_environment.py --output results/wsl_environment_audit.json
python -m compileall seq_core analysis scripts
python -m pytest -q
```

Observed environment: Python 3.12.13, PyTorch 2.13.0, CUDA 13.0, transformers 5.13.1, datasets 5.0.0, accelerate 1.14.0, HQQ 0.2.8.post1, RTX 5090 31.84 GiB. Hugging Face cache is under `~/.cache/huggingface` and contains Llama-3.2-1B/3B and WikiText-2.

## Commands

Use `scripts/run_final_seq_pipeline.sh --dry-run` to inspect phase commands. Uniform HQQ runs use `scripts/run_uniform_hqq_sweep.sh`; LLMC baselines use `scripts/run_final_baselines.sh` after passing `--llmc-repo` and `--llmc-venv`. Every completed sweep writes `channel_pareto.json` and Markdown beneath its output directory.

## Missing components

LightCompress is installed at `/mnt/d/LightCompress` with Python 3.11 environment `/mnt/d/LightCompress/.venv-llmc`; its commit and every command are recorded in the LLMC summaries. `lm_eval==0.4.12` is installed in `.venv-seq`; the saved SEQ checkpoint has a bounded 10-example smoke result, while full downstream evaluation remains pending. The GPTQ fake-quant path must be validated with `scripts/validate_gptq_llmc_base.py` before any GPTQ-SEQ row is admissible.

## Determinism and resources

Default seed is 1234. Canonical PPL uses the full WikiText-2 token stream in non-overlapping 2048-token chunks. A 1B scalar HQQ sweep uses roughly nine minutes; 3B residual/greedy sweeps take 13-16 minutes on the observed GPU. LLMC AWQ/GPTQ 3B runs can take 10-16 minutes. Allow extra disk for model cache and fake-quant checkpoints. Resume markers are stored under the master output root.
