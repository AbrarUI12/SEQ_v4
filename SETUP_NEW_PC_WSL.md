# SEQ New PC WSL Setup

## Audit-based situation

- WSL2 is installed and the active distro is Ubuntu 26.04.
- The SEQ Windows path is `D:\Abrar\SEQ\SEQ-clean-v3-main (1)\SEQ-clean-v3-main`.
- The SEQ WSL path is `/mnt/d/Abrar/SEQ/SEQ-clean-v3-main (1)/SEQ-clean-v3-main`.
- The stable working symlink is `~/SEQ_Clean`.
- `E:\` is absent, so old `/mnt/e/SEQ_Clean` and `/mnt/e/LightCompress` paths are stale.
- LightCompress was not found under `/mnt/d` or `/home/abrar`.

## Environment

- WSL distro: Ubuntu 26.04 LTS.
- System Python: Python 3.14.4.
- SEQ venv Python: Python 3.12.13 installed user-locally with `uv`.
- Venv target: `/home/abrar/.venvs/SEQ_Clean/.venv-seq`.
- Repo venv symlink: `~/SEQ_Clean/.venv-seq`.
- Activation command:

```bash
cd ~/SEQ_Clean
source .venv-seq/bin/activate
```

Python 3.14 could not resolve the pinned SEQ lock because `pandas==3.0.2` requires `numpy>=2.3.3` on Python 3.14 while the SEQ lock pins `numpy==2.2.6`. The active environment therefore uses Python 3.12.

## Installed dependency files

- `requirements.clean.lock.txt`
- `requirements.lm_eval.txt`
- `requirements.wsl_compare_extras.txt`

PyTorch was installed first with:

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

Then the rest of `requirements.clean.lock.txt` was installed without the torch line, followed by lm-eval and WSL compare extras.

## CUDA/GPU result

- GPU visible in WSL: NVIDIA GeForce RTX 5090.
- VRAM reported by torch: 31.84 GiB.
- Torch: `2.10.0+cu128`.
- Torch CUDA build: `12.8`.
- `torch.cuda.is_available()` returned `True`.

## bitsandbytes result

`bitsandbytes==0.49.1` imported and executed `bnb.nn.Linear8bitLt` on CUDA successfully.

## Hugging Face

The `hf` and `huggingface-cli` commands are installed inside `.venv-seq`, but the environment is not logged in.

Run interactively when needed:

```bash
cd ~/SEQ_Clean
source .venv-seq/bin/activate
hf auth login
```

Do not print tokens. The target model `meta-llama/Llama-3.1-8B` is gated and may require approved Hugging Face access.

## Output directories

Writing generated results directly under `/mnt/d` originally caused a `PermissionError` during `shutil.copy(... copymode ...)`. The SEQ report copy now uses a content-only copy, so generated results can live under the Windows/VS Code workspace.

- `results/`
- `results_smoke/`
- `results_smoke_lmeval/`

The earlier failed DrvFS smoke output was preserved as a `results_smoke.drvfs-broken-*` directory. Helper scripts now write repo-local paths so new run summaries are visible directly in VS Code.

## Verify command

```bash
cd ~/SEQ_Clean
bash verify_seq_wsl_env.sh
```

## Smoke command

```bash
cd ~/SEQ_Clean
bash run_seq_smoke.sh
```

Last observed smoke result: `facebook/opt-125m` base and SEQ PPL both succeeded.

## lm-eval smoke command

```bash
cd ~/SEQ_Clean
bash run_seq_lmeval_smoke.sh
```

Last observed lm-eval result: `facebook/opt-125m` HellaSwag completed with `lm_eval__status=ok`.

## LightCompress status

LightCompress repo location/setup not found; user must provide the repo folder or clone URL.

When provided, use an explicit path, for example:

```bash
--llmc_repo "$HOME/LightCompress"
--llmc_venv "$HOME/LightCompress/.venv-llmc"
```

Do not rely on `/mnt/e/LightCompress` on this PC.

## Final safe next command

```bash
cd ~/SEQ_Clean
bash verify_seq_wsl_env.sh
```
