# Paper-writing bundle — the 15 things that matter

Everything needed to finish, compile, verify and submit the paper. Directories and file groups
count as one entry, so the 15 entries below expand to 45 files (~800 KB).

## A. Paper source — compile and edit (5)

| # | entry | why it matters |
|---|---|---|
| 1 | `paper/main.tex` | The paper. ACL format, anonymous, compiles clean (0 undefined refs/citations). |
| 2 | `paper/refs.bib` | 17 bibliography entries. |
| 3 | `paper/figures/` | The 4 cited figures. `fig0_decoupling` is the headline. |
| 4 | `paper/Makefile` | `make` builds; `make check` fails on undefined refs. |
| 5 | `paper/README.md` | How to switch to the official ACL style; **anonymity rules**. |

`paper/main.pdf` is included as the compiled artifact for reading.

## B. Prose and rendered tables (3)

| # | entry | why it matters |
|---|---|---|
| 6 | `docs/FINDINGS_PAPER.md` | The working document. Full prose and every number; `main.tex` is its typeset form. Edit prose here first. |
| 7 | `docs/DOWNSTREAM.md` | Rendered downstream table for all operating points. |
| 8 | `docs/COMPARISON.md` | Matched-storage comparison / Pareto table (§7). |

## C. The artifacts every number traces to (6)

| # | entry | supports |
|---|---|---|
| 9 | `results/downstream.json`, `.csv` | §4.1 accuracies **and** the paired-bootstrap CIs. The authoritative source — macro values here match the CIs exactly (67.61 − 67.10 = +0.51). |
| 10 | `results/pertoken_Llama-3.2-3B.json` | §4.3: 74.6% tokens worse, median ΔNLL +0.287, concentration and trimmed-perplexity tables. |
| 11 | `results/align_Llama-3.2-{3B,1B}.json` | §6, the falsified mechanism: ρ +0.988 for safe `residual_rms` vs +0.390 for catastrophic `greedy_gain`. |
| 12 | `results/e5_matchedcalib_{3B,1B}.json` | §5.3: selector Hessian matched to the base (3B 58.24, 1B 11.39). Check `selector_hessian_tokens = 262144`. |
| 13 | `results/sweeps/` | §5.1 selector panel and §7 audits — every perplexity in the paper's main tables, for both bases and both models. Each carries its own `skip_lm_head` / `base_group_size` / `seed` provenance. |
| 14 | `results/reload_*.log` | §4.2 checkpoint identity: 3B 51.76 `PASS`, 1B 63.83 `PASS`. This is what makes the headline claim defensible — without it the decoupling is unverifiable. |

## D. Reproducibility (1)

| # | entry | why it matters |
|---|---|---|
| 15 | `repro/run_manifest.json`, `requirements.txt` | Git commit, hardware, package versions, resolved HF revision of every model; pinned environment. Required for the ARR reproducibility checklist. |

---

## Deliberately excluded

- **Model weights and fake-quant checkpoints** — gigabytes, and never committed.
- **Raw lm-eval sample logs** (`runs/final/downstream/**/lm_eval/`) — hundreds of MB of
  per-example JSONL. The aggregated `downstream.json` carries the numbers and the CIs.
- **Code** (`seq_core/`, `scripts/`, `analysis/`) — needed to *re-run* experiments, not to write
  the paper. Clone the repo for that.
- **Excluded experiment artifacts** (`results/owq_seq_*`) — the ordering intervention ran on a
  degenerate base (3437 PPL) and is reported as excluded in §9; kept in the repo for audit only.

## Before submitting

1. Drop `acl.sty` + `acl_natbib.bst` into `paper/` and rebuild — **then check the page count**
   against ACL's 8-page main-content limit. The 12 pages of the current PDF are the
   single-column fallback layout and are *not* the real count.
2. Keep it anonymous: no author block, no link to an authored repository.
