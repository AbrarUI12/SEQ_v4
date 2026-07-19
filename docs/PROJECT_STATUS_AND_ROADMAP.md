# SEQ — project status, results, and roadmap (both tracks)

_Last updated after the RTX-4090 final pipeline run (framing: **audit**)._

This is the single source of truth for **what is done, what is left, what to do
next, the exact commands, and the real numbers** — for both **Track A (findings /
audit paper)** and **Track B (improving SEQ as a method)**.

---

## 0. Repo / git / LFS state — READ FIRST

- Authoritative branch: `claude/seq-compression-quantization-yktihv`, remote HEAD
  `92681be` (force-rewritten history from an isolated results clone).
- **Your WSL working tree is on an old, unrelated history.** Sync it (your `runs/`
  files are gitignored/LFS and are preserved):
  ```bash
  cd "/mnt/e/seq v4/SEQ-clean-v4"
  git fetch origin
  git status --short          # note any tracked edits you want to keep; git stash if so
  git reset --hard origin/claude/seq-compression-quantization-yktihv
  git lfs pull                # materialize the result JSONs (they are LFS objects)
  ```
- **Known infrastructure bug (root cause of the broken comparison table):**
  `.gitattributes` LFS-tracks `runs/final/**/*.json`. In any fresh clone those
  become 129-byte **LFS pointer stubs**, and `build_comparison` used to silently
  skip them → it published a **baseline-only `COMPARISON.md` with zero SEQ rows.**
  - **Fixed in code:** `analysis/build_comparison.py` now detects LFS pointers,
    prints a hard ERROR, and (with `--require-sweep-points`) refuses to publish.
  - **Still to do on your machine (small JSONs should not be in LFS):**
    ```bash
    git lfs untrack "runs/final/**/*.json"        # edits .gitattributes
    git add --renormalize runs/final              # re-add JSONs as normal text (real content)
    git add .gitattributes
    git commit -m "Stop LFS-tracking result JSONs; keep LFS for model binaries only"
    ```
    Do this **only on the WSL tree that has the real JSONs** (never in a
    pointer-only clone, or you would commit the stubs).

---

## 1. Current results (authoritative — from `runs/final/reports/GATE_SUMMARY.md`)

FP16 PPL: **1B 9.757**, **3B 7.817**. Axis = **weight-only bits/param** (embeddings
excluded; comparable to GPTQ-4 = 4.0). Framing = **audit**.

### 1a. Baselines @ ~4 bits (single env, one evaluator, matched bits)
| model | GPTQ-4 | AWQ-4 | RTN-4 | HQQ-4 | HQQ-5 | HQQ-6 | HQQ-8 |
|---|---|---|---|---|---|---|---|
| 1B | **10.363** (4.29b) | 11.278 | 11.710 | 11.187 | 10.064 | 9.829 | 9.762 |
| 3B | **8.304** (4.28b) | 8.405 | 8.498 | 8.387 | 7.957 | 7.845 | 7.820 |

### 1b. Selectors on the **HQQ (RTN) base** — protection helps; interactions don't
PPL by protected fraction (weight bits). All selectors ≫ random.
| model | frac | bits | greedy | greedy_indep | residual_max | random (mean [95% CI]) |
|---|---|---|---|---|---|---|
| 1B | 0.02 | 4.82 | 10.495 | 10.506 | 10.533 | 11.165 [11.155, 11.174] |
| 1B | 0.20 | 7.70 | **10.207** | 10.230 | 10.232 | 10.975 [10.932, 11.018] |
| 3B | 0.02 | 4.82 | 8.149 | 8.151 | 8.161 | 8.376 [8.370, 8.382] |
| 3B | 0.20 | 7.70 | **8.028** | 8.037 | 8.048 | 8.243 [8.025, 8.462] |

→ greedy beats greedy_indep/residual_max by only **~0.02–0.03 PPL**, and on 3B
greedy_indep sometimes wins. **The interaction-aware machinery is not worth it.**

### 1c. Selectors on the **GPTQ base** — the headline finding
| model | frac | residual_max | random | **greedy** | **greedy_indep** |
|---|---|---|---|---|---|
| 1B | 0.02 | 10.391 | 10.680 | **104.16** | **15.64** |
| 1B | 0.20 | 10.350 | 10.650 | **106.82** | **15.61** |
| 3B | 0.02 | 8.099 | 8.161 | **55.34** | **44.88** |
| 3B | 0.20 | 8.070 | 8.342 | **43.21** | **45.41** |

→ **Residual-driven set selection is catastrophic on an error-compensated base**
(greedy 43–107, greedy_indep 15–47), while activation-magnitude `residual_max`
stays safe and even *helps*, and `random` is harmless. **Protection is antagonistic
to GPTQ's error compensation.**

### 1d. Gate verdict (pre-registered rule)
`greedy` must beat greedy_indep, residual_max, and random-CI in ≥3/4 budgets **in
every** model×base stratum. Result: **1B/HQQ PASS (4/4/4), 3B/HQQ PASS (3/3/4),
1B/GPTQ FAIL (0/0/0), 3B/GPTQ FAIL** → framing = **audit**.

### 1e. Honest Pareto (weight-only axis)
- **1B: no SEQ point on the frontier** — GPTQ-4 (4.29/10.363), HQQ-5 (5.00/10.064),
  HQQ-6, HQQ-8, FP16 dominate all protected-RTN points.
- **3B: one SEQ frontier point** — `residual_max` on the GPTQ base (**4.82 bits /
  8.099 PPL**) is non-dominated: lower PPL than GPTQ-4 (8.304) at a +0.5-bit premium.
  This is SEQ's single surviving positive, and the (currently broken) baseline-only
  table hides it — regenerating the table (§3, W1) will surface it.

---

## 2. What is DONE (both tracks)

- Reproducible staged pipeline `scripts/run_final_seq_pipeline.sh` (preflight →
  LLMC baselines → checkpoint validation → gate → gate summary → full matrix →
  validate → comparison → plots → atomic publish), portable paths, `.gitattributes`
  BOM fixed, matrix spec `configs/final_comparison_matrix.json`.
- Validated LightCompress **GPTQ-4** checkpoints for 1B/3B (replay diff <0.006 vs
  LLMC-reported PPL, full module coverage).
- Single-environment **matched baselines** RTN/AWQ/GPTQ/HQQ + FP16 on 1B/3B.
- Per-channel **selector sweep**: act_max, act_scale, residual_rms, residual_max,
  greedy, greedy_indep, random(×3 seeds) on **HQQ and GPTQ** bases, fractions
  {0,.02,.05,.1,.2}; value-tier (1B); uniform HQQ 4/5/6/8.
- **Honest weight-only storage accounting** (`seq_core/storage_accounting.py`); the
  earlier 7.9-vs-4.0 mis-plot is fixed.
- **greedy_indep** interaction-free ablation (isolates the interaction term).
- Reload/serialization validation; seed-CI on the random control.
- Earlier module-level audit **runs 1–6** (`docs/FINDINGS_run{1..6}.md`): entropy /
  Hessian / reconstruction proxies decoupled from PPL; module allocation ≤ uniform.
- **This commit:** `build_comparison` fails loudly on LFS-pointer/corrupt sweep
  inputs instead of silently degrading to a baseline-only table.

---

## 3. Track A — Findings / audit paper (the realistic publication)

**Status: ~75% of evidence exists.** Draft: `docs/FINDINGS_PAPER.md`.

### What is LEFT / TO DO (run in parallel)
- **W1 — Fix + regenerate the comparison table (CPU, minutes).** After the LFS
  migration (§0), regenerate so SEQ rows appear and the 3B `residual_max`-on-GPTQ
  frontier point shows:
  ```bash
  cd "/mnt/e/seq v4/SEQ-clean-v4" && source .venv-seq/bin/activate
  python analysis/build_comparison.py --sweeps runs/final/sweeps \
    --baselines runs/final/staging/results/baselines.json \
    --signals act_max,act_scale,residual_rms,residual_max,greedy,greedy_indep,random,tier_alloc \
    --require-sweep-points \
    --out docs/COMPARISON.md --csv results/final_comparison.csv --json results/final_comparison.json
  python analysis/plot_final_results.py --input results/final_comparison.csv --output-dir figures/final
  ```
  Verify: `grep -c 'SEQ:' docs/COMPARISON.md` is large (not 0); the 3B table shows
  `SEQ:residual_max(gptq_llmc-4b …)` marked ★.
- **W2 — Downstream task accuracy (GPU; biggest missing piece).** lm-eval
  (hellaswag, arc-easy, arc-challenge, piqa, winogrande) at the operating points
  (FP16, GPTQ-4, HQQ-4, best SEQ point) using the saved checkpoints, ≥ few-hundred
  examples/task. Needs a `scripts/run_downstream_eval.sh` phase (TO BUILD). Reviewers
  require more than PPL.
- **W3 — Consolidate module-level runs 1–6** into one "proxies mislead" section;
  build the related-work table (LLM.int8 [column, not row], SpQR, OWQ, CLAQ, Atom,
  SqueezeLLM; CoopQ/AMQ/SliM-LLM; EWQ).
- **W4 — (optional) robustness:** 8B and/or one cross-family (Qwen2.5-3B) for the
  two headline findings only (base-is-ceiling; GPTQ antagonism).
- **W5 — Writing + stats appendix** (seed CIs; paired bootstrap on downstream).

**Venue:** EMNLP/ACL **Findings** or an efficiency workshop (ENLSP@NeurIPS).
**Claim ceiling:** an audit with a sharp mechanistic finding — **not** a SOTA claim.

---

## 4. Track B — Improving SEQ as a method (mostly closed by the data)

**Status: the interaction-aware greedy direction is dead**; it does not beat simple
selection on HQQ and is *toxic* on GPTQ. The only surviving positive is
**activation-magnitude protection (`residual_max`/`act_max`) on a GPTQ base**, a
Pareto point on 3B (§1e).

### The one idea with real headroom — **protect-then-recompensate**
The catastrophe (§1c) is *caused* by protecting AFTER compensation. The fix is to
**choose the FP16/high-precision columns first, then run GPTQ on the remaining
columns so it compensates around them** — so protection and compensation cooperate
instead of fighting. This is the only experiment that could turn the audit into a
method paper. Sketch:
1. Pick protected set S per layer (act_max or residual on the FP16 residual).
2. Feed LightCompress a per-layer mask (or a custom loop) that keeps S in FP16 and
   runs GPTQ error-compensation over the complement.
3. Compare to plain GPTQ-4 and to naive post-hoc protection at matched weight bits.
**Risk:** near SpQR/OWQ territory; novelty rests on the *ordering* and the matched-
bit audit. Keep it separate from the Track-A paper.

### Explicitly rejected (do not revisit)
- Module-level entropy allocation (runs 1–6) — dead prior art (EWQ), ≤ uniform.
- Interaction-aware greedy set selection — no robust win; toxic on GPTQ.
- Naive stacking of FP16 protection on a GPTQ base — catastrophic.

---

## 5. Command reference

```bash
# Full pipeline (resume-safe; publish only after validation)
bash scripts/run_final_seq_pipeline.sh --llmc-repo /mnt/e/LightCompress \
  --llmc-venv /mnt/e/LightCompress/.venv-llmc --output-root runs/final --resume
# ... inspect runs/final/reports/GATE_SUMMARY.md and MATRIX_VALIDATION.md, then:
bash scripts/run_final_seq_pipeline.sh ... --resume --publish

# Gate-only greedy/greedy_indep reruns (both bases, available models)
bash scripts/run_fixed_greedy_sweeps.sh --gptq-root runs/final/llmc \
  --output-root runs/final/sweeps --resume

# Regenerate the comparison table + plots (CPU) — see §3 W1
# Downstream eval (TO BUILD) — see §3 W2
```

---

## 6. Bottom line
The science is settled and honest: **activation-outlier protection gives a modest,
real gain on a data-free RTN/HQQ base; the interaction-aware selector does not earn
its cost; and protection is antagonistic to error-compensated (GPTQ) bases** — with
the base quantizer as the Pareto ceiling. That is a legitimate **Findings/workshop**
paper (Track A). Track B has one live lead (**protect-then-recompensate**) and is
otherwise closed. Immediate next actions: fix LFS + regenerate the table (W1), then
run downstream eval (W2).
