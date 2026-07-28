# Responsible NLP Research checklist — prepared answers

ARR requires this checklist at submission. It is filled in the submission portal, not in the PDF;
this file holds the answers and a pointer to where each is evidenced, so submission is
transcription rather than authorship. Section letters follow the ARR form.

Paper: *When Perplexity Lies: A Controlled Perplexity Collapse with No Downstream Degradation in
Weight-Quantized LLMs.*

---

## A. For every submission

**A1. Did you describe the limitations of your work?** **Yes** — a required, unnumbered
`Limitations` section (`paper/main.tex`, after §7). It states: no positive mechanism account is
offered, only a refuted one and a structural condition (§6, §5.2); §5.3's comparison is
size-matched rather than sample-matched, so calibration-sample mismatch is bounded but not
eliminated; the no-pad calibration robustness check was run on 3B only *(update to "both models"
if Job A lands)*; two of four checkpoints show the collapse, so model scope is a real boundary
(§5.4); and one 1B export failed reload validation and was re-exported (Appendix A).

**A2. Did you discuss any potential risks of your work?** **Yes** — `Ethical considerations`.
The paper documents a way to make a model look badly degraded on the standard metric while it is
not. The honest risk is the converse inference: a small perplexity movement is equally weak
evidence that a quantized model is *safe*. Practitioners with limited evaluation budgets are the
most exposed. We state the results are benchmark findings, not guarantees for safety-critical
deployment.

**A3. Do the abstract and introduction summarize the paper's main claims?** **Yes** — abstract
(196 words) and §1 Contributions. Both scope the claim explicitly: we do not claim perplexity is
uninformative, nor that this configuration is one a practitioner would choose.

**A4. Have you used AI assistants in this research?** **Yes — declare.** Claude (Anthropic) was
used as a coding and analysis assistant: implementing and debugging the quantization/selection
code, running experiments, and drafting prose that the authors reviewed and revised. All
experimental results are produced by the committed code and are reproducible from the released
artifacts. Every number in the paper is read from a committed JSON artifact, not from model
output. An AI-assisted analysis error was found by external audit and is corrected and disclosed
in-place in §6.

---

## B. Did you use or create scientific artifacts?

**B1. Did you cite the creators?** **Yes** — §2 and the bibliography: GPTQ, HQQ, AWQ, LLM.int8(),
OWQ, SpQR, the Llama and Qwen model families, WikiText-2, and `lm-evaluation-harness` (v0.4.12).

**B2. Did you discuss the licence / terms of use?** **To add at submission.** Models are used
under their published licences (Llama 3.2 Community Licence; Qwen2.5 Apache-2.0; Llama-2
Community Licence). WikiText-2 is CC BY-SA 3.0. `lm-evaluation-harness` and LightCompress are
MIT. Our released code and result artifacts: MIT.

**B3. Is your use consistent with their intended use?** **Yes.** All artifacts are research
benchmarks and openly released research models, used for benchmark evaluation of a quantization
method — their stated purpose. No model is deployed or served.

**B4. Did you discuss steps taken to protect privacy / anonymize?** **Not applicable to data.**
No personal or personally identifying data is used: WikiText-2 is Wikipedia text and the six
zero-shot tasks are public benchmarks. No new data was collected. For *author* anonymity see the
artifact note below.

**B5. Did you document the artifacts?** **Yes** — `docs/REPRODUCIBILITY.md`, `docs/PROJECT_STATUS_AND_ROADMAP.md`,
Appendix A (Provenance), and the curated release `protection-paradox` with per-experiment READMEs,
`COMMANDS.txt`, `docs/PROVENANCE.md` and `docs/KNOWN_DEVIATIONS.md`. Domains, languages
(English only) and demographic scope are stated; known deviations and excluded experiments are
documented rather than omitted.

**B6. Did you report statistics of the data?** **Yes** — §3: WikiText-2 test evaluated as a
continuous stream in non-overlapping 2048-token windows (141 windows, 288,627 supervised targets
on 3B); calibration is 128 × 2048 tokens; the six zero-shot tasks are evaluated on their full
sets (1172/2376/10042/5153/1838/1267 examples), with any reduced-scope run marked in
`docs/DOWNSTREAM.md`.

---

## C. Did you run computational experiments?

**C1. Did you report the number of parameters and total computational budget?**
**Partially — needs a line added to the paper.** Parameter counts are implicit in the model names
(1.24B / 3.21B / 3.09B / 6.74B). Compute is **not** currently reported. Add to §3 or Appendix A:
experiments ran on a single consumer GPU (NVIDIA RTX 4090 / 5090, 24–32 GB) under WSL2; the twelve
LightCompress baseline quantizations total 2.2 GPU-hours (recorded in each
`runs/final/llmc/**/summary.json` as `duration_sec`); the full sweep, export, reload-validation and
downstream-evaluation matrix is on the order of 60–80 GPU-hours across the project, including
discarded and re-run experiments.

**C2. Did you discuss experimental setup, including hyperparameters?** **Yes** — §3 and Appendix A.
GPTQ-4 g128 with `actorder`, `percdamp` 0.01, blocksize 128, true-sequential; HQQ-4 g64; protection
budgets {2, 5, 10, 20}%; selector calibration size and source recorded per run; `skip_lm_head` and
`base_group_size` recorded in every output JSON so each artifact is self-describing.

**C3. Did you report descriptive statistics — mean/variance, and how results were computed?**
**Yes, with a stated caveat.** Downstream contrasts are paired bootstrap over per-example
correctness with 95% CIs (Appendix B). Random channel-selection controls are run at three seeds
and the seed-1234 value is tabled with the range shown. Perplexity points are **single
deterministic runs**, not multi-seed means — stated rather than implied, since the effect sizes
(8.17 → 51.68) are orders of magnitude larger than the seed spread.

**C4. Did you report the implementation, model and parameters of existing packages?** **Yes** —
`lm-evaluation-harness` v0.4.12 with the exact task list, zero-shot, batch size 1; LightCompress
at a pinned commit recorded in each `summary.json` (`llmc_commit`); Hugging Face `transformers`
for loading. Every evaluation's raw output JSON is released.

---

## D. Did you use human annotators or research with human participants?

**No** — not applicable. No human annotation, no crowdsourcing, no participants, so D1–D5 are
all "not applicable."

---

## E. AI assistants in research or writing

Covered in **A4** above. ARR's policy is disclosure, not prohibition: Claude was used for
implementation, debugging, experiment orchestration and drafting; the authors are responsible for
all claims. Note in the disclosure that an AI-assisted analysis probe contained an error
(measuring a diagonal proxy instead of the selector's true objective), that it was caught by
external audit, and that §6 carries the correction and the superseded number in-place rather than
silently replacing it.

---

## Artifact anonymity (not a checklist item, but blocking for ARR)

The submission must not link an authored remote. Plan: serve the repository through
`anonymous.4open.science`, which hides the origin URL, from a scrubbed branch — the author's name
currently appears in 105 tracked files and 11 lm-eval output directory names
(`__mnt__d__Abrar__SEQ__seq_v4__...`). Verify with `git grep -i` before publishing the link. The
artifact link goes in the camera-ready only; the marked block in `paper/main.tex` (search
`ANONYMOUS: add the artifact link`) is where it belongs.

---

## Open items before submission

1. **C1** — add the compute paragraph to the paper. This is the one checklist answer the PDF
   cannot currently support.
2. **B2** — confirm each licence string against the model cards at submission time.
3. **A1** — if Job A lands, change the no-pad limitation from "3B only" to both models.
4. Anonymised mirror prepared and link-checked.
