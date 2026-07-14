# SEQ: Research Plan & Paper Narrative

Target venue: ACL (August ARR cycle) / ICLR — an *analysis + method* paper.
Goal: at **5–7 effective bits**, match or beat FP16 perplexity, and show *why*
by identifying the signal that should decide per-unit precision.

---

## 1. Framing (the honest version)

SEQ today allocates precision from **one scalar entropy per Linear module**,
computed from a z-standardized 256-bin histogram of the weights (and a second
one from activations). Percentile thresholds on those two numbers pick the
`int4 / int8 / fp16` tier. The base quantizer is **bitsandbytes** (nf4 RTN /
LLM.int8).

Three structural weaknesses motivate the paper:

1. **The entropy signal is near-constant across modules.** After per-tensor
   standardization every weight matrix is ~Gaussian, and the discretized
   entropy of a standard-normal histogram is a fixed number. The only thing
   that moves it is deviation from Gaussianity, so module entropy is a *coarse
   kurtosis proxy* with low dynamic range and low discriminative power.
2. **Entropy is not sensitivity.** Quantization error → loss is governed by the
   activation covariance / Hessian and by weight outliers, not by the Shannon
   entropy of the weight marginal. For weights the mapping is arguably
   *backwards*: high-entropy (light-tailed) matrices are the *easiest* to
   quantize, yet the policy protects them.
3. **Module granularity is too coarse.** LLM quantization sensitivity is
   dominated by a few **input channels** (outlier features). Averaging over the
   whole matrix erases that structure.

The paper does not assume any of this — it **measures** it.

---

## 2. Research questions

- **RQ1 (signal quality).** How predictive of *true* quantization sensitivity
  is the current module-level entropy signal, relative to alternatives, and at
  what granularity (module / output-channel / input-channel / group)?
- **RQ2 (metric zoo).** Do other per-unit metrics — magnitude, kurtosis,
  activation scale (AWQ), Hessian-diagonal (GPTQ), Fisher, direct output-MSE —
  predict sensitivity better than entropy, alone or combined? By how much does
  the downstream Pareto front move?
- **RQ3 (backend).** Is bitsandbytes a limiting factor? Does decoupling the
  allocation policy from the backend (HQQ / GPTQ / torchao / quanto), and
  unlocking arbitrary bit-widths, improve the bits↔PPL Pareto front?

---

## 3. The keystone: a ground-truth sensitivity benchmark

Every claim about "signal quality" is grounded in a measured sensitivity per
unit `u`:

- **One-hot degrade (primary).** From FP16, quantize *only* `u` to `b` bits,
  keep everything else FP16, measure `ΔPPL_u = PPL(quant u) − PPL(FP16)`.
  Rank all units. Feasible per-module on 1B–3B models.
- **One-hot protect (complement).** Quantize *everything* to `b` bits, restore
  only `u` to FP16, measure the PPL recovered. Measures the marginal value of
  protection.
- **Second-order estimate (per-channel scale).** `ΔL ≈ ½ Σ_i H_ii Δw_i²` using
  `diag(XᵀX)` from one calibration pass — the GPTQ/AWQ quantity. Lets us reach
  channel granularity without O(#channels) forward evals.

**RQ1/RQ2 result = rank-correlation (Spearman ρ, Kendall τ) of each candidate
signal against this ground truth**, per granularity, plus the downstream test:
allocate a fixed bit budget by each signal and compare PPL.

Implemented in `seq_core/sensitivity.py` (harness) + `seq_core/stats_utils.py`
(correlation + allocation, pure-stdlib, unit-tested).

---

## 4. Signal × granularity grid (RQ1/RQ2)

Signals (`seq_core/signals.py`), each computable at module / out-channel /
in-channel granularity:

| Signal | Captures | Cost | Hypothesis |
|---|---|---|---|
| Entropy (current) | tail-shape of the marginal | cheap | weak (baseline) |
| Magnitude `E|w|` | scale | cheap | weak–moderate |
| Kurtosis / outlier-fraction | what entropy really proxies | cheap | moderate |
| Activation scale `E|x|` (AWQ) | salient input channels | 1 calib pass | strong (in-ch) |
| Hessian-diag `E[x²]·w²` (GPTQ) | 2nd-order sensitivity | 1 calib pass | strongest |
| Fisher `E[(∂L/∂w)²]` | 2nd-order (label-aware) | 1 grad pass | strong |
| Output-MSE `‖WX−Q(W)X‖` | empirical local error | cheap-ish | strong |

Deliverable: a table of ρ/τ per (signal × granularity), and a Pareto plot of
PPL vs. effective bits when each signal drives allocation.

---

## 5. Backend study (RQ3)

Decouple policy from backend behind `seq_core/quantizers/` (`QuantBackend`
interface). Concrete backends:

- **bitsandbytes** (existing): 4/8/16 only, RTN — the incumbent / lower bound.
- **HQQ** (first new): 1–8 bit, fast, minimal calibration — unlocks 3/5/6-bit
  and cheap per-channel bit allocation.
- **GPTQ** (via existing LightCompress wrapper): best 3–4 bit quality.
- **torchao / quanto** (optional): native low-bit + per-axis granularity.

Claims to establish: (a) SEQ's policy is **backend-agnostic** (improves every
backend's Pareto front); (b) bnb's coarse bit grid is a real limitation —
arbitrary-bit backends dominate at 5–7 effective bits; (c) best combo =
`signal ∘ backend` beats each SOTA baseline at equal effective bits.

---

## 6. Experimental matrix

- **Models:** TinyLlama-1.1B (dev), Llama-3.2-1B / 3B, Qwen2.5-3B/7B, Mistral-7B.
- **Eval:** canonical WikiText-2 PPL (2048, full corpus) + lm-eval
  (hellaswag, arc-e/c, piqa, winogrande, lambada) + SEQ metrics (tail-risk,
  json-stress, latency/memory).
- **Baselines:** RTN, GPTQ, AWQ, SmoothQuant, OmniQuant, SpinQuant (already
  wired via LightCompress/OmniQuant), plus uniform 4-bit at matched bits.
- **Bit budgets:** sweep effective bits ∈ {4, 4.5, 5, 5.5, 6, 7}.
- **Seeds:** ≥3 for the headline table; report mean ± std.

---

## 7. Contributions (as they'd read in the paper)

1. A **sensitivity ground-truth protocol** for measuring quantization-signal
   quality, reframing "which signal decides the bits" as a measurable question.
2. A **signal × granularity study** showing entropy is a weak, low-variance
   proxy and that Hessian/activation-aware **channel** signals are far more
   predictive — with the correlation-to-Pareto link made explicit.
3. A **backend-agnostic** mixed-precision policy showing bitsandbytes is a
   limiting factor, and that arbitrary-bit backends + our signal push the
   bits↔PPL Pareto front past strong baselines at 5–7 effective bits.

---

## 8. Risks / reviewer objections to pre-empt

- *"5–7 bits lossless is easy."* True for good 4-bit methods — so the headline
  is the **Pareto** and the **analysis**, not the lossless point itself.
- *"Why not just GPTQ/AWQ?"* Because those are *uniform-bit* per-tensor/-group;
  our claim is orthogonal (per-unit bit allocation on top of any of them).
- *"Entropy strawman."* We include it as one signal among many and let the
  measured correlation speak; if it wins anywhere, we report that.

---

## 9. Run 1 result (module-level, 3 models) — see `docs/FINDINGS_run1.md`

First study on Llama-3.2-1B / 3B / 8B (HQQ 3-bit one-hot degrade, proxy PPL):

- **The module-level ground truth is under-powered.** 26–65% of modules are
  usable; the rest sit in the noise (up to 40% show *negative* ΔPPL). Effect is
  carried by 1–2 modules per model (a `down_proj` and `lm_head`).
- **Entropy is not validated.** It leads a weak field (ρ = +0.16/+0.12/+0.19)
  but is significant in only 1/3 models. Its weak positive signal is essentially
  the inverse of kurtosis (uniform weights ≈ slightly more 3-bit damage).
- **`hessian_diag` predicts within `down_proj`** (+0.51/+0.20/+0.39) but its
  extensive module sum is size-dominated and just flags `lm_head` → motivates
  per-parameter normalization (added: `hessian_diag_pp`, `salience_pp`) and
  per-channel granularity.

Conclusion: module-level scalar signals are the wrong abstraction; the protocol
must be strengthened (bigger PPL, operating-regime bits) and moved to channels.

## 10. Build status (this branch)

- [x] `seq_core/stats_utils.py` — rank-correlation + **significance + reliability** + bit-allocation (stdlib, tested)
- [x] `seq_core/signals.py` — per-module + per-channel signals (now incl. `_pp` normalized forms)
- [x] `seq_core/sensitivity.py` — ground-truth one-hot sensitivity harness
- [x] `seq_core/quantizers/` — pluggable backend interface + bnb + HQQ
- [x] `seq_core/signal_study.py` — RQ1/RQ2 driver (signals → sensitivity → ρ/τ)
- [x] `analysis/analyze_runs.py` — reliability/significance/per-type analysis → `docs/FINDINGS_run1.md`
- [x] `seq_core/recon_sensitivity.py` — deterministic reconstruction-error ground truth
      (per-module + per-channel), wired into `signal_study.py` via `--ground_truth recon`
- [x] backend `dequantize_weight` (HQQ + bnb-4bit) for the real ΔW
- [x] `seq_core/pareto_sweep.py` — downstream PPL-vs-bits per signal (the non-circular test)
- [ ] Multi-model recon (3B/8B) to confirm the flip generalizes
- [ ] Per-channel (within-module) allocation
- [ ] GPTQ/AWQ-under-policy integration

## 11. Run 2 result (reconstruction ground truth, Llama-3.2-1B) — `docs/FINDINGS_run2.md`

Switching to the deterministic reconstruction ground truth flips the picture and
removes the noise (0% negative, 96% usable):

- `hessian_diag` ρ = **+0.997** (robust: same without `lm_head`, +0.85–1.00
  within every module type); `salience` +0.99.
- `entropy` ρ = **−0.18** — actively anti-correlated, not merely weak.
- Caveat: recon objective and `hessian_diag` are both E[x²]·weight-energy sums
  and ‖ΔW‖²≈c‖W‖² at fixed bits (CV≈0.19), so ρ≈1 confirms the `w²` proxy is
  faithful — it is not yet proof of the best end-to-end model.

## 12. Run 3 result (downstream Pareto, 3 models) — `docs/FINDINGS_run3.md`

The non-circular end-to-end test **overturns the reconstruction result**:

- `hessian_diag`/`salience` (recon winners, ρ≈0.99) are the **worst** allocators
  — mean PPL gap vs random +3.4 / +2.0. Their extensive per-layer sums
  over-protect `lm_head` and starve the rest to 3-bit (e.g. 1B @4b: 79% of
  params at 3-bit → PPL 30 vs ~11).
- `entropy` ≈ `magnitude` ≈ `random` (entropy −0.12 vs random): at module
  granularity, 3–7 bits, **no signal reliably beats uniform**. Entropy only
  "works" because it is near-uniform, so allocating by it ≈ uniform.
- Reconstruction correlation is therefore a **misleading proxy**: acing it (a
  local per-layer sum) predicts end-to-end failure because errors compound and
  concentration starves the network.

Implication: the module-level signal thesis is not supported. Open levers:
per-channel (within-layer) allocation, per-parameter-normalized signals
(`hessian_diag_pp`) with a low-bit floor, and a compounding-aware objective.
Salvage run (cheap): `--signals hessian_diag_pp,salience_pp,entropy,uniform,random --levels 4,8`.

## 16. GPTQ base (error-compensated) — the RQ3 lever (`seq_core/gptq.py`)

Run 6 proved RTN (bitsandbytes/HQQ) is the Pareto ceiling. GPTQ compensates
quantization error with the layer's input Hessian, so a GPTQ 4-bit base is far
closer to FP16 — the base a competitive Pareto needs. Faithful dependency-free
GPTQ (mirrors auto-gptq), fake-quant output usable by the same per-channel
`act_max` protection. Wired into `channel_sweep.py` via `--base_quantizer gptq`.

**Decisive question:** does `GPTQ-4bit + act_max protection` reach ≤ FP16 PPL at
5–7 effective bits (and beat AWQ/GPTQ baselines)? If yes → competitive method.

### Run 7 recipe: GPTQ base + per-channel protection

```bash
python -m seq_core.channel_sweep \
  --model meta-llama/Llama-3.2-1B --base_quantizer gptq \
  --base_bits 4 --gptq_group_size 128 \
  --protect_fracs 0,0.02,0.05,0.1,0.2 \
  --signals act_max,act_scale,random \
  --ppl_mode canonical --calibration_prompts calibration_prompts.json \
  --out_dir runs/gptq7_b4/Llama-3.2-1B
```

**SANITY-CHECK FIRST:** the `k=0` row = uniform GPTQ-4bit PPL. For Llama-3.2-1B
it should be ~10–11 (near FP16 9.76), *not* ~11.2 (HQQ) or 30+. If k=0 looks
wrong, GPTQ is misconfigured — stop and report before trusting the sweep.
(GPTQ code is faithful to auto-gptq and the affine math is unit-tested, but the
torch path is compile-checked only here.) Also try `--base_bits 3`. Run 1B/3B
(GPTQ Hessians + fake-quant weights are memory-heavy).

## 15. Per-channel true-sensitivity audit (`seq_core/channel_audit.py`)

The honest ground truth for channel importance: quantize the whole model to
`base_bits`, then for sampled layers protect each rank-bucket of channels one at
a time and measure the ΔPPL it recovers. Outputs (a) protection value by
`act_max`-rank bucket (monotonic ⇒ well-ordered) and (b) Spearman of every
signal's per-bucket mean vs measured value (a signal beating `act_max` here is a
candidate to beat the outlier-magnitude heuristic — the novelty lead). Pure
logic (`bucket_by_rank`) unit-tested; torch paths compile-checked.

### Audit recipe

```bash
python -m seq_core.channel_audit \
  --model meta-llama/Llama-3.2-1B --backend hqq \
  --base_bits 3 --rank_by act_max --num_buckets 8 --audit_layers 6 \
  --ppl_mode proxy --ppl_max_examples 128 \
  --calibration_prompts calibration_prompts.json \
  --out_dir runs/audit/Llama-3.2-1B
```

Cost ≈ 1 + audit_layers×num_buckets model reloads (proxy PPL); run on 1B/3B.

## 14. Run 4 result — per-channel protection WORKS (`docs/FINDINGS_run4.md`)

Activation-aware channel selection beats a random-channel control at every k,
all 3 models (1B gap −0.54 at k=2%, 4.24 eff bits; 3B/8B −0.17/−0.18). Weight
`magnitude` ≈ random → the signal that matters is **activation**, not weights.
The positive result module-level could not produce.

**Novelty caveat:** `act_scale` (best current signal) = AWQ's metric; FP16
protection of outlier channels = LLM.int8. Reproducing these is not A*-novel.
Novelty must come from (a) a channel signal that **beats activation magnitude**
(new: `act_max`, `act_rms`, `act_kurt`, `act_entropy`), (b) a per-channel
true-sensitivity audit, and/or (c) the falsification-methodology spine. Run 5
(below) is decisive. New per-channel signals + `--channel_entropy` are built.

## 13. Pivot: per-channel protection (the positive-result bet)

Run 3 killed module-level allocation. Within a layer, uniform is **not** the
ceiling — a small fraction of input channels (outlier features) genuinely need
precision (LLM.int8 / AWQ). We quantize each layer to `base_bits` and keep the
top-k% input channels (by a per-input-channel signal) in FP16, realized as a
full-layer quant + exact FP16 correction on protected columns
(`seq_core/channel_protect.py`, `ChannelProtectedLinear`). Base is numerically
identical to the uniform baseline, isolating the effect of protection.

**Decisive comparison:** signal-chosen vs. `random`-chosen protected channels at
the same k (same effective bits). Signal < random ⇒ per-channel importance is
real. Signals: `act_scale` (AWQ), `hessian_diag` (per-channel), `salience`,
`magnitude`; `k` small (0–20%) so effective bits land in 4–6.

### Run 4 recipe: per-channel sweep

```bash
python -m seq_core.channel_sweep \
  --model meta-llama/Llama-3.2-1B --backend hqq \
  --base_bits 4 --protect_fracs 0,0.02,0.05,0.1,0.2 \
  --signals act_scale,hessian_diag,salience,magnitude,random \
  --ppl_mode canonical --calibration_prompts calibration_prompts.json \
  --out_dir runs/channel/Llama-3.2-1B
```

Also try `--base_bits 3` (protection matters more at 3-bit) and
`--protect_bits 8` (int8 columns instead of FP16, cheaper). Status:
`channel_utils` unit-tested (12/12); torch paths compile-checked only.

### Run 3 recipe: downstream Pareto

```bash
python -m seq_core.pareto_sweep \
  --model meta-llama/Llama-3.2-1B --backend hqq \
  --signals hessian_diag,salience,magnitude,entropy,random \
  --budgets 4,5,6,7 --levels 3,4,8 --ppl_mode canonical \
  --calibration_prompts calibration_prompts.json \
  --out_dir runs/pareto
```

Produces PPL-vs-effective-bits per signal (the headline figure). If
`hessian_diag`/`salience` sit below `entropy`/`random` at 5–7 bits, the signal
story is proven end-to-end.

### Recommended run 2 (reconstruction ground truth; kills the PPL noise)

```bash
python -m seq_core.signal_study \
  --model meta-llama/Llama-3.2-1B \
  --backend hqq --sensitivity_bits 4 --group_size 64 \
  --ground_truth recon \
  --calibration_prompts calibration_prompts.json \
  --max_calib_prompts 64 --target_bits 6.0 \
  --out_dir runs/recon_study
```

Then `python analysis/analyze_runs.py` (it reads `sensitivity_recon.json` too) for
the significance-aware comparison. Use `--sensitivity_bits 4` because the target
is 5–7 effective bits — measure sensitivity in the operating regime, not 3-bit
collapse.
