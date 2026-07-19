# Findings run 4 — per-channel protection works (and the novelty question)

Per-channel column protection: quantize each layer to 4-bit, keep the top-k%
input channels (by signal) in FP16. Canonical PPL, 3 models. `random` is the
control at matched k (== matched effective bits).

## Result: activation-aware channel selection beats random, everywhere

PPL gap vs `random` at each k (negative = the signal beats chance):

| model | signal | k=2% | k=5% | k=10% | k=20% |
|---|---|---|---|---|---|
| 1B | `act_scale` | −0.54 | −0.56 | −0.59 | −0.59 |
| 1B | `hessian_diag` | −0.53 | −0.55 | −0.56 | −0.56 |
| 1B | `magnitude` (weights) | −0.02 | −0.04 | −0.03 | −0.01 |
| 3B | `act_scale` | −0.17 | −0.19 | −0.18 | −0.20 |
| 8B | `act_scale` | −0.18 | −0.19 | −0.19 | −0.18 |

Concretely (1B): uniform 4-bit = PPL 11.19 (FP16 = 9.76). Protecting **2%** of
channels by `act_scale` → **10.62** at only **4.24 effective bits** — recovers
~40% of the quantization loss for +0.24 bits. Weight `magnitude` ≈ random.

**Conclusion:** per-channel importance is real and **activation-driven**. This
is the positive result module-level (run 3) could not produce.

## The novelty problem (read this)

The result above, on its own, **reproduces known work**:

- `act_scale` = mean|x| per channel = **AWQ**'s salience metric (Lin et al. 2023).
- keeping outlier channels in FP16 = **LLM.int8()** (Dettmers et al. 2022).
- `hessian_diag` = the **GPTQ** second-order quantity.
- `salience` = **AWQ** salience almost verbatim.

And empirically `act_scale` (pure activation magnitude) is the **best** of the
current signals — i.e. we re-derived "protect high-activation channels." That is
not, by itself, an ACL/ICLR contribution.

### Where genuine novelty can come from

1. **A channel signal that beats the magnitude heuristic.** AWQ/LLM.int8 select
   by magnitude (E|x| / max|x|). We now also compute per-channel `act_max`,
   `act_rms`, `act_kurt` (heavy-tail index), and `act_entropy` (information
   content, orthogonal to magnitude). **If any beats `act_scale` consistently,
   that is a new, SOTA-relevant selection criterion.** (Run 5.)
2. **A per-channel true-sensitivity audit.** Measure, per channel group, the real
   ΔPPL from protecting it, and show *which channels magnitude misses* — then a
   signal that captures them. This is the per-channel analog of the run-1→3
   audit that already produced surprising, publishable reversals.
3. **The falsification methodology itself.** The paper's spine is unusual and
   honest: each level of rigor overturned the previous "winner" —
   entropy (run 1) → reconstruction/Hessian (run 2) → *both wrong end-to-end*,
   module allocation can't beat uniform (run 3) → value exists only per-channel
   (run 4). The headline insight — **local proxies (entropy, reconstruction
   error) are decoupled from, even anti-correlated with, end-to-end quality** —
   is a real methodological contribution with receipts, and reframes how the
   field should validate quantization "importance" signals.
4. **Backend-agnostic, arbitrary-bit protection.** LLM.int8 is int8-specific,
   AWQ is ~4-bit; our protection sits on any HQQ base bit-width (3/4/5/6). A
   Pareto showing it dominates at 3-bit base (where protection matters most)
   across backends is a systems contribution.

**Honest recommendation:** the strongest paper = (3) as the spine + (1)/(2) as
the positive method, i.e. *"a rigorous audit of quantization importance signals
showing popular proxies mislead, and a per-channel activation signal that beats
the magnitude heuristic used by AWQ/LLM.int8."* Novelty rides on beating
`act_scale`, so Run 5 is the decisive experiment.

## Answering: "should we measure entropy channel-wise?"

Yes — and it is now built (`--channel_entropy` → signal `act_entropy`;
`collect_channel_activation_entropy`). It measures a channel's information
content, **orthogonal to the magnitude** AWQ/LLM.int8 use, so it is exactly the
kind of signal that could beat them. `act_max` (outlier magnitude) and
`act_kurt` (heavy-tailedness) are also new candidates. Whether they win is an
empirical question — Run 5 answers it.

### Run 5 recipe: can any signal beat activation magnitude?

```bash
python -m seq_core.channel_sweep \
  --model meta-llama/Llama-3.2-1B --backend hqq \
  --base_bits 4 --protect_fracs 0,0.01,0.02,0.05,0.1 \
  --signals act_scale,act_max,act_rms,act_kurt,hessian_diag,random \
  --channel_entropy --entropy_bins 32 \
  --ppl_mode canonical --calibration_prompts calibration_prompts.json \
  --out_dir runs/channel5/Llama-3.2-1B
```

Also run `--base_bits 3` (protection matters most there). The signal with the
most-negative gap vs `act_scale` — if any — is the novel contribution.
