#!/usr/bin/env python3
"""Decisive GPTQ diagnostic — locate the bug (math vs. application vs. numerics).

Runs on one model, quantizes a few individual layers, and reports for each:
  - finite? / ||Wq|| vs ||W||        (is the quantized weight sane at all)
  - rel weight error ||W-Wq||/||W||  (GPTQ vs plain round-to-nearest)
  - rel OUTPUT error ||(W-Wq)X||/||WX|| on real calibration activations
    (this is what GPTQ minimizes; it MUST be <= RTN if GPTQ is correct)
  - Hessian condition (damped) — catches rank-deficiency / singular H

Reading:
  * GPTQ output error >> RTN, or NaN/Inf, or huge ||Wq||  -> the GPTQ MATH is broken.
  * GPTQ output error <= RTN and Wq sane                  -> math is fine; the bug is
    in how the fake-quant base is APPLIED (precomputed_base path), not gptq.py.

Usage:
  python scripts/diag_gptq.py --model meta-llama/Llama-3.2-1B --bits 4 --gptq_calib_samples 64
"""
import argparse
import sys

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group_size", type=int, default=128)
    ap.add_argument("--gptq_calib_samples", type=int, default=64)
    ap.add_argument("--seq_len", type=int, default=2048)
    args = ap.parse_args()

    from seq_core.pipeline import load_model_and_tokenizer, resolve_device, resolve_dtype
    from seq_core.gptq import (
        build_gptq_calibration,
        collect_gptq_hessians,
        gptq_quantize_weight,
    )

    dev = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, dev)
    model, tok = load_model_and_tokenizer(args.model, dev, dtype)

    texts = build_gptq_calibration(tok, n_samples=args.gptq_calib_samples, seq_len=args.seq_len)

    # capture a few real input batches per target layer (for output-error check)
    linears = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    targets = []
    for key in ("self_attn.q_proj", "mlp.down_proj", "mlp.gate_proj"):
        hit = next((n for n in linears if n.endswith(key) and ".0." in n), None) or \
              next((n for n in linears if n.endswith(key)), None)
        if hit:
            targets.append(hit)
    print("target layers:", targets)

    captured = {n: [] for n in targets}

    def mk(n):
        def hook(_m, inp):
            if inp and isinstance(inp[0], torch.Tensor) and len(captured[n]) < 4:
                captured[n].append(inp[0].detach().reshape(-1, inp[0].shape[-1])[:512].float().cpu())
        return hook

    hs = [dict(model.named_modules())[n].register_forward_pre_hook(mk(n)) for n in targets]

    accs = collect_gptq_hessians(model, tok, texts, seq_len=args.seq_len, device=dev, hessian_device="cpu")
    for h in hs:
        h.remove()

    mods = dict(model.named_modules())
    for name in targets:
        W = mods[name].weight.detach().float()
        H = accs[name][0].to(W.device)
        # damped condition number
        damp = 0.01 * torch.mean(torch.diag(H))
        Hd = H + damp * torch.eye(H.shape[0], device=H.device)
        try:
            cond = torch.linalg.cond(Hd).item()
        except Exception:
            cond = float("nan")
        rank = torch.linalg.matrix_rank(H).item()

        Wq = gptq_quantize_weight(W, H, args.bits, group_size=args.group_size)
        finite = bool(torch.isfinite(Wq).all())
        werr = (W - Wq).norm().item() / (W.norm().item() + 1e-9)

        # RTN baseline (per-group asymmetric, no error compensation)
        from seq_core.gptq import _find_params, _quantize_affine
        maxq = float(2 ** args.bits - 1)
        Wr = W.clone()
        gs = args.group_size
        for c in range(0, W.shape[1], gs):
            sc, ze = _find_params(W[:, c:c + gs], maxq)
            for j in range(c, min(c + gs, W.shape[1])):
                Wr[:, j] = _quantize_affine(W[:, j], sc, ze, maxq)
        rerr = (W - Wr).norm().item() / (W.norm().item() + 1e-9)

        # output error on real activations
        X = torch.cat(captured[name], 0).to(W.device) if captured[name] else None
        if X is not None and X.shape[1] == W.shape[1]:
            base = (X @ W.t()).norm().item() + 1e-9
            o_gptq = (X @ (W - Wq).t()).norm().item() / base
            o_rtn = (X @ (W - Wr).t()).norm().item() / base
        else:
            o_gptq = o_rtn = float("nan")

        print(f"\n[{name}]  shape={tuple(W.shape)}  H_rank={rank}/{H.shape[0]}  cond(Hd)={cond:.2e}")
        print(f"  Wq finite={finite}  ||Wq||/||W||={Wq.norm().item()/(W.norm().item()+1e-9):.3f}")
        print(f"  weight rel err : GPTQ={werr:.4f}   RTN={rerr:.4f}")
        print(f"  OUTPUT rel err : GPTQ={o_gptq:.4f}   RTN={o_rtn:.4f}   "
              f"({'GPTQ<=RTN OK' if o_gptq <= o_rtn + 1e-4 else 'GPTQ>RTN -> MATH BUG'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
