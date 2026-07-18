#!/usr/bin/env python3
"""Validate that an LLMC fake-quant checkpoint reloads as the trusted base."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True); ap.add_argument('--gptq-model-path',required=True); ap.add_argument('--llmc-reported-ppl',type=float,required=True); ap.add_argument('--tolerance',type=float,default=0.05); ap.add_argument('--device',default='cuda'); ap.add_argument('--dtype',default='float16', choices=['float16','bfloat16','float32']); ap.add_argument('--output',type=Path,default=Path('results/gptq_llmc_validation.json')); args=ap.parse_args()
    import torch
    from seq_core.pipeline import load_model_and_tokenizer, resolve_dtype, unload_model
    from seq_core.sensitivity import make_ppl_fn
    from seq_core.gptq_llmc_base import load_llmc_fake_quant_base
    dtype=resolve_dtype(args.dtype,args.device); model,tok=load_model_and_tokenizer(args.model,args.device,dtype)
    names={n:m.in_features for n,m in model.named_modules() if isinstance(m,torch.nn.Linear)}
    ppl_fn=make_ppl_fn(dataset_name='wikitext2',split='test',seq_len=2048,device=args.device,dtype=dtype,mode='canonical',max_examples=None,full_corpus=True,seed=1234)
    fp16=float(ppl_fn(model,tok)); base=load_llmc_fake_quant_base(args.gptq_model_path,names,device='cpu',dtype=dtype)
    matched=missing=shape=0; max_diff=0.0
    for n,m in model.named_modules():
        if not isinstance(m,torch.nn.Linear): continue
        if n not in base: missing+=1; continue
        w=base[n]
        if tuple(w.shape)!=tuple(m.weight.shape): shape+=1; continue
        matched+=1; max_diff=max(max_diff,float((m.weight.detach().cpu().float()-w.float()).abs().max()))
        m.weight.data.copy_(w.to(device=m.weight.device,dtype=m.weight.dtype))
    loaded=float(ppl_fn(model,tok)); unload_model(model,tok); diff=abs(loaded-args.llmc_reported_ppl)
    report={'model':args.model,'original_fp16_ppl':fp16,'llmc_reported_ppl':args.llmc_reported_ppl,'loaded_gptq_base_ppl':loaded,'ppl_difference':diff,'maximum_sampled_weight_difference':max_diff,'matched_modules':matched,'missing_modules':missing,'shape_mismatches':shape,'tolerance':args.tolerance,'status':'PASS' if diff<=args.tolerance and matched>0 and shape==0 else 'FAIL'}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2)); return 0 if report['status']=='PASS' else 1
if __name__=='__main__': raise SystemExit(main())
