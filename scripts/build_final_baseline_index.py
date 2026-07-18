#!/usr/bin/env python3
"""Collect validated per-method LLMC summaries into a comparison baseline file."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path('runs/final_llmc')); ap.add_argument('--output',type=Path,default=Path('results/final_baselines.json')); args=ap.parse_args()
    out={}
    for p in sorted(args.root.glob('*/*/summary.json')):
        d=json.loads(p.read_text());
        if d.get('status') != 'completed' or d.get('ppl') is None: continue
        model=d['model']; artifact=p.parent/'artifacts'/'fake_quant_model'; size=sum(x.stat().st_size for x in artifact.glob('*.safetensors')) if artifact.exists() else None
        row={'method':d['method'],'bits':float(d.get('bits',4.0)),'nominal_bits':float(d.get('bits',4.0)),'ppl':float(d['ppl']),'model_path':str(artifact),'serialized_checkpoint_bytes':size,'calibration':d.get('calibration'),'source':'LLMC'}
        out.setdefault(model,[]).append(row)
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print('wrote',args.output); return 0
if __name__=='__main__': raise SystemExit(main())
