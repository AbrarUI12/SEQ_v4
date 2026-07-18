#!/usr/bin/env python3
"""Add measured FP16 and uniform-HQQ rows to the LLMC baseline index."""
from __future__ import annotations
import argparse,json
from pathlib import Path
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--index',type=Path,required=True); ap.add_argument('--uniform-root',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args(); d=json.loads(args.index.read_text())
 for p in args.uniform_root.glob('*/*/channel_pareto.json'):
  x=json.loads(p.read_text()); model=x['model']; r=x['results'][0]; label='HQQ-%d uniform'%int(x['base_bits']); d.setdefault(model,[]).append({'method':label,'bits':float(r.get('actual_effective_bits',r['effective_bits'])),'nominal_bits':float(r['effective_bits']),'ppl':float(r['ppl']),'source':'HQQ'})
  fp=x.get('baseline_fp16_ppl');
  if fp is not None and not any(z['method']=='FP16' for z in d[model]): d[model].append({'method':'FP16','bits':16.0,'nominal_bits':16.0,'ppl':float(fp),'source':'FP16'})
 args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n'); print('wrote',args.output); return 0
if __name__=='__main__': raise SystemExit(main())
