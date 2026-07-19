#!/usr/bin/env python3
"""Generate publication plots from validated comparison data or sweep JSON."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path('runs'))
    ap.add_argument('--input',type=Path,help='authoritative comparison CSV (preferred)')
    ap.add_argument('--output-dir','--out-dir',dest='output_dir',type=Path,default=Path('figures'))
    ap.add_argument('--require-output',action='store_true',help='fail if plotting is unavailable or no plot is written')
    args=ap.parse_args()
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f'matplotlib unavailable; plots pending: {exc}'); return 1 if args.require_output else 0
    points={}
    if args.input:
        with args.input.open(newline='', encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                try:
                    points.setdefault(row['model'],[]).append({
                        'bits': float(row['bits']), 'ppl': float(row['ppl']),
                        'method': str(row['method']),
                        'ci_low': float(row['ppl_ci_low']) if row.get('ppl_ci_low') else None,
                        'ci_high': float(row['ppl_ci_high']) if row.get('ppl_ci_high') else None,
                    })
                except (TypeError,ValueError,KeyError):
                    pass
    else:
        for p in args.root.glob('**/channel_pareto.json'):
            try: d=json.loads(p.read_text(encoding='utf-8'))
            except Exception: continue
            model=d.get('model',p.parent.name)
            for r in d.get('results',[]):
                try:
                    bits = r.get('actual_effective_bits', r.get('effective_bits'))
                    points.setdefault(model,[]).append({
                        'bits': float(bits), 'ppl': float(r['ppl']),
                        'method': str(r.get('signal','unknown')), 'ci_low': None, 'ci_high': None,
                    })
                except (TypeError,ValueError,KeyError): pass
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for model, rows in points.items():
        if not rows: continue
        fig,ax=plt.subplots(figsize=(6.2,4.2)); labels=sorted(set(x['method'] for x in rows))
        for label in labels:
            vals=[x for x in rows if x['method']==label]; vals.sort(key=lambda x:x['bits'])
            xs=[x['bits'] for x in vals]; ys=[x['ppl'] for x in vals]
            ax.plot(xs,ys,marker='o',label=label)
            if vals and all(x['ci_low'] is not None and x['ci_high'] is not None for x in vals):
                ax.fill_between(xs,[x['ci_low'] for x in vals],[x['ci_high'] for x in vals],alpha=.15)
        ax.set_xlabel('Average bits per quantized linear weight (metadata included)'); ax.set_ylabel('WikiText-2 perplexity'); ax.set_title(model); ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout()
        slug=model.replace('/','_').replace(' ','_'); fig.savefig(args.output_dir/f'ppl_vs_actual_bits_{slug}.pdf'); plt.close(fig)
    if args.require_output and not any(args.output_dir.glob('ppl_vs_actual_bits_*.pdf')):
        print('no plots were written'); return 1
    print(f'wrote plots to {args.output_dir}'); return 0
if __name__=='__main__': raise SystemExit(main())
