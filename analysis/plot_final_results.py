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
    args=ap.parse_args()
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f'matplotlib unavailable; plots pending: {exc}'); return 0
    points={}
    if args.input:
        with args.input.open(newline='', encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                try:
                    points.setdefault(row['model'],[]).append(
                        (float(row['bits']),float(row['ppl']),str(row['method']))
                    )
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
                    points.setdefault(model,[]).append((float(bits),float(r['ppl']),str(r.get('signal','unknown'))))
                except (TypeError,ValueError,KeyError): pass
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for model, rows in points.items():
        if not rows: continue
        fig,ax=plt.subplots(figsize=(6.2,4.2)); labels=sorted(set(x[2] for x in rows))
        for label in labels:
            vals=[x for x in rows if x[2]==label]; vals.sort(); ax.plot([x[0] for x in vals],[x[1] for x in vals],marker='o',label=label)
        ax.set_xlabel('Effective bits per parameter'); ax.set_ylabel('WikiText-2 perplexity'); ax.set_title(model); ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout()
        slug=model.replace('/','_').replace(' ','_'); fig.savefig(args.output_dir/f'ppl_vs_actual_bits_{slug}.pdf'); plt.close(fig)
    print(f'wrote plots to {args.output_dir}'); return 0
if __name__=='__main__': raise SystemExit(main())
