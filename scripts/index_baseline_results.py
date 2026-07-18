#!/usr/bin/env python3
"""Build a model_index.json from run_compare_matrix method summaries."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, required=True); ap.add_argument('--output', type=Path, required=True); ap.add_argument('--models', default=''); args = ap.parse_args()
    out = {m.strip(): {} for m in args.models.split(',') if m.strip()}
    for p in args.root.glob('**/summary.json'):
        try: d = json.loads(p.read_text())
        except Exception: continue
        model = d.get('model') or d.get('model_name'); method = str(d.get('method') or d.get('method_name') or p.parent.name).lower()
        if not model: continue
        row = out.setdefault(model, {})
        ppl = d.get('ppl', d.get('perplexity', d.get('baseline_ppl')))
        try:
            if ppl is not None and math.isfinite(float(ppl)): row[f'{method}_ppl'] = float(ppl)
        except (TypeError, ValueError): pass
        for k in ('model_path', 'save_dir', 'run_dir'):
            if d.get(k): row[f'{method}_model_path'] = d[k]; break
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + '\n'); print('wrote', args.output)
    return 0
if __name__ == '__main__': raise SystemExit(main())
