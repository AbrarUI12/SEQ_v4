# Paper source

LaTeX source for *When Perplexity Lies: A Controlled Perplexity Collapse with No Downstream
Degradation in Weight-Quantized LLMs*. The prose and every number are kept in sync with
[`../docs/FINDINGS_PAPER.md`](../docs/FINDINGS_PAPER.md), which is the working document; this
directory is the typeset version.

## Build

```bash
make          # latexmk -pdf main.tex
make clean    # remove build artifacts
```

or directly:

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## ACL style

`main.tex` compiles **out of the box** with a plain fallback layout, so the content can be
proof-read anywhere. To produce the real ACL layout, drop the two official style files into this
directory and rebuild — the preamble switches automatically:

```bash
curl -O https://raw.githubusercontent.com/acl-org/acl-style-files/master/latex/acl.sty
curl -O https://raw.githubusercontent.com/acl-org/acl-style-files/master/latex/acl_natbib.bst
make
```

With `acl.sty` present the document uses `\usepackage[review]{acl}` (line numbers, anonymous).
For camera-ready, change `review` to `final`.

## Anonymity

This source is **anonymous**, as ARR requires:

- the author block is `Anonymous ARR submission`;
- there is no link to the code repository anywhere in the source.

Two marked comment blocks in `main.tex` show where to add the author list and the artifact link
when de-anonymising. **Do not add a link to an authored GitHub remote in a submission** — ARR
treats that as an anonymity violation. Prepare an anonymized mirror instead.

## Layout

| file | contents |
|---|---|
| `main.tex` | the paper |
| `refs.bib` | bibliography (17 entries) |
| `figures/` | the four figures cited, copied from `../figures/final/` |
| `Makefile` | build targets |

Figures are regenerated from committed result JSON by
`python ../analysis/plot_final_results.py --input ../results/final_comparison.csv --output-dir ../figures/final`,
then copied here.
