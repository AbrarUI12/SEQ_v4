# Build the paper. `latexmk` handles the pdflatex/bibtex/pdflatex/pdflatex cycle;
# the fallback target runs it manually if latexmk is unavailable.

MAIN = main

.PHONY: all clean check

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex refs.bib $(wildcard figures/*.pdf)
	@if command -v latexmk >/dev/null 2>&1; then \
		latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex; \
	else \
		pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex && \
		bibtex $(MAIN) && \
		pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex && \
		pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex; \
	fi

# Fail loudly on the two things that silently ruin a submission.
check: $(MAIN).pdf
	@echo "--- undefined references / citations ---"
	@! grep -E "(Reference|Citation).*undefined" $(MAIN).log || (echo "FAILED: see above" && false)
	@echo "none"
	@echo "--- overfull boxes ---"
	@grep -c "Overfull" $(MAIN).log || true

clean:
	@if command -v latexmk >/dev/null 2>&1; then latexmk -C >/dev/null 2>&1 || true; fi
	@rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out \
	       $(MAIN).fdb_latexmk $(MAIN).fls $(MAIN).pdf
