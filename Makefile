# All targets run from code/. Paper numbers come only from aggregate.py -> paper/numbers.tex.
SEEDS ?= 0 1 2 3 4
PY ?= python

.PHONY: all train posthoc triangulate aggregate validate paper smoke clean-rerun

all: train posthoc triangulate aggregate validate paper

train:
	cd code && for s in $(SEEDS); do for m in clean confounded; do \
	  $(PY) run.py --mode $$m --seed $$s --episodes 1000 --epochs 15 > results/$${m}_s$$s.log 2>&1 || exit 1; done; done
	cd code && RERUN=1 $(PY) run.py --mode confounded --seed 0 --episodes 1000 --epochs 15 > rerun/rerun.log 2>&1

posthoc:
	cd code && $(PY) posthoc.py

triangulate:
	cd code && $(PY) triangulate.py $(SEEDS)

aggregate:
	cd code && $(PY) aggregate.py

validate:
	cd code && $(PY) validate.py

paper:
	pdflatex -interaction=nonstopmode main >/dev/null && bibtex main >/dev/null && pdflatex -interaction=nonstopmode main >/dev/null && pdflatex -interaction=nonstopmode main >/dev/null

# Fast end-to-end test of the whole pipeline on a tiny budget (numbers are meaningless; checks the plumbing).
smoke:
	cd code && RERUN=1 $(PY) run.py --mode clean --seed 123 --episodes 60 --epochs 1 && RERUN=1 $(PY) run.py --mode confounded --seed 123 --episodes 60 --epochs 1
	@echo "smoke OK"
