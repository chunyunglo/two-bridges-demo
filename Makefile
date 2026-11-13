PYTHON ?= python3

.PHONY: demo quick test
demo:            ## the whole demonstration (about two minutes)
	$(PYTHON) demo.py
quick:           ## a shorter run for a first look
	$(PYTHON) demo.py --quick
test:
	$(PYTHON) -m pytest -q
