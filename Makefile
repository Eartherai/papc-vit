PYTHON ?= python3

.PHONY: install test analyze figures logs banner reproduce smoke-gpu clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

# Reproduce the paper's tables from the recorded result JSONs (no GPU needed).
analyze:
	$(PYTHON) scripts/analyze_results.py --results-dir results

# Regenerate result figures into assets/ (no GPU needed).
figures:
	$(PYTHON) scripts/make_figures.py --results-dir results --out-dir assets

# Export per-run logs from the result JSONs into logs/ (no GPU needed).
logs:
	$(PYTHON) scripts/export_logs.py --results-dir results --out-dir logs

# Regenerate the README banner.
banner:
	$(PYTHON) scripts/make_banner.py

# Everything that can be reproduced without a GPU.
reproduce: test analyze figures logs

# Example single-dataset GPU run.
smoke-gpu:
	$(PYTHON) scripts/run_experiments.py --dataset pneumoniamnist \
		--conditions vanilla fixed:0.05 papc --seeds 1 --output-dir results_repro

clean:
	rm -rf results_repro medmnist_data cifar checkpoints \
		__pycache__ */__pycache__ *.egg-info .pytest_cache
