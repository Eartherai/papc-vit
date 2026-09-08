.PHONY: install test analyze figures clean

install:
	pip install -e ".[dev]"

test:
	pytest -q

# Reproduce the paper's tables from the checked-in result JSONs (no GPU needed).
analyze:
	python scripts/analyze_results.py --results-dir results

# Regenerate figures into assets/ (no GPU needed).
figures:
	python scripts/make_figures.py --results-dir results --out-dir assets

# Example single-dataset GPU run.
smoke-gpu:
	python scripts/run_experiments.py --dataset pneumoniamnist \
		--conditions vanilla fixed:0.05 papc --seeds 1 --output-dir results_repro

clean:
	rm -rf results_repro medmnist_data cifar __pycache__ */__pycache__ *.egg-info
