.PHONY: test smoke lint build
test:
	python -m pytest -q
smoke:
	python -m tap_merge smoke --output runs/smoke --all-ablations
lint:
	python -m ruff check src tests scripts
build:
	python -m build
