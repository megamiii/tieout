# Thin wrappers around the `tieout` CLI. On Windows without make, run the
# commands on the right directly (inside the activated .venv).

.PHONY: install db ingest test lint

install:
	python -m pip install -e ".[dev]"

db:
	docker compose up -d --wait

ingest: db
	tieout data all

test:
	pytest -q

lint:
	ruff check .
	ruff format --check .
	mypy
