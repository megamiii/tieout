# Thin wrappers around the `tieout` CLI. On Windows without make, run the
# commands on the right directly (inside the activated .venv).

.PHONY: install db ingest test lint api web

install:
	python -m pip install -e ".[dev]"
	npm --prefix frontend ci

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
	npm --prefix frontend run lint
	npm --prefix frontend run typecheck

# Development: run both, then open http://localhost:5173 (Vite proxies /api to :8000)
api:
	tieout serve --reload

web:
	npm --prefix frontend run dev
