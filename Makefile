UNAME_S := $(shell uname -s)
EXTRAS := sae_lens
ifeq ($(UNAME_S),Linux)
	EXTRAS := $(EXTRAS),triton,npu
endif

.PHONY: install test lint clean docs-serve docs-build ui-dev ui-build help

install:
	uv sync --all-groups --extra $(EXTRAS)
	cd ui && bun install
	uv run pre-commit install

ui-dev:
	cd ui && bun run dev

ui-build:
	cd ui && bun run build

test:
	uv run pytest

lint:
	uv run ruff format .
	uv run ruff check --fix .
	uv run basedpyright
	cd ui && bun run lint

docs-serve:
	uv run mkdocs serve --livereload

docs-build:
	uv run mkdocs build

clean:
	rm -rf dist build .coverage .pytest_cache .ruff_cache site
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf ui/node_modules

help:
	@echo "Available commands:"
	@echo "  make install       - Install all dependencies (backend, frontend, pre-commit)"
	@echo "  make test          - Run backend tests"
	@echo "  make lint          - Run all linters and formatters (backend, frontend)"
	@echo "  make clean         - Clean build artifacts and node_modules"
	@echo "  make docs-serve    - Serve documentation locally"
	@echo "  make docs-build    - Build documentation"
	@echo "  make ui-dev        - Run ui in development mode"
	@echo "  make ui-build      - Build ui for production"
