.PHONY: install lint format test data-pin data-check

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

# Once, after downloading the two CSVs by hand: record their SHA-256 in config/data.sha256
data-pin:
	uv run python -m nids.data pin

data-check:
	uv run python -m nids.data check
