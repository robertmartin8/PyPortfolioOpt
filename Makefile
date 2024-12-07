.DEFAULT_GOAL := help

SHELL=/bin/bash

UNAME=$(shell uname -s)

.PHONY: install
install:  ## Install a virtual environment with all extras
	@curl -LsSf https://astral.sh/uv/install.sh | sh
	@uv venv
	@uv sync -vv --all-extras
	@echo "Virtual environment created with all extras. Activate with:"
	@echo "source .venv/bin/activate"

.PHONY: fmt
fmt:  ## Run autoformatting and linting
	@uv pip install pre-commit
	@uv run pre-commit install
	@uv run pre-commit run --all-files


.PHONY: test
test: install ## Run tests
	@uv run pytest


.PHONY: clean
clean:  ## Clean up caches and build artifacts
	@git clean -X -d -f


.PHONY: coverage
coverage: install ## test and coverage
	@uv run coverage run --source=cvx/. -m pytest
	@uv run coverage report -m
	@uv run coverage html

	@if [ ${UNAME} == "Darwin" ]; then \
		open htmlcov/index.html; \
	elif [ ${UNAME} == "linux" ]; then \
		xdg-open htmlcov/index.html 2> /dev/null; \
	fi


.PHONY: help
help:  ## Display this help screen
	@echo -e "\033[1mAvailable commands:\033[0m"
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' | sort
