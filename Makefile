.PHONY: requirements clean install format check_type lint test build

################################################################################
# GLOBALS                                                                      #
################################################################################

SRC = A2N
TEST = tests/
PATH_COV_BADGE = figs/coverage.svg

################################################################################
# COMMANDS                                                                     #
################################################################################

## Export Python Dependencies as requirements.txt
requirements:
	@poetry export --dev -f requirements.txt --without-hashes > requirements.txt

## Delete all compiled Python files
clean:
	@find . -not -path "./.venv/*" -type f -name "*.py[co]"  -exec rm -rf {} 2>/dev/null \;
	@find . -not -path "./.venv/*" -type d -name "__pycache__"  -exec rm -rf {} 2>/dev/null \;
	@find . -not -path "./.venv/*" -type d -name "*.egg-info"  -exec rm -rf {} 2>/dev/null \;
	@find . -not -path "./.venv/*" -type d -name "dist"  -exec rm -rf {} 2>/dev/null \;
	@rm -rf htmlcov .coverage .hypothesis

## Set up the environment using poetry
install:
	@poetry install

## Format code using black
format:
	@poetry run black $(SRC) --config pyproject.toml --check
	@poetry run isort $(SRC) --profile black --check

## type check using mypy
check_type:
	@poetry run mypy $(SRC)

## Lint using pylint
lint:
	@poetry run pylint-fail-under --fail_under 6.0 $(SRC) --exit-zero

## Run tests
test:
	@poetry run pytest $(TEST) --cov=$(SRC) --cov-branch --cov-report term --cov-fail-under 80 --log-cli-level DEBUG
	@poetry run coverage-badge -fo $(PATH_COV_BADGE)

## Build wheel package
build:
	@poetry build

pre-commit: format lint 