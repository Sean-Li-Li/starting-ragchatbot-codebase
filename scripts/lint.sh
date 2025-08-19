#!/bin/bash
# Code quality checks script - runs linting and formatting checks

set -e

echo "🔍 Running code quality checks..."

echo "  - Checking code formatting with black..."
uv run black --check --diff .

echo "  - Checking import sorting with isort..."
uv run isort --check-only --diff .

echo "  - Running flake8 linter..."
uv run flake8 .

echo "✅ All code quality checks passed!"