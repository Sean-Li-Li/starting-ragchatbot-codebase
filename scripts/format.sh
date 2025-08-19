#!/bin/bash
# Code formatting script - formats Python code using black and isort

set -e

echo "🎨 Formatting Python code..."

echo "  - Running black..."
uv run black .

echo "  - Running isort..."
uv run isort .

echo "✅ Code formatting complete!"