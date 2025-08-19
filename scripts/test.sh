#!/bin/bash
# Testing script - runs pytest with proper configuration

set -e

echo "🧪 Running tests..."

cd backend
uv run pytest -v

echo "✅ Tests completed!"