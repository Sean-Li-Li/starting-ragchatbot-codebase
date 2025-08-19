#!/bin/bash
# Complete quality check script - formats, lints, and tests

set -e

echo "🚀 Running complete quality check..."

# Format code
./scripts/format.sh

# Run linting
./scripts/lint.sh

# Run tests  
./scripts/test.sh

echo "🎉 All quality checks passed! Code is ready for commit."