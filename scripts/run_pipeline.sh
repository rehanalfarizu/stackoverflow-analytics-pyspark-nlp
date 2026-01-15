#!/bin/bash
# =============================================================================
# Run Full Pipeline Script
# =============================================================================

set -e

echo "=============================================="
echo "  Stack Overflow Analytics Pipeline"
echo "=============================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: Virtual environment not activated"
    echo "Consider running: source venv/bin/activate"
    echo ""
fi

# Create necessary directories
mkdir -p data/raw data/processed data/output logs models

# Run pipeline
echo "Starting pipeline..."
echo ""

python3 main.py --mode full --log-level INFO

echo ""
echo "Pipeline completed!"
echo "=============================================="
