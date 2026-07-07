#!/bin/bash
# Quick local start script for DocMind
# Run from the project root: ./run_local.sh

set -e

echo "🔧 Installing dependencies..."
cd backend
pip install -r requirements.txt --quiet

echo ""
echo "🚀 Starting DocMind on http://localhost:8000"
echo "   Press Ctrl+C to stop"
echo ""

uvicorn main:app --reload --port 8000 --host 0.0.0.0
